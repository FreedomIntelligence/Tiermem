"""
HaluMem数据集加载器

HaluMem数据集结构：
- 每个用户（user）有多个sessions
- 每个session包含：
  - dialogue: 对话列表（user/assistant轮次）
  - memory_points: 记忆点列表（ground truth）
  - questions: 问题列表（用于QA评估）

评估任务：
1. Memory Integrity: 评估记忆完整性（是否遗漏关键记忆点）
2. Memory Accuracy: 评估记忆准确性（提取的记忆是否准确）
3. Memory Update: 评估记忆更新能力（能否正确更新记忆）
4. Question Answering: 端到端QA评估

关键设计：同时提供两种粒度的turns
- "turns": 细粒度，逐句对话（给Mem0/LightMem/MIRIX/RAW+LLM用）
- "session_chunks": 粗粒度，按session组织（给GAM用）

系统可以通过preferred_turns_key属性选择使用哪个粒度
"""
import json
import os
from typing import Dict, Any, List, Optional, Iterator
from pathlib import Path


def load_halumem_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    加载HaluMem JSONL文件
    
    Args:
        jsonl_path: JSONL文件路径
        
    Returns:
        用户数据列表（每个元素是一个用户的所有sessions）
    """
    users = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                user_data = json.loads(line)
                users.append(user_data)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}")
                continue
    return users


def session_to_text(session: Dict[str, Any], session_idx: int) -> str:
    """
    将session转换为文本块（用于粗粒度chunk）
    
    Args:
        session: session字典
        session_idx: session索引
        
    Returns:
        格式化的文本块
    """
    lines = [f"=== SESSION {session_idx} - {session.get('start_time', '')} to {session.get('end_time', '')} ==="]
    lines.append("")  # 空行分隔
    
    dialogue = session.get("dialogue", [])
    for turn in dialogue:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        timestamp = turn.get("timestamp", "")
        lines.append(f"[{timestamp}] {role}: {content}")
        if role == "assistant":
            lines.append("")  # assistant后加空行
    
    return "\n".join(lines).strip()


def iter_sessions(
    data_path: Optional[str] = None,
    split: str = "Medium",
    limit: Optional[int] = None
) -> Iterator[Dict[str, Any]]:
    """
    迭代HaluMem数据集中的sessions
    
    统一接口：返回格式为
    {
        "session_id": str,
        "turns": [...],           # 细粒度：逐句对话（给Mem0/LightMem/MIRIX/RAW+LLM用）
        "session_chunks": [...],  # 粗粒度：按session组织的chunk（给GAM用）
        "qa_pairs": [...],        # 需要回答的问题
        "memory_points": [...],   # Ground truth记忆点（用于评估）
        "meta": {...}             # 元数据
    }
    
    关键设计：
    - 同时提供两种粒度的turns，让不同系统选择最适合自己的粒度
    - 系统可以通过preferred_turns_key属性指定使用哪个粒度
    - GAM使用"session_chunks"（粗粒度），其他系统使用"turns"（细粒度）
    - 按"整用户"聚合：官方流程，记忆不重置
    
    Args:
        data_path: 数据文件路径，默认使用 ./data/HaluMem/halumem_raw/HaluMem-{split}.jsonl
        split: 数据集版本（默认"Medium"）
        limit: 限制返回的用户数量（用于测试）
        
    Yields:
        Session字典，包含turns和session_chunks两种粒度的数据
    """
    if data_path is None:
        data_path = f"./data/HaluMem/halumem_raw/HaluMem-Medium.jsonl"
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"HaluMem data file not found: {data_path}")
    
    # 加载所有用户数据
    users = load_halumem_jsonl(data_path)
    
    if limit:
        users = users[:limit]
    
    print(f"Loaded {len(users)} users from {data_path}")
    
    for user_idx, user_data in enumerate(users):
        uuid = user_data.get("uuid", f"user_{user_idx}")
        persona_info = user_data.get("persona_info", "")
        sessions = user_data.get("sessions", [])
        
        # 按"整用户"聚合：官方流程，记忆不重置
        session_id = f"halumem_{split}_{uuid}"
        
        # 1. 构造session_chunks（粗粒度，给GAM用）
        session_chunks_turns = []
        for session_idx, session in enumerate(sessions):
            if session.get('is_generated_qa_session', False):
                continue
            
            session_text = session_to_text(session, session_idx)
            session_chunks_turns.append({
                "speaker": "system",  # 标记为系统构造的session块
                "text": session_text,   # 完整的session_chunk文本
                "timestamp": session.get("start_time"),  # 保留时间戳信息
                "chunk_idx": session_idx  # 使用chunk_idx与locomo格式一致
            })
        
        # 2. 构造细粒度turns（给Mem0/LightMem/MIRIX/RAW+LLM用）
        fine_grain_turns = []
        for session_idx, session in enumerate(sessions):
            if session.get('is_generated_qa_session', False):
                continue
            
            dialogue = session.get("dialogue", [])
            for turn_idx, turn in enumerate(dialogue):
                fine_grain_turns.append({
                    "speaker": turn.get("role", "unknown"),
                    "text": turn.get("content", ""),
                    "timestamp": turn.get("timestamp", ""),
                    "dialogue_turn": turn.get("dialogue_turn", turn_idx),
                    "session_idx": session_idx,
                })
        
        # 3. 收集QA pairs
        qa_pairs = []
        for session_idx, session in enumerate(sessions):
            if session.get('is_generated_qa_session', False):
                continue
            
            questions = session.get("questions", [])
            for qa_idx, qa in enumerate(questions):
                qa_pairs.append({
                    "query_id": f"{session_id}_s{session_idx}_qa_{qa_idx}",
                    "question": qa.get("question", ""),
                    "ground_truth": qa.get("answer", ""),
                    "difficulty": qa.get("difficulty", ""),
                    "question_type": qa.get("question_type", ""),
                    "evidence": qa.get("evidence", []),
                    "meta": {
                        "session_id": session_id,
                        "user_uuid": uuid,
                        "session_idx": session_idx,
                        "qa_index": qa_idx
                    }
                })
        
        # 4. 收集记忆点
        memory_points_all = []
        for session_idx, session in enumerate(sessions):
            if session.get('is_generated_qa_session', False):
                continue
            
            for mp in session.get("memory_points", []):
                mp_with_sid = dict(mp)
                mp_with_sid["session_idx"] = session_idx
                memory_points_all.append(mp_with_sid)
        
        # 返回格式：同时提供两种粒度的turns
        yield {
            "session_id": session_id,
            "turns": fine_grain_turns,           # 细粒度：逐句对话
            "session_chunks": session_chunks_turns,  # 粗粒度：GAM的session_chunk
            "qa_pairs": qa_pairs,
            "memory_points": memory_points_all,
            "meta": {
                "user_uuid": uuid,
                "user_idx": user_idx,
                "num_sessions": len([s for s in sessions if not s.get('is_generated_qa_session', False)]),
                "num_turns": len(fine_grain_turns),
                "num_chunks": len(session_chunks_turns),
                "num_qa": len(qa_pairs),
                "num_memory_points": len(memory_points_all),
                "persona_info": persona_info[:200] if persona_info else ""
            }
        }


if __name__ == "__main__":
    # 测试代码
    print("Testing HaluMem data loader...")
    count = 0
    for session in iter_sessions(limit=10):
        count += 1
        print(f"\nSession {count}:")
        print(f"  session_id: {session['session_id']}")
        print(f"  num_turns (fine-grain): {len(session['turns'])}")
        print(f"  num_session_chunks (coarse-grain): {len(session.get('session_chunks', []))}")
        print(f"  num_qa_pairs: {len(session['qa_pairs'])}")
        print(f"  num_memory_points: {len(session.get('memory_points', []))}")
        os.makedirs("./tmp/halumem_debug", exist_ok=True)
        with open(f"./tmp/halumem_debug/session_{count}.json", "w") as f:
            json.dump(session, f, indent=4)
        if session['turns']:
            print(f"  first fine-grain turn: {session['turns'][0]['speaker']}: {session['turns'][0]['text'][:50]}...")
        if session.get('session_chunks'):
            print(f"  first session_chunk: {session['session_chunks'][0]['speaker']}: {session['session_chunks'][0]['text'][:80]}...")
        if session['qa_pairs']:
            print(f"  first question: {session['qa_pairs'][0]['question']}")
            print(f"  first answer: {session['qa_pairs'][0]['ground_truth']}")
    print(f"\nTotal sessions: {count}")
