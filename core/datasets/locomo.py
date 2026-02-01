"""
LoCoMo数据集加载器

关键设计：同时提供两种粒度的turns
- "turns": 细粒度，逐句对话（给Mem0/LightMem/MIRIX/RAW+LLM用）
- "session_chunks": 粗粒度，官方GAM的session_chunk（给GAM用）

系统可以通过preferred_turns_key属性选择使用哪个粒度
"""
import json
import re
from typing import Dict, Any, List, Optional, Iterator, Tuple
from pathlib import Path
import os

def load_locomo_json(json_path: str) -> List[Dict[str, Any]]:
    """
    加载LoCoMo JSON文件
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        样本列表
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    if isinstance(data, list):
        return data
    raise ValueError("Unrecognized LoCoMo JSON shape. Expect a list or {'samples': [...]}.")


def extract_sessions(conv_obj: Dict[str, Any]) -> List[Tuple[int, str, List[Dict[str, Any]], Optional[str]]]:
    """
    从conversation对象中提取sessions（官方GAM的方式）
    
    Returns:
        List of (idx, timestamp, turns, session_summary)
    """
    sessions: List[Tuple[int, str, List[Dict[str, Any]], Optional[str]]] = []
    for k, v in conv_obj.items():
        m = re.match(r'^session_(\d+)$', k)
        if not (m and isinstance(v, list)):
            continue
        original_idx = int(m.group(1))
        idx = original_idx - 1
        ts = conv_obj.get(f"session_{original_idx}_date_time", "")
        ssum = conv_obj.get(f"session_{original_idx}_summary", None)
        sessions.append((
            idx, 
            ts, 
            v, 
            ssum if isinstance(ssum, str) and ssum.strip() else None
        ))
    sessions.sort(key=lambda x: x[0])
    return sessions


def session_to_text(idx: int, ts: str, turns: List[Dict[str, Any]], session_summary: Optional[str]) -> str:
    """
    将session转换为文本块（官方GAM的方式）
    
    关键：包含时间信息、speaker、dia_id等，格式与官方一致
    """
    lines = [f"=== SESSION {idx} - Dialogue Time(available to answer questions): {ts} ==="]
    lines.append("")  # 空行分隔
    
    for turn in turns:
        speaker = turn.get("speaker", "Unknown")
        dia_id = turn.get("dia_id", "")
        text = turn.get("text", "")
        blip_caption = turn.get("blip_caption", "")
        if blip_caption:    
            lines.append(f"{speaker} ({dia_id}): {text} (blip_caption: {blip_caption})")
        else:
            lines.append(f"{speaker} ({dia_id}): {text}")
    
    if session_summary:
        lines.append("")
        lines.append(f"Session {idx} summary: {session_summary}")
    
    return "\n".join(lines).strip()


def build_session_chunks_for_sample(sample: Dict[str, Any]) -> List[str]:
    """
    为sample构建session chunks（官方GAM的方式）
    
    这是关键函数：将整个sample的对话按session组织成文本块
    """
    conv = sample.get("conversation", {})
    sessions = extract_sessions(conv)
    chunks: List[str] = []
    for idx, ts, turns, ssum in sessions:
        chunks.append(session_to_text(idx, ts, turns, ssum))
    return chunks


def iter_sessions(
    data_path: Optional[str] = None,
    split: str = "test",
    limit: Optional[int] = None
) -> Iterator[Dict[str, Any]]:
    """
    迭代LoCoMo数据集中的sessions
    
    统一接口：返回格式为
    {
        "session_id": str,
        "turns": [...],           # 细粒度：逐句对话（给Mem0/LightMem/MIRIX/RAW+LLM用）
        "session_chunks": [...],  # 粗粒度：官方GAM的session_chunk（给GAM用）
        "qa_pairs": [...]         # 需要回答的问题
    }
    
    关键设计：
    - 同时提供两种粒度的turns，让不同系统选择最适合自己的粒度
    - 系统可以通过preferred_turns_key属性指定使用哪个粒度
    - GAM使用"session_chunks"（官方方式），其他系统使用"turns"（细粒度）
    
    Args:
        data_path: 数据文件路径，默认使用 ./data/locomo/locomo10.json
        split: 数据集split（LoCoMo通常只有一个文件）
        limit: 限制返回的样本数量（用于测试）
        
    Yields:
        Session字典，包含turns和session_chunks两种粒度的数据
    """
    if data_path is None:
        data_path = "./data/locomo/locomo10.json"
    
    samples = load_locomo_json(data_path)
    
    if limit:
        samples = samples[:limit]
    
    for sample_idx, sample in enumerate(samples):
        sample_id = sample.get("sample_id", f"sample_{sample_idx}")
        conv = sample.get("conversation", {})
        sessions_meta = extract_sessions(conv)
        
        # 1. 构造session_chunks（粗粒度，给GAM用）
        session_chunks = build_session_chunks_for_sample(sample)
        # 转换为pseudo-turns格式，每个chunk作为一个turn
        session_chunks_turns = []
        for chunk_idx, chunk_text in enumerate(session_chunks):
            session_chunks_turns.append({
                "speaker": "system",  # 标记为系统构造的session块
                "text": chunk_text,   # 完整的session_chunk文本
                "timestamp": None,    # timestamp已在chunk文本中
                "chunk_idx": chunk_idx
            })
        
        # 2. 构造细粒度turns（给Mem0/LightMem/MIRIX/RAW+LLM用）
        fine_grain_turns = []
        for idx, ts, turns, ssum in sessions_meta:
            for turn in turns:
                blip_caption = turn.get("blip_caption", "")
                text = turn.get("text", "")
                if blip_caption:
                    text = f"(blip_caption: {blip_caption}) text:{text} "
                else:
                    text = text
                fine_grain_turns.append({
                    "speaker": turn.get("speaker", "Unknown"),
                    "text": text,
                    "timestamp": ts,
                    "dia_id": turn.get("dia_id", ""),
                    "session_idx": idx
                })
        
        # 3. 收集QA pairs（过滤category==5，因为官方不计入评估）
        qa_pairs = []
        for qa_idx, qa in enumerate(sample.get("qa", [])):
            category = qa.get("category")
            # 跳过category==5（官方不计入评估）
            if category == 5:
                continue
            qa_pairs.append({
                "query_id": f"{sample_id}_qa_{qa_idx}",
                "question": qa.get("question", ""),
                "ground_truth": qa.get("answer", ""),
                "category": category,
                "evidence": qa.get("evidence", []),
                "meta": {
                    "sample_id": sample_id,
                    "qa_index": qa_idx,
                    "category": category  # 确保category在meta中，供answer使用
                }
            })
        
        # 返回格式：同时提供两种粒度的turns
        yield {
            "session_id": sample_id,
            "turns": fine_grain_turns,           # 细粒度：逐句对话
            "session_chunks": session_chunks_turns,  # 粗粒度：GAM的session_chunk
            "qa_pairs": qa_pairs,
            "meta": {
                "num_sessions": len(sessions_meta),
                "num_fine_turns": len(fine_grain_turns),
                "num_chunks": len(session_chunks_turns),
                "num_qa": len(qa_pairs)
            }
        }


if __name__ == "__main__":
    # 测试代码
    print("Testing LoCoMo data loader...")
    count = 0
    for session in iter_sessions(limit=10):
        count += 1
        print(f"\nSession {count}:")
        print(f"  session_id: {session['session_id']}")
        print(f"  num_turns (fine-grain): {len(session['turns'])}")
        print(f"  num_session_chunks (coarse-grain): {len(session.get('session_chunks', []))}")
        print(f"  num_qa_pairs: {len(session['qa_pairs'])}")
        if session['turns']:
            print(f"  first fine-grain turn: {session['turns'][0]['speaker']}: {session['turns'][0]['text']}...")
        if session.get('session_chunks'):
            print(f"  first session_chunk: {session['session_chunks'][0]['speaker']}: {session['session_chunks'][0]['text'][:80]}...")
        if session['qa_pairs']:
            print(f"  first question: {session['qa_pairs'][0]['question']}")
        os.makedirs("./core/datasets/tmp_locomo", exist_ok=True)
        with open(f"./core/datasets/tmp_locomo/session_{count}.json", "w") as f:
            json.dump(session, f, indent=4)
    print(f"\nTotal sessions: {count}")

