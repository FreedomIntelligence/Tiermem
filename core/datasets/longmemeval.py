"""
LongMemEval数据集加载器

参考实现：
- LongMemEval/src/generation/run_generation.py
- LongMemEval/src/retrieval/run_retrieval.py
- EverMemOS/evaluation/src/converters/longmemeval_converter.py

数据格式：
- 每个样本包含一个问题及其相关的历史对话（haystack_sessions）
- haystack_sessions: 包含多个session，每个session包含多个turn（role和content）
- 问题类型包括：single-session-user, single-session-assistant, multi-session, 
  temporal-reasoning, knowledge-update, single-session-preference

关键设计：同时提供两种粒度的turns
- "turns": 细粒度，逐turn对话（给Mem0/LightMem/MIRIX/RAW+LLM用）
- "session_chunks": 粗粒度，按session组织的chunk（给GAM用）

关键特性（参考EverMemOS实现）：
- 标记包含答案的turns（has_answer字段）
- 生成evidence列表（格式：D{session_idx}:{turn_idx}）
- 在QA pairs中包含evidence和category信息

系统可以通过preferred_turns_key属性选择使用哪个粒度

转换功能：
- convert_to_locomo_style: 将LongMemEval格式转换为LoCoMo格式（参考EverMemOS实现）
"""
import json
import re
from typing import Dict, Any, List, Optional, Iterator, Tuple
from pathlib import Path
from datetime import datetime
import os

# 尝试导入 tiktoken 用于 token 级别的 chunking
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None


def load_longmemeval_json(json_path: str) -> List[Dict[str, Any]]:
    """
    加载LongMemEval JSON文件
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        样本列表
    """
    # 尝试作为JSON数组加载
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "samples" in data:
            return data["samples"]
        raise ValueError("Unrecognized LongMemEval JSON shape. Expect a list or {'samples': [...]}.")
    except json.JSONDecodeError:
        # 如果整个文件不是有效JSON，尝试按行读取（JSONL格式）
        samples = []
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return samples


def session_to_text(session_id: str, session_date: str, turns: List[Dict[str, Any]]) -> str:
    """
    将session转换为文本块（类似GAM的方式）
    
    Args:
        session_id: session ID
        session_date: session日期
        turns: session中的turns列表
        
    Returns:
        格式化的文本块
    """
    dia_id = turns[0].get("dia_id", "") if turns else ""
    lines = [f"=== SESSION {session_id} - Date: {session_date} - Dia ID: {dia_id} ==="]
    lines.append("")  # 空行分隔
    
    for turn in turns:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        if content:
            lines.append(f"{role}: {content}")
    
    return "\n".join(lines).strip()


def _get_encoding(encoding_model: str = "gpt-4o-mini"):
    """获取 tiktoken 编码器"""
    if not TIKTOKEN_AVAILABLE:
        return None
    
    try:
        return tiktoken.encoding_for_model(encoding_model)
    except Exception:
        try:
            return tiktoken.get_encoding(encoding_model)
        except Exception:
            return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str, encoding_model: str = "gpt-4o-mini") -> int:
    """
    计算文本的 token 数量
    
    Args:
        text: 要计算的文本
        encoding_model: 用于 token 编码的模型名称
        
    Returns:
        token 数量
    """
    encoding = _get_encoding(encoding_model)
    if encoding is None:
        # 如果没有 tiktoken，粗略估算（1 token ≈ 4 字符）
        return len(text) // 4
    
    try:
        tokens = encoding.encode(text, allowed_special={"<|endoftext|>"})
        return len(tokens)
    except Exception:
        try:
            tokens = encoding.encode(text, disallowed_special=())
            return len(tokens)
        except Exception:
            # 如果都失败，回退到字符估算
            return len(text) // 4


def _get_encoding(encoding_model: str = "gpt-4o-mini"):
    """获取 tiktoken 编码器"""
    if not TIKTOKEN_AVAILABLE:
        return None
    
    try:
        return tiktoken.encoding_for_model(encoding_model)
    except Exception:
        try:
            return tiktoken.get_encoding(encoding_model)
        except Exception:
            return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str, encoding_model: str = "gpt-4o-mini") -> int:
    """
    计算文本的 token 数量
    
    Args:
        text: 要计算的文本
        encoding_model: 用于 token 编码的模型名称
        
    Returns:
        token 数量
    """
    encoding = _get_encoding(encoding_model)
    if encoding is None:
        # 如果没有 tiktoken，粗略估算（1 token ≈ 4 字符）
        return len(text) // 4
    
    try:
        tokens = encoding.encode(text, allowed_special={"<|endoftext|>"})
        return len(tokens)
    except Exception:
        try:
            tokens = encoding.encode(text, disallowed_special=())
            return len(tokens)
        except Exception:
            # 如果都失败，回退到字符估算
            return len(text) // 4


def _format_turn_locomo(turn: Dict[str, Any]) -> str:
    """将单个 turn 格式化为文本（LoCoMo 格式）"""
    speaker = turn.get("speaker", "Unknown")
    dia_id = turn.get("dia_id", "")
    text = turn.get("text", "")
    blip_caption = turn.get("blip_caption", "")
    if blip_caption:
        return f"{speaker} ({dia_id}): {text} (blip_caption: {blip_caption})"
    else:
        return f"{speaker} ({dia_id}): {text}"


def _format_turn_original(turn: Dict[str, Any]) -> str:
    """将单个 turn 格式化为文本（原始 LongMemEval 格式）"""
    role = turn.get("role", "unknown")
    content = turn.get("content", "")
    return f"{role}: {content}" if content else ""


def _chunk_sessions_with_turns_locomo(
    sessions: List[Tuple[int, str, List[Dict[str, Any]], Optional[str]]],
    chunk_size: int,
    encoding_model: str = "gpt-4o-mini"
) -> List[str]:
    """
    按 session 边界切分，session 内部按对话合并
    
    逻辑：
    1. 保持 session 边界（每个 chunk 只包含一个 session 的内容）
    2. 在 session 内部，按对话（turn）合并
    3. 如果当前 chunk 加上下一个 turn 不超过 chunk_size，就合并
    4. 如果超过，就创建新的 chunk（但仍在同一个 session 内）
    
    Args:
        sessions: session 列表，每个元素为 (idx, timestamp, turns, session_summary)
        chunk_size: 每个 chunk 的最大 token 数量
        encoding_model: 用于 token 编码的模型名称
        
    Returns:
        chunk 文本列表
    """
    chunks: List[str] = []
    
    for session_idx, ts, turns, ssum in sessions:
        # Session 头部
        session_header = f"=== SESSION {session_idx} - Dialogue Time(available to answer questions): {ts} ===\n"
        header_tokens = _count_tokens(session_header, encoding_model)
        
        # 当前 chunk 的内容和 token 数
        current_chunk_lines: List[str] = []
        current_chunk_tokens = header_tokens
        
        for turn_idx, turn in enumerate(turns):
            turn_text = _format_turn_locomo(turn)
            turn_tokens = _count_tokens(turn_text, encoding_model)
            
            # 如果当前 chunk 为空，先添加 session 头部
            if not current_chunk_lines:
                current_chunk_lines.append(session_header.rstrip())
                current_chunk_tokens = header_tokens
            
            # 检查加上这个 turn 是否会超过限制
            if current_chunk_tokens + turn_tokens <= chunk_size:
                # 可以合并
                current_chunk_lines.append(turn_text)
                current_chunk_tokens += turn_tokens
            else:
                # 超过限制，先保存当前 chunk（不添加 summary，因为可能还有更多 turns）
                if current_chunk_lines:
                    chunks.append("\n".join(current_chunk_lines))
                
                # 开始新的 chunk（仍属于同一个 session）
                current_chunk_lines = [session_header.rstrip(), turn_text]
                current_chunk_tokens = header_tokens + turn_tokens
        
        # 保存最后一个 chunk，并添加 session summary（如果有）
        if current_chunk_lines:
            if ssum:
                current_chunk_lines.append("")
                current_chunk_lines.append(f"Session {session_idx} summary: {ssum}")
            chunks.append("\n".join(current_chunk_lines))
    
    return chunks


def _chunk_sessions_with_turns_original(
    haystack_dates: List[str],
    haystack_session_ids: List[str],
    haystack_sessions: List[List[Dict[str, Any]]],
    chunk_size: int,
    encoding_model: str = "gpt-4o-mini"
) -> List[str]:
    """
    按 session 边界切分，session 内部按对话合并（原始 LongMemEval 格式）
    
    逻辑同 _chunk_sessions_with_turns_locomo
    """
    chunks: List[str] = []
    
    for session_idx, (session_date, session_id, session_entry) in enumerate(
        zip(haystack_dates, haystack_session_ids, haystack_sessions)
    ):
        if not isinstance(session_entry, list):
            session_entry = [session_entry]
        
        # Session 头部
        dia_id = session_entry[0].get("dia_id", "") if session_entry else ""
        session_header = f"=== SESSION {session_id} - Date: {session_date} - Dia ID: {dia_id} ===\n"
        header_tokens = _count_tokens(session_header, encoding_model)
        
        # 当前 chunk 的内容和 token 数
        current_chunk_lines: List[str] = []
        current_chunk_tokens = 0
        
        for turn in session_entry:
            turn_text = _format_turn_original(turn)
            if not turn_text:  # 跳过空内容
                continue
            
            turn_tokens = _count_tokens(turn_text, encoding_model)
            
            # 如果当前 chunk 为空，先添加 session 头部
            if not current_chunk_lines:
                current_chunk_lines.append(session_header.rstrip())
                current_chunk_tokens = header_tokens
            
            # 检查加上这个 turn 是否会超过限制
            if current_chunk_tokens + turn_tokens <= chunk_size:
                # 可以合并
                current_chunk_lines.append(turn_text)
                current_chunk_tokens += turn_tokens
            else:
                # 超过限制，先保存当前 chunk
                if current_chunk_lines:
                    chunks.append("\n".join(current_chunk_lines))
                
                # 开始新的 chunk（仍属于同一个 session）
                current_chunk_lines = [session_header.rstrip(), turn_text]
                current_chunk_tokens = header_tokens + turn_tokens
        
        # 保存最后一个 chunk
        if current_chunk_lines:
            chunks.append("\n".join(current_chunk_lines))
    
    return chunks


def _chunk_text_by_tokens(text: str, chunk_size: int, encoding_model: str = "gpt-4o-mini") -> List[str]:
    """
    按 token 数量切分文本（参考 memr3 的实现）
    
    Args:
        text: 要切分的文本
        chunk_size: 每个 chunk 的 token 数量（-1 表示不切分）
        encoding_model: 用于 token 编码的模型名称
        
    Returns:
        切分后的文本列表
    """
    if chunk_size == -1:
        return [text]
    
    encoding = _get_encoding(encoding_model)
    if encoding is None:
        # 如果没有 tiktoken，回退到按字符数粗略估算（1 token ≈ 4 字符）
        char_size = chunk_size * 4
        chunks = []
        for i in range(0, len(text), char_size):
            chunks.append(text[i:i + char_size])
        return chunks
    
    # 编码时允许特殊 token，避免遇到 <|endoftext|> 等特殊 token 时报错
    try:
        tokens = encoding.encode(text, allowed_special={"<|endoftext|>"})
    except Exception:
        # 如果仍然失败，尝试禁用所有特殊 token 检查
        tokens = encoding.encode(text, disallowed_special=())
    
    chunks: List[str] = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i : i + chunk_size]
        chunk = encoding.decode(chunk_tokens)
        chunks.append(chunk)
    
    return chunks


def build_session_chunks_for_sample(sample: Dict[str, Any], chunk_size: Optional[int] = None) -> List[str]:
    """
    为sample构建session chunks（粗粒度，给GAM用）
    
    Args:
        sample: 样本字典
        chunk_size: 如果提供，按 session 边界切分，session 内部按对话合并（不超过 chunk_size tokens）；
                   如果为 None，按 session 边界切分（每个 session 一个 chunk）
        
    Returns:
        session chunks文本列表
    """
    haystack_dates = sample.get("haystack_dates", [])
    haystack_session_ids = sample.get("haystack_session_ids", [])
    haystack_sessions = sample.get("haystack_sessions", [])
    
    # 如果指定了 chunk_size，按 session 边界切分，session 内部按对话合并
    if chunk_size is not None:
        return _chunk_sessions_with_turns_original(
            haystack_dates, haystack_session_ids, haystack_sessions, chunk_size
        )
    
    # 否则按 session 边界切分（原有逻辑，每个 session 一个 chunk）
    chunks: List[str] = []
    for session_date, session_id, session_entry in zip(haystack_dates, haystack_session_ids, haystack_sessions):
        if isinstance(session_entry, list):
            chunks.append(session_to_text(session_id, session_date, session_entry))
        else:
            # 如果session_entry不是列表，尝试转换
            chunks.append(session_to_text(session_id, session_date, [session_entry]))
    
    return chunks


def _is_locomo_format(sample: Dict[str, Any]) -> bool:
    """
    检测数据样本是否是 LoCoMo 格式
    
    LoCoMo 格式特征：
    - 有 "conversation" 字段，且包含 "session_0", "session_0_date_time" 等
    - 有 "qa" 字段，且是列表格式
    
    Args:
        sample: 数据样本
        
    Returns:
        True 如果是 LoCoMo 格式，False 如果是原始 LongMemEval 格式
    """
    # 检查是否有 conversation 字段且包含 session_* 键
    conv = sample.get("conversation", {})
    if isinstance(conv, dict):
        # 检查是否有 session_0, session_1 等键
        has_session_keys = any(re.match(r'^session_\d+$', k) for k in conv.keys())
        if has_session_keys:
            return True
    
    # 检查是否有原始 LongMemEval 格式的特征
    if "haystack_sessions" in sample and "question" in sample:
        return False
    
    # 默认假设是 LoCoMo 格式（如果都不匹配，可能是其他格式）
    return True


def extract_sessions(conv_obj: Dict[str, Any]) -> List[Tuple[int, str, List[Dict[str, Any]], Optional[str]]]:
    """
    从conversation对象中提取sessions（参考 locomo.py 的实现）
    
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


def session_to_text_locomo(idx: int, ts: str, turns: List[Dict[str, Any]], session_summary: Optional[str]) -> str:
    """
    将session转换为文本块（参考 locomo.py 的实现，用于 LoCoMo 格式）
    
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


def build_session_chunks_for_locomo_sample(sample: Dict[str, Any], chunk_size: Optional[int] = None) -> List[str]:
    """
    为 LoCoMo 格式的 sample 构建 session chunks（参考 locomo.py 的实现）
    
    Args:
        sample: 样本字典
        chunk_size: 如果提供，按 session 边界切分，session 内部按对话合并（不超过 chunk_size tokens）；
                   如果为 None，按 session 边界切分（每个 session 一个 chunk）
    """
    conv = sample.get("conversation", {})
    sessions = extract_sessions(conv)
    
    # 如果指定了 chunk_size，按 session 边界切分，session 内部按对话合并
    if chunk_size is not None:
        return _chunk_sessions_with_turns_locomo(sessions, chunk_size)
    
    # 否则按 session 边界切分（原有逻辑，每个 session 一个 chunk）
    chunks: List[str] = []
    for idx, ts, turns, ssum in sessions:
        chunks.append(session_to_text_locomo(idx, ts, turns, ssum))
    return chunks


def iter_sessions(
    data_path: Optional[str] = None,
    split: str = "test",
    limit: Optional[int] = None,
    use_locomo_format: bool = True,
    chunk_size: Optional[int] = 1000
) -> Iterator[Dict[str, Any]]:
    """
    迭代LongMemEval数据集中的sessions
    
    统一接口：返回格式为
    {
        "session_id": str,
        "turns": [...],           # 细粒度：逐turn对话（给Mem0/LightMem/MIRIX/RAW+LLM用）
        "session_chunks": [...],  # 粗粒度：按session组织的chunk（给GAM用）
        "qa_pairs": [...]         # 需要回答的问题
    }
    
    关键设计：
    - 同时提供两种粒度的turns，让不同系统选择最适合自己的粒度
    - 系统可以通过preferred_turns_key属性指定使用哪个粒度
    - GAM使用"session_chunks"（粗粒度），其他系统使用"turns"（细粒度）
    - 支持自动检测数据格式：如果是原始 LongMemEval 格式，会自动转换为 LoCoMo 格式处理
    - 如果 use_locomo_format=True，会先转换为 LoCoMo 格式，然后按照 locomo.py 的方式处理
    - 如果指定 chunk_size，session_chunks 将按 token 数量切分，而不是按 session 边界
    
    Args:
        data_path: 数据文件路径，默认使用 ./data/LongMemEval/longmemeval_s_cleaned.json
        split: 数据集split（LongMemEval通常只有一个文件）
        limit: 限制返回的样本数量（用于测试）
        use_locomo_format: 如果为 True，会将原始格式转换为 LoCoMo 格式后处理（参考 locomo.py）
        chunk_size: 如果提供，session_chunks 将按此 token 数量切分；如果为 None，按 session 边界切分
        
    Yields:
        Session字典，包含turns和session_chunks两种粒度的数据
    """
    if data_path is None:
        data_path = "./data/LongMemEval/longmemeval_s_cleaned.json"
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"LongMemEval data file not found: {data_path}")
    
    samples = load_longmemeval_json(data_path)
    
    if limit:
        samples = samples[:limit]
    
    # 检测第一个样本的格式，并决定是否需要转换
    converted_to_locomo = False
    if samples:
        is_locomo = _is_locomo_format(samples[0])
        if not is_locomo and use_locomo_format:
            # 转换为 LoCoMo 格式（在内存中）
            print(f"🔄 检测到原始 LongMemEval 格式，正在转换为 LoCoMo 格式...")
            samples = convert_to_locomo_style(samples)
            print(f"✅ 转换完成，共 {len(samples)} 条数据")
            converted_to_locomo = True
            is_locomo = True  # 转换后都是 LoCoMo 格式
    
    for sample_idx, sample in enumerate(samples):
        # 如果已经转换过，所有样本都是 LoCoMo 格式；否则检测当前样本的格式
        if converted_to_locomo:
            is_locomo = True
        else:
            is_locomo = _is_locomo_format(sample)
        
        if is_locomo:
            # LoCoMo 格式处理（参考 locomo.py）
            # 从 qa 中获取 question_id 作为 session_id（使用原来的 question_id）
            qa_list = sample.get("qa", [])
            question_id = None
            if qa_list and len(qa_list) > 0:
                question_id = qa_list[0].get("question_id", None)
            
            # 如果没有 question_id，尝试使用 sample_id 或生成默认值
            if not question_id:
                question_id = sample.get("sample_id", f"question_{sample_idx}")
            
            conv = sample.get("conversation", {})
            sessions_meta = extract_sessions(conv)
            
            # 1. 构造session_chunks（粗粒度，给GAM用）
            session_chunks = build_session_chunks_for_locomo_sample(sample, chunk_size=chunk_size)
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
            
            # 3. 收集QA pairs
            qa_pairs = []
            for qa_idx, qa in enumerate(qa_list):
                category = qa.get("category")
                # 注意：LongMemEval 没有 category==5 的过滤规则
                qa_question_id = qa.get("question_id", question_id)
                qa_pairs.append({
                    "query_id": qa_question_id,
                    "question": qa.get("question", ""),
                    "ground_truth": qa.get("answer", ""),
                    "category": category,
                    "evidence": qa.get("evidence", []),
                    "meta": {
                        "sample_id": sample.get("sample_id", question_id),
                        "qa_index": qa_idx,
                        "category": category,
                        "question_id": qa_question_id
                    }
                })
            
            # 返回格式：同时提供两种粒度的turns
            yield {
                "session_id": question_id,  # 使用原来的 question_id
                "turns": fine_grain_turns,           # 细粒度：逐句对话
                "session_chunks": session_chunks_turns,  # 粗粒度：GAM的session_chunk
                "qa_pairs": qa_pairs,
                "meta": {
                    "question_id": question_id,  # 添加 question_id 到 meta
                    "num_sessions": len(sessions_meta),
                    "num_fine_turns": len(fine_grain_turns),
                    "num_chunks": len(session_chunks_turns),
                    "num_qa": len(qa_pairs)
                }
            }
        else:
            # 原始 LongMemEval 格式处理（保持原有逻辑）
            question_id = sample.get("question_id", f"question_{sample_idx}")
            question_type = sample.get("question_type", "unknown")
            question = sample.get("question", "")
            question_date = sample.get("question_date", "")
            answer = sample.get("answer", "")
            answer_session_ids = sample.get("answer_session_ids", [])
            
            haystack_dates = sample.get("haystack_dates", [])
            haystack_session_ids = sample.get("haystack_session_ids", [])
            haystack_sessions = sample.get("haystack_sessions", [])
            
            # 1. 构造session_chunks（粗粒度，给GAM用）
            session_chunks = build_session_chunks_for_sample(sample, chunk_size=chunk_size)
            # 转换为pseudo-turns格式，每个chunk作为一个turn
            session_chunks_turns = []
            for chunk_idx, (chunk_text, session_id, session_date) in enumerate(
                zip(session_chunks, haystack_session_ids, haystack_dates)
            ):
                session_chunks_turns.append({
                    "speaker": "system",  # 标记为系统构造的session块
                    "text": chunk_text,   # 完整的session_chunk文本
                    "timestamp": session_date,
                    "session_id": session_id,
                    "chunk_idx": chunk_idx
                })
            
            # 2. 构造细粒度turns（给Mem0/LightMem/MIRIX/RAW+LLM用）
            # 首先标记哪些session包含答案（参考EverMemOS的实现）
            evidence_session_idx = []
            for idx, session_id in enumerate(haystack_session_ids):
                if session_id in answer_session_ids:
                    evidence_session_idx.append(idx)
            
            fine_grain_turns = []
            evidence = []  # 收集evidence信息，格式为 "D{session_idx}:{turn_idx}"
            
            for session_idx, (session_date, session_id, session_entry) in enumerate(
                zip(haystack_dates, haystack_session_ids, haystack_sessions)
            ):
                if not isinstance(session_entry, list):
                    session_entry = [session_entry]
                
                # 标记当前session是否包含答案
                session_has_answer = session_idx in evidence_session_idx
                
                for turn_idx, turn in enumerate(session_entry):
                    role = turn.get("role", "unknown")
                    content = turn.get("content", "")
                    
                    # 检查当前turn是否包含答案（如果session包含答案，则标记）
                    # 注意：LongMemEval原始数据可能没有turn级别的has_answer标记
                    # 这里我们基于session级别来标记
                    has_answer = session_has_answer
                    
                    # 如果原始数据中有has_answer字段，使用它
                    if "has_answer" in turn:
                        has_answer = turn["has_answer"]
                    
                    # 如果包含答案，添加到evidence列表
                    if has_answer:
                        evidence.append(f"D{session_idx}:{turn_idx}")
                    
                    fine_grain_turns.append({
                        "speaker": role,
                        "text": content,
                        "timestamp": session_date,
                        "session_id": session_id,
                        "session_idx": session_idx,
                        "turn_idx": turn_idx,
                        "has_answer": has_answer,  # 标记是否包含答案
                        "dia_id": f"D{session_idx}:{turn_idx}"  # 添加dia_id，与EverMemOS格式一致
                    })
            
            # 3. 收集QA pairs（参考EverMemOS的实现，添加evidence信息）
            # 确保answer转换为字符串（因为有些答案是整数）
            answer_str = str(answer) if answer is not None else ""
            
            qa_pairs = [{
                "query_id": question_id,
                "question": question,
                "ground_truth": answer_str,  # 统一转换为字符串
                "question_type": question_type,
                "question_date": question_date,
                "answer_session_ids": answer_session_ids,
                "evidence": evidence,  # 添加evidence列表
                "category": question_type,  # 使用question_type作为category
                "meta": {
                    "question_id": question_id,
                    "question_type": question_type,
                    "question_date": question_date,
                    "evidence_session_indices": evidence_session_idx,
                    "original_answer": answer  # 保留原始答案（可能是int或str）
                }
            }]
            
            # 返回格式：同时提供两种粒度的turns
            yield {
                "session_id": question_id,
                "turns": fine_grain_turns,           # 细粒度：逐turn对话
                "session_chunks": session_chunks_turns,  # 粗粒度：按session组织的chunk
                "qa_pairs": qa_pairs,
                "meta": {
                    "question_id": question_id,
                    "question_type": question_type,
                    "num_sessions": len(haystack_sessions),
                    "num_fine_turns": len(fine_grain_turns),
                    "num_chunks": len(session_chunks_turns),
                    "num_qa": len(qa_pairs)
                }
            }


def convert_time_format(input_str: str) -> str:
    """
    转换时间格式：从 "YYYY/MM/DD (Day) HH:MM" 转换为 "H:MM am/pm on D Month, YYYY"
    
    参考 EverMemOS 实现：
    - 输入格式: %Y/%m/%d (%a) %H:%M
    - 输出格式: %-I:%M %p on %-d %B, %Y (然后转换为小写并首字母大写月份)
    
    Args:
        input_str: 输入时间字符串，格式如 "2023/03/15 (Wed) 14:30"
        
    Returns:
        转换后的时间字符串，格式如 "2:30 pm on 15 March, 2023"
    """
    # 输入格式: %Y: year, %m: month, %d: day, %a: weekday abbr, %H: 24-hour, %M: minute
    input_format = "%Y/%m/%d (%a) %H:%M"
    
    try:
        # 解析输入字符串为 datetime 对象
        dt_object = datetime.strptime(input_str, input_format)
        
        # 输出格式: %-I: 12-hour (no leading zero), %M: minute, %p: AM/PM, 
        #           %-d: day (no leading zero), %B: full month name, %Y: year
        # 注意：Python的strftime不支持%-I和%-d，需要手动处理
        hour_12 = dt_object.hour % 12
        if hour_12 == 0:
            hour_12 = 12
        am_pm = "am" if dt_object.hour < 12 else "pm"
        day = dt_object.day
        month = dt_object.strftime("%B")
        year = dt_object.year
        
        # 格式化：H:MM am/pm on D Month, YYYY
        formatted_string = f"{hour_12}:{dt_object.minute:02d} {am_pm} on {day} {month}, {year}"
        
        return formatted_string
    except ValueError as e:
        # 如果解析失败，返回原始字符串
        print(f"Warning: Failed to parse time format '{input_str}': {e}")
        return input_str


def convert_to_locomo_style(
    lmeval_data: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    将 LongMemEval 格式转换为 LoCoMo 格式
    
    参考 EverMemOS 实现 (longmemeval_converter.py)
    
    转换要点：
    1. 时间格式转换：从 "YYYY/MM/DD (Day) HH:MM" 转换为 "H:MM am/pm on D Month, YYYY"
    2. 对话结构：将 haystack_sessions 转换为 LoCoMo 的 conversation 格式
       - conversation["session_0"], conversation["session_0_date_time"], ...
    3. QA 结构：转换为 LoCoMo 的 qa 格式
       - 包含 question_id, question, answer, evidence, category
    4. Speaker 命名：为每个 speaker 添加 question_id 后缀
    5. Evidence 标记：生成 dia_id 格式为 D{session_idx}:{turn_idx}
    
    Args:
        lmeval_data: LongMemEval 原始数据列表
        output_path: 可选，如果提供则保存转换后的数据到文件
        
    Returns:
        LoCoMo 格式的数据列表
    """
    locomo_style_data = []
    
    for data in lmeval_data:
        data_dict = {
            "qa": [],
            "conversation": {}
        }
        
        question_id = data.get("question_id", "")
        
        # 找到包含答案的 session 索引
        evidence_session_idx = []
        haystack_session_ids = data.get("haystack_session_ids", [])
        answer_session_ids = data.get("answer_session_ids", [])
        
        for idx, session_id in enumerate(haystack_session_ids):
            if session_id in answer_session_ids:
                evidence_session_idx.append(idx)
        
        # 标记包含答案的消息
        haystack_sessions = data.get("haystack_sessions", [])
        for idx, session in enumerate(haystack_sessions):
            if not isinstance(session, list):
                session = [session]
            for i, msg in enumerate(session):
                # 标记当前消息是否包含答案
                msg["has_answer"] = idx in evidence_session_idx
                # 如果原始数据中有 has_answer 字段，使用它
                if "has_answer" in msg:
                    # 已经设置，保持原值
                    pass
        
        # 收集 evidence（格式：D{session_idx}:{turn_idx}）
        evidence = []
        for idx, session in enumerate(haystack_sessions):
            if not isinstance(session, list):
                session = [session]
            for i, msg in enumerate(session):
                if msg.get("has_answer", False):
                    evidence.append(f"D{idx}:{i}")
        
        # 构建 QA
        answer = data.get("answer", "")
        # 确保 answer 转换为字符串（因为有些答案是整数）
        answer_str = str(answer) if answer is not None else ""
        
        data_dict["qa"].append({
            "question_id": question_id,
            "question": data.get("question", ""),
            "answer": answer_str,
            "evidence": evidence,
            "category": data.get("question_type", "unknown")
        })
        
        # 构建 conversation
        data_dict["conversation"]["speaker_a"] = f"user_{question_id}"
        data_dict["conversation"]["speaker_b"] = f"assistant_{question_id}"
        
        haystack_dates = data.get("haystack_dates", [])
        for idx, session in enumerate(haystack_sessions):
            # 转换时间格式
            session_date = haystack_dates[idx] if idx < len(haystack_dates) else ""
            converted_date = convert_time_format(session_date) if session_date else ""
            
            # 设置 session 日期时间
            data_dict["conversation"][f"session_{idx}_date_time"] = converted_date
            
            # 设置 session 内容
            session_key = f"session_{idx}"
            data_dict["conversation"][session_key] = []
            
            if not isinstance(session, list):
                session = [session]
            
            for i, msg in enumerate(session):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                
                data_dict["conversation"][session_key].append({
                    "speaker": f"{role}_{question_id}",
                    "text": content,
                    "dia_id": f"D{idx}:{i}"
                })
        
        locomo_style_data.append(data_dict)
    
    # 如果提供了输出路径，保存文件
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(locomo_style_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 已保存 {len(locomo_style_data)} 条转换后的数据到: {output_path}")
    
    return locomo_style_data


if __name__ == "__main__":
    import sys
    

        # 默认测试模式：测试数据加载器
    print("Testing LongMemEval data loader...")
    count = 0
    for session in iter_sessions():
        count += 1
        print(f"\nSession {count}:")
        print(f"  session_id: {session['session_id']}")
        print(f"  question_type: {session['meta'].get('question_type', 'unknown')}")
        print(f"  num_turns (fine-grain): {len(session['turns'])}")
        print(f"  num_session_chunks (coarse-grain): {len(session.get('session_chunks', []))}")
        print(f"  num_qa_pairs: {len(session['qa_pairs'])}")
        if session['turns']:
            first_turn = session['turns'][0]
            print(f"  first fine-grain turn: {first_turn['speaker']}: {first_turn['text'][:80]}...")
        if session.get('session_chunks'):
            first_chunk = session['session_chunks'][0]
            print(f"  first session_chunk: {first_chunk['speaker']}: {first_chunk['text'][:80]}...")
        if session['qa_pairs']:
            qa = session['qa_pairs'][0]
            print(f"  question: {qa['question']}")
            # 安全处理ground_truth（可能是字符串或整数）
            ground_truth = qa['ground_truth']
            if isinstance(ground_truth, str):
                print(f"  answer: {ground_truth[:80]}...")
            else:
                print(f"  answer: {ground_truth}")
        
        # 保存示例到临时目录
        os.makedirs("./core/datasets/tmp_longmemeval", exist_ok=True)
        with open(f"./core/datasets/tmp_longmemeval/{session['session_id']}.json", "w", encoding='utf-8') as f:
            json.dump(session, f, indent=4, ensure_ascii=False)
    
    print(f"\nTotal sessions: {count}")
    print("\n💡 提示: 使用 'python longmemeval.py convert [input_path] [output_path]' 来转换数据为 LoCoMo 格式")

