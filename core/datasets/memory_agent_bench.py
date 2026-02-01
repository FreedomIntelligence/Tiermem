"""
MemoryAgentBench数据集加载器

MemoryAgentBench使用HuggingFace数据集，格式为：
- 每个样本有一个context（长文本）和多个question-answer pairs
- 评估流程：先memorize context，然后回答questions
"""
import os
import re
from typing import Dict, Any, List, Optional, Iterator, Union
from pathlib import Path
import json
import nltk
import tiktoken
try:
    from datasets import load_dataset, load_from_disk
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")


def load_data_huggingface(
    dataset_name: str,
    sub_dataset_source: Optional[Union[str, List[str]]] = None,
    max_test_samples: Optional[int] = None,
    seed: int = 42,
    local_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    从HuggingFace加载MemoryAgentBench数据集
    
    Args:
        dataset_name: 主数据集名称 (Accurate_Retrieval, Test_Time_Learning, 
                     Long_Range_Understanding, Conflict_Resolution)
        sub_dataset_source: 子数据集名称（用于过滤source字段，支持通配符匹配）
        max_test_samples: 最大样本数
        seed: 随机种子
        local_path: 本地数据集路径（如果已下载）
        
    Returns:
        处理后的数据列表
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library is required. Install with: pip install datasets")
    
    # 支持的split
    supported_splits = {
        "Accurate_Retrieval", "Test_Time_Learning", 
        "Long_Range_Understanding", "Conflict_Resolution"
    }
    
    if dataset_name not in supported_splits:
        raise ValueError(f"Unknown dataset {dataset_name}. Available: {sorted(supported_splits)}")
    
    # 尝试从本地路径加载
    if local_path and Path(local_path).exists():
        print(f"Loading from local path: {local_path}")
        try:
            raw_data = load_from_disk(local_path)
            # 如果本地路径是数据集字典，需要选择对应的split
            if isinstance(raw_data, dict):
                if dataset_name in raw_data:
                    raw_data = raw_data[dataset_name]
                else:
                    print(f"Warning: split '{dataset_name}' not found in local dataset, falling back to HuggingFace")
                    local_path = None
        except Exception as e:
            print(f"Warning: Failed to load from local path {local_path}: {e}")
            print("Falling back to HuggingFace...")
            local_path = None
    
    # 如果本地加载失败，从HuggingFace加载
    if not local_path or not Path(local_path).exists():
        print(f"Loading {sub_dataset_source} from HuggingFace: ai-hyz/MemoryAgentBench")
        raw_data = load_dataset("ai-hyz/MemoryAgentBench", split=dataset_name, revision="main")
    
    print(f"Loaded {len(raw_data)} samples from {dataset_name}")
    
    # 按source过滤，支持通配符和列表；None 表示不过滤
    original_length = len(raw_data)
    if sub_dataset_source is None:
        filtered_data = raw_data
        print(f"No source filter applied, keep all {original_length} samples")
    else:
        import fnmatch

        # 支持逗号分隔字符串或列表
        if isinstance(sub_dataset_source, str):
            # 允许用户传 "a,b,c"
            parts = [p.strip() for p in sub_dataset_source.split(",") if p.strip()]
        else:
            parts = [str(p).strip() for p in sub_dataset_source if str(p).strip()]

        def match_any(source_val: str) -> bool:
            for pat in parts:
                if "*" in pat or "?" in pat:
                    if fnmatch.fnmatch(source_val, pat):
                        return True
                else:
                    if source_val == pat:
                        return True
                    # 针对 longmemeval_s_-1_500 自动兜底到 longmemeval_s*
                    if "longmemeval" in pat.lower() and fnmatch.fnmatch(source_val, "longmemeval_s*"):
                        return True
            return False

        filtered_data = raw_data.filter(
            lambda sample: match_any(sample.get("metadata", {}).get("source", ""))
        )

        print(f"Filtered to {len(filtered_data)} samples matching sources {parts} "
              f"(from {original_length} total)")
    
    # 限制样本数
    if max_test_samples is not None and len(filtered_data) > max_test_samples:
        filtered_data = filtered_data.select(range(max_test_samples))
        print(f"Subsampled to {max_test_samples} samples")
    
    # 转换为列表格式
    processed_data = []
    for sample in filtered_data:
        processed_sample = _process_single_sample(sample)
        processed_data.append(processed_sample)
    
    return processed_data


def _process_single_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理单个样本，确保所有字段格式正确
    
    Args:
        sample: HuggingFace数据集样本
        
    Returns:
        处理后的样本字典
    """
    metadata = sample.get("metadata", {})
    
    # 确保questions和answers是列表
    questions = sample.get("questions", [])
    answers = sample.get("answers", [])
    
    if not isinstance(questions, list):
        questions = [questions] if questions else []
    if not isinstance(answers, list):
        answers = [answers] if answers else []
    
    # 确保长度一致
    num_qa = max(len(questions), len(answers))
    if len(questions) < num_qa:
        questions.extend([""] * (num_qa - len(questions)))
    if len(answers) < num_qa:
        answers.extend([""] * (num_qa - len(answers)))
    
    # 提取metadata字段
    question_ids = metadata.get("question_ids", [])
    qa_pair_ids = metadata.get("qa_pair_ids", [])
    
    if not isinstance(question_ids, list):
        question_ids = [str(question_ids)] if question_ids else []
    if not isinstance(qa_pair_ids, list):
        qa_pair_ids = [str(qa_pair_ids)] if qa_pair_ids else []
    
    # 确保qa_pair_ids长度足够
    while len(qa_pair_ids) < num_qa:
        qa_pair_ids.append(f"qa_{len(qa_pair_ids)}")
    
    return {
        "context": _normalize_context_text(sample.get("context", "")),
        "questions": questions,
        "answers": answers,
        "source": metadata.get("source", ""),
        "question_ids": question_ids[:num_qa],
        "qa_pair_ids": qa_pair_ids[:num_qa],
        "context_length": len(sample.get("context", "")),
        "metadata": metadata
    }


def _normalize_context_text(raw_context: Any) -> str:
    """
    将context字段统一成纯文本。
    - HF的LongMemEval部分context存成字符串化的对话列表，需要解析后再拼成文本。
    - 格式可能是：
      1. 单个对话段: ['Chat Time: ...', [{'role': ...}, ...]]
      2. 多个对话段（嵌套）: [['Chat Time: ...', [messages]], ['Chat Time: ...', [messages]], ...]
      3. 多个对话段（交替）: ['Chat Time: ...', [messages], 'Chat Time: ...', [messages], ...]
    - 其他情况直接转成字符串。
    """
    # 已是字符串但形如 "['Chat Time: ...', [{'role': ...}, ...]]" 或包含多个对话段
    if isinstance(raw_context, str) and raw_context.strip().startswith("[") and "role" in raw_context:
        import ast
        try:
            parsed = ast.literal_eval(raw_context)
        except Exception:
            return raw_context

        if not isinstance(parsed, (list, tuple)) or len(parsed) == 0:
            return str(raw_context)

        all_lines = []
        
        # 检查是否是嵌套格式：[[chat_time, messages], [chat_time, messages], ...]
        if isinstance(parsed[0], (list, tuple)) and len(parsed[0]) >= 2:
            # 嵌套格式：每个元素都是 [chat_time, messages]
            for segment in parsed:
                if isinstance(segment, (list, tuple)) and len(segment) >= 2:
                    chat_time = segment[0]
                    msgs = segment[1]
                    if isinstance(chat_time, str):
                        all_lines.append(chat_time)
                    if isinstance(msgs, (list, tuple)):
                        for m in msgs:
                            role = m.get("role", "user") if isinstance(m, dict) else "user"
                            content = m.get("content", "") if isinstance(m, dict) else str(m)
                            all_lines.append(f"{role}: {content}")
        # 检查是否是交替格式：['Chat Time: ...', [messages], 'Chat Time: ...', [messages], ...]
        elif isinstance(parsed[0], str) and "Chat Time:" in parsed[0]:
            # 交替格式：字符串（时间）和列表（消息）交替
            i = 0
            while i < len(parsed):
                if isinstance(parsed[i], str) and "Chat Time:" in parsed[i]:
                    # 这是时间戳
                    all_lines.append(parsed[i])
                    i += 1
                    # 下一个应该是消息列表
                    if i < len(parsed) and isinstance(parsed[i], (list, tuple)):
                        for m in parsed[i]:
                            role = m.get("role", "user") if isinstance(m, dict) else "user"
                            content = m.get("content", "") if isinstance(m, dict) else str(m)
                            all_lines.append(f"{role}: {content}")
                        i += 1
                else:
                    i += 1
        # 单个对话段格式：['Chat Time: ...', [messages]]
        elif len(parsed) >= 2 and isinstance(parsed[0], str) and isinstance(parsed[1], (list, tuple)):
            chat_time = parsed[0]
            msgs = parsed[1]
            if isinstance(chat_time, str):
                all_lines.append(chat_time)
            if isinstance(msgs, (list, tuple)):
                for m in msgs:
                    role = m.get("role", "user") if isinstance(m, dict) else "user"
                    content = m.get("content", "") if isinstance(m, dict) else str(m)
                    all_lines.append(f"{role}: {content}")

        if all_lines:
            return "\n\n".join(all_lines)

    # 兜底：直接字符串化
    return str(raw_context)


def iter_sessions(
    data_dir: Optional[str] = None,
    split: str = "Accurate_Retrieval",
    sub_dataset: Optional[Union[str, List[str]]] = None,
    limit: Optional[int] = None,
    seed: int = 42,
    chunk_size_fine: int = 512,
    chunk_size_coarse: int = 2048,  # 官方默认4096
    agent_name: str = "long_context_agent",
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    迭代MemoryAgentBench数据集中的sessions
    
    统一接口：返回格式为
    {
        "session_id": str,
        "turns": [...],  # context chunks（用于写入memory）
        "qa_pairs": [...]  # 需要回答的问题
    }
    
    Args:
        data_dir: 数据目录路径（如果已下载到本地）
        split: 数据集split (Accurate_Retrieval, Test_Time_Learning, 
               Long_Range_Understanding, Conflict_Resolution)
        sub_dataset: 子数据集名称（如 "longmemeval_s_-1_500"）
                     如果为None，将使用split作为sub_dataset
        limit: 限制返回的样本数量（用于测试）
        seed: 随机种子
        
    Yields:
        Session字典
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library is required. Install with: pip install datasets")
    
    # sub_dataset=None 时不过滤 source；支持列表或逗号分隔
    # 缓存路径（按 split/sub_dataset/chunk_size/agent_name 区分）
    cache_base = Path(cache_dir) if cache_dir else Path(__file__).parent / "cache"
    cache_base.mkdir(parents=True, exist_ok=True)
    sub_key = "all" if sub_dataset is None else (
        sub_dataset if isinstance(sub_dataset, str) else "-".join(sub_dataset)
    )
    cache_name = f"{split}_{sub_key}_f{chunk_size_fine}_c{chunk_size_coarse}_{agent_name}.jsonl"
    cache_path = cache_base / cache_name

    if use_cache and cache_path.exists():
        for s in _load_cached_sessions(cache_path, limit):
            yield s
        return

    # 确定本地路径
    local_path = None
    if data_dir:
        local_path = data_dir
    else:
        # 尝试使用多个可能的默认路径
        default_paths = [
            "./data/MemoryAgentBench_data/data"
        ]
        for default_path in default_paths:
            if Path(default_path).exists():
                local_path = default_path
                print(f"Found local dataset at: {local_path}")
                break
    
    # 加载数据
    samples = load_data_huggingface(
        dataset_name=split,
        sub_dataset_source=sub_dataset,
        max_test_samples=limit,
        seed=seed,
        local_path=local_path
    )
    
    # 转换为统一格式
    sessions_buffer: List[Dict[str, Any]] = []
    for sample_idx, sample in enumerate(samples):
        session_id = f"mab_{split}_{sample_idx}"
        context = sample.get("context", "")
        questions = sample.get("questions", [])
        answers = sample.get("answers", [])
        qa_pair_ids = sample.get("qa_pair_ids", [])

        # ================= 粗粒度与细粒度双轨输出（对齐官方句子切分思路） =================
        fine_chunks = _chunk_text_into_sentences(context, chunk_size=chunk_size_fine)
        coarse_chunks = _chunk_text_into_sentences(context, chunk_size=chunk_size_coarse)
        session_chunks_turns = [
            {
                "speaker": "system",
                "text": chunk,
                "chunk_id": idx,
            }
            for idx, chunk in enumerate(coarse_chunks)
        ]

        fine_turns = [
            {
                "speaker": "user",
                "text": chunk,
                "chunk_id": idx,
            }
            for idx, chunk in enumerate(fine_chunks)
        ]
        
        # 构建QA pairs（官方模板化 query）
        qa_pairs = []
        for qa_idx, (question, answer) in enumerate(zip(questions, answers)):
            qa_pair_id = qa_pair_ids[qa_idx] if qa_idx < len(qa_pair_ids) else f"qa_{qa_idx}"
            formatted_question = _format_query(
                raw_question=question,
                raw_answer=answer,
                sample=sample,
                qa_idx=qa_idx,
                qa_pair_id=qa_pair_id,
                agent_name=agent_name,
            )
            qa_pairs.append({
                "query_id": f"{session_id}_{qa_pair_id}",
                "question": formatted_question,
                "ground_truth": answer,
                "meta": {
                    "sample_id": session_id,
                    "qa_index": qa_idx,
                    "qa_pair_id": qa_pair_id,
                    "source": sample.get("source", "")
                }
            })
        
        session_obj = {
            "session_id": session_id,
            "turns": fine_turns,               # 细粒度
            "session_chunks": session_chunks_turns,  # 粗粒度
            "qa_pairs": qa_pairs,
            "meta": {
                "num_context_chunks": len(fine_turns),
                "num_session_chunks": len(session_chunks_turns),
                "context_length": len(context),
                "num_qa_pairs": len(qa_pairs),
                "source": sample.get("source", ""),
                "chunk_size_fine": chunk_size_fine,
                "chunk_size_coarse": chunk_size_coarse,
            }
        }
        sessions_buffer.append(session_obj)
        if limit is not None and len(sessions_buffer) >= limit:
            break

    # 写入缓存（仅全量）
    if use_cache and limit is None and sessions_buffer:
        _save_cached_sessions(cache_path, sessions_buffer)

    for session_obj in sessions_buffer:
        yield session_obj


def _chunk_text_into_sentences(text: str, chunk_size: int = 2048) -> List[str]:
    """
    官方思路：按句子切分，基于token窗口打包。
    默认使用 gpt-4o-mini 编码器，窗口为 chunk_size tokens。
    """
    text = text.strip()
    if not text:
        return []

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    except KeyError:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    sentences = nltk.sent_tokenize(text)
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sent in sentences:
        token_cnt = len(encoding.encode(sent, allowed_special={"<|endoftext|>"}))
        if current and current_tokens + token_cnt > chunk_size:
            chunks.append(" ".join(current))
            current = [sent]
            current_tokens = token_cnt
        else:
            current.append(sent)
            current_tokens += token_cnt

    if current:
        chunks.append(" ".join(current))
    return chunks


# ---------------- 缓存读写 ----------------
def _load_cached_sessions(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    sessions: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            sessions.append(json.loads(line))
            if limit is not None and len(sessions) >= limit:
                break
    return sessions


def _save_cached_sessions(path: Path, sessions: List[Dict[str, Any]]) -> None:
    with path.open("w") as f:
        for s in sessions:
            f.write(json.dumps(s))
            f.write("\n")


# ---------------- 辅助：官方模板化query ----------------
_DATASET_MAPPING = [
    (("ruler_", "qa"), "ruler_qa"),
    (("icl_",), "in_context_learning"),
    (("infbench_", "sum"), "infbench_sum"),
    (("eventqa_",), "eventqa"),
    (("recsys_", "redial"), "recsys_redial"),
    (("longmemeval_",), "longmemeval"),
    (("factconsolidation_",), "factconsolidation"),
    (("detective_", "qa"), "detective_qa"),
]

_AGENT_MAPPING = {
    "rag": "rag_agent",
    "long_context_agent": "long_context_agent",
    "agentic_memory": "agentic_memory_agent",
}

_BASE_TEMPLATES = {
    "ruler_qa": {
        "query": {
            "long_context_agent": "Answer the question based on the memorized documents. Only give me the answer and do not output any other words. \n\nQuestion: {question} \n\n Answer:",
            "rag_agent": "Answer the question based on the memorized documents. Only give me the answer and do not output any other words. \n\n Now Answer the Question: {question}",
            "agentic_memory_agent": "Search Archival Memory and answer my question. Only give me the answer and do not output any other words. \n\nQuestion: {question} \n\n Answer:",
        }
    },
    "longmemeval": {
        "query": {
            "long_context_agent": "The history chats are between you and a user. Based on the relevant chat history, answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:",
            "rag_agent": "The history chats are between you and a user. Based on the relevant chat history, answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:",
            "agentic_memory_agent": "Search Archival Memory and answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:",
        }
    },
    "eventqa": {
        "query": {
            "long_context_agent": "Based on the context you memorized, complete the task below:\n\n{question}\n\n The event that happens next is:",
            "rag_agent": "Based on the context you memorized, complete the task below:\n\n{question}\n\n The event that happens next is:",
            "agentic_memory_agent": "Search Archival Memory, complete the task below:\n\n{question}\n\n The event that happens next is:",
        }
    },
    "in_context_learning": {
        "query": {
            "long_context_agent": 'Use the provided mapping from the context to numerical label to assign a numerical label to the context. Only output "label: {{label}}" and nothing else. \n\n{question} \n\n label:',
            "rag_agent": 'Use the provided mapping from the context to numerical label to assign a numerical label to the context. Only output "label: {{label}}" and nothing else. \n\nQuestion:{question} \n\n label:',
            "agentic_memory_agent": 'Search Archival Memory and use the provided mapping from the context to numerical label to assign a numerical label to the context. Only output "label: {{label}}" and nothing else. \n\n{question} \n\n label:',
        }
    },
    "recsys_redial": {
        "query": {
            "long_context_agent": "Pretend you are a movie recommender system. You need to recommend movies based on the dialogues you have memorized. Now I will give you a new conversation between a user and you (a recommender system). Based on the conversation, you reply me with 20 recommendations without extra sentences. \n\n{question} \n\n The recommendations are:\n",
            "rag_agent": "Pretend you are a movie recommender system. You need to recommend movies based on the dialogues you have memorized. Now I will give you a new conversation between a user and you (a recommender system). Based on the conversation, you reply me with 20 recommendations without extra sentences. \n\n{question} \n\n The recommendations are:\n",
            "agentic_memory_agent": "Pretend you are a movie recommender system. You need to recommend movies based on the dialogues you have memorized. Search Archival Memory. \n\n{question} \n\n The recommendations are:\n",
        }
    },
    "infbench_sum": {
        "query": {
            "long_context_agent": "You are given a book above and you are tasked to summarize it. \n\n{question} \n\n Now summarize the book.",
            "rag_agent": "You are given a book above and you are tasked to summarize it. \n\n{question} \n\n Now summarize the book.",
            "agentic_memory_agent": "You are given a book above and you are tasked to summarize it. \n\n{question} \n\n Now summarize the book.",
        }
    },
    "detective_qa": {
        "query": {
            "long_context_agent": "Based on the context you memorized, answer the question below. You are required to answer the question based on the strict output format.\n\n {question} \n\n",
            "rag_agent": "Based on the context you memorized, answer the question below. You are required to answer the question based on the strict output format.\n\n {question} \n\n",
            "agentic_memory_agent": "Search Archival Memory and answer the question below. You are required to answer the question based on the strict output format.\n\n {question} \n\n",
        }
    },
    "factconsolidation": {
        "query": {
            "long_context_agent": "Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a serial number at the beginning, and the newer fact has larger serial number. You need to solve the conflicts of facts in the knowledge pool by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question only from the knowledge pool you have memorized rather than the real facts in real world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on the provided Knowledge Pool, what is the name of the current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the provided Knowledge Pool, {question} \nAnswer:",
            "rag_agent": "Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a serial number at the beginning, and the newer fact has larger serial number. You need to solve the conflicts of facts in the knowledge pool by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question only from the knowledge pool you have memorized rather than the real facts in real world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on the provided Knowledge Pool, what is the name of the current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the provided Knowledge Pool, {question} \nAnswer:",
            "agentic_memory_agent": "Pretend you are a knowledge management system. Each fact in the Archival Memory is provided with a serial number at the beginning, and the newer fact has larger serial number. You need to solve the conflicts of facts in the Archival Memory by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question only from the knowledge pool you have memorized rather than the real facts in real world. \n\nFor example:\n\n [Archival Memory] \n\n Question: Based on the Archival Memory, what is the name of the current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the Archival Memory, {question} \nAnswer:",
        }
    },
}


def _normalize_dataset_name(source: str) -> Optional[str]:
    for patterns, name in _DATASET_MAPPING:
        if all(p in source for p in patterns):
            return name
    return None


def _normalize_agent_name(agent_name: str) -> str:
    for key, val in _AGENT_MAPPING.items():
        if key in agent_name:
            return val
    return _AGENT_MAPPING["long_context_agent"]


def _format_query(
    raw_question: str,
    raw_answer: Any,
    sample: Dict[str, Any],
    qa_idx: int,
    qa_pair_id: str,
    agent_name: str = "long_context_agent",
) -> str:
    source = sample.get("source", "") or sample.get("metadata", {}).get("source", "")
    dataset_name = _normalize_dataset_name(source.lower())
    agent = _normalize_agent_name(agent_name)

    if dataset_name and dataset_name in _BASE_TEMPLATES:
        tpl = _BASE_TEMPLATES[dataset_name]["query"].get(agent)
        if tpl:
            return tpl.format(
                question=raw_question,
                answer=raw_answer,
                qa_pair_id=qa_pair_id,
                qa_index=qa_idx,
            )
    # Fallback：原始问题
    return raw_question


if __name__ == "__main__":
    # 预处理并写入本地缓存（全量，无limit）
    print("Preprocessing MemoryAgentBench and caching...")
    if not DATASETS_AVAILABLE:
        print("ERROR: datasets library not available. Install with: pip install datasets")
        exit(1)

    supported_splits = [
        "Accurate_Retrieval", "Test_Time_Learning",
        "Long_Range_Understanding"
    ]
    for split in supported_splits:
        try:
            count = 0
            qa_len = 0
            for _ in iter_sessions(
                split=split,
                limit=None,
                use_cache=True,
                cache_dir=None  # 默认写入 core/datasets/cache 下
            ):
                count += 1
                # qa_len 只作简单累计（示意）
                # 实际 QA 数在 meta 中，可根据需要统计
            print(f"[{split}] cached sessions: {count}")
        except Exception as e:
            print(f"Error preprocessing split {split}: {e}")
            print("\nNote: If you see 'dataset not found', you may need to:")
            print("1. Download the dataset from HuggingFace: ai-hyz/MemoryAgentBench")
            print("2. Or set the data_dir parameter to point to local dataset")
            import traceback
            traceback.print_exc()


