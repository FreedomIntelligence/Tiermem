"""
MemoryAgentBench数据集加载器

参考实现：
- /share/home/qmzhu/benchmark_github/MemoryAgentBench/utils/eval_data_utils.py
- /share/home/qmzhu/benchmark_github/MemoryAgentBench/conversation_creator.py

MemoryAgentBench使用HuggingFace数据集，格式为：
- 每个样本有一个context（长文本）和多个question-answer pairs
- 评估流程：先memorize context，然后回答questions
"""
import os
from typing import Dict, Any, List, Optional, Iterator
from pathlib import Path

try:
    from datasets import load_dataset, load_from_disk
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")


def load_data_huggingface(
    dataset_name: str, 
    sub_dataset_source: str, 
    max_test_samples: Optional[int] = None, 
    seed: int = 42,
    local_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    从HuggingFace加载MemoryAgentBench数据集
    
    Args:
        dataset_name: 主数据集名称 (Accurate_Retrieval, Test_Time_Learning, 
                     Long_Range_Understanding, Conflict_Resolution)
        sub_dataset_source: 子数据集名称（用于过滤source字段）
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
    
    # 尝试从本地加载（如果已下载）
    if local_path:
        local_dataset_path = Path(local_path) / dataset_name
        if local_dataset_path.exists():
            print(f"Loading from local path: {local_dataset_path}")
            try:
                raw_data = load_from_disk(str(local_dataset_path))
                print(f"Loaded {len(raw_data)} samples from local disk")
            except Exception as e:
                print(f"Failed to load from local disk: {e}, falling back to HuggingFace")
                raw_data = load_dataset("ai-hyz/MemoryAgentBench", split=dataset_name, revision="main")
        else:
            print(f"Local path not found, loading from HuggingFace...")
            raw_data = load_dataset("ai-hyz/MemoryAgentBench", split=dataset_name, revision="main")
    else:
        # 从HuggingFace加载
        print(f"Loading {sub_dataset_source} from HuggingFace: ai-hyz/MemoryAgentBench")
        raw_data = load_dataset("ai-hyz/MemoryAgentBench", split=dataset_name, revision="main")
    
    print(f"Loaded {len(raw_data)} samples from {dataset_name}")
    
    # 按source过滤
    original_length = len(raw_data)
    filtered_data = raw_data.filter(
        lambda sample: sample.get("metadata", {}).get("source", "") == sub_dataset_source
    )
    print(f"Filtered to {len(filtered_data)} samples matching source '{sub_dataset_source}' "
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
        "context": sample.get("context", ""),
        "questions": questions,
        "answers": answers,
        "source": metadata.get("source", ""),
        "question_ids": question_ids[:num_qa],
        "qa_pair_ids": qa_pair_ids[:num_qa],
        "context_length": len(sample.get("context", "")),
        "metadata": metadata
    }


def iter_sessions(
    data_dir: Optional[str] = None,
    split: str = "Accurate_Retrieval",
    sub_dataset: Optional[str] = None,
    limit: Optional[int] = None,
    seed: int = 42
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
    
    # 默认使用split作为sub_dataset（如果未指定）
    if sub_dataset is None:
        sub_dataset = split
    
    # 确定本地路径
    local_path = None
    if data_dir:
        local_path = data_dir
    else:
        # 尝试使用默认路径
        default_path = "/share/home/qmzhu/AGMS/data/MemoryAgentBench_data"
        if Path(default_path).exists():
            local_path = default_path
    
    # 加载数据
    samples = load_data_huggingface(
        dataset_name=split,
        sub_dataset_source=sub_dataset,
        max_test_samples=limit,
        seed=seed,
        local_path=local_path
    )
    
    # 转换为统一格式
    for sample_idx, sample in enumerate(samples):
        session_id = f"mab_{split}_{sample_idx}"
        context = sample.get("context", "")
        questions = sample.get("questions", [])
        answers = sample.get("answers", [])
        qa_pair_ids = sample.get("qa_pair_ids", [])
        
        # 将context分割成chunks（模拟多轮对话）
        # 简单实现：按段落分割（双换行符）
        context_chunks = [chunk.strip() for chunk in context.split("\n\n") if chunk.strip()]
        
        # 如果context太长，可能需要进一步分割
        # 这里先简单处理，后续可以根据chunk_size参数优化
        if not context_chunks:
            # 如果没有段落分隔，按句子分割
            import re
            sentences = re.split(r'[.!?]\s+', context)
            context_chunks = [s.strip() for s in sentences if s.strip()]
        
        # 构建turns（用于写入memory）
        turns = []
        for chunk_idx, chunk in enumerate(context_chunks):
            turns.append({
                "speaker": "user",  # context作为用户提供的背景信息
                "text": chunk,
                "chunk_id": chunk_idx
            })
        
        # 构建QA pairs
        qa_pairs = []
        for qa_idx, (question, answer) in enumerate(zip(questions, answers)):
            qa_pair_id = qa_pair_ids[qa_idx] if qa_idx < len(qa_pair_ids) else f"qa_{qa_idx}"
            qa_pairs.append({
                "query_id": f"{session_id}_{qa_pair_id}",
                "question": question,
                "ground_truth": answer,
                "meta": {
                    "sample_id": session_id,
                    "qa_index": qa_idx,
                    "qa_pair_id": qa_pair_id,
                    "source": sample.get("source", "")
                }
            })
        
        yield {
            "session_id": session_id,
            "turns": turns,
            "qa_pairs": qa_pairs,
            "meta": {
                "num_context_chunks": len(turns),
                "context_length": len(context),
                "num_qa_pairs": len(qa_pairs),
                "source": sample.get("source", "")
            }
        }


if __name__ == "__main__":
    # 测试代码
    print("Testing MemoryAgentBench data loader...")
    
    if not DATASETS_AVAILABLE:
        print("ERROR: datasets library not available. Install with: pip install datasets")
        exit(1)
    
    try:
        count = 0
        # 测试一个小数据集
        for session in iter_sessions(
            split="Accurate_Retrieval",
            sub_dataset="longmemeval_s_-1_500",  # 这是一个较小的子集
            limit=2
        ):
            count += 1
            print(f"\nSession {count}:")
            print(f"  session_id: {session['session_id']}")
            print(f"  num_turns: {len(session['turns'])}")
            print(f"  num_qa_pairs: {len(session['qa_pairs'])}")
            if session['turns']:
                print(f"  first turn: {session['turns'][0]['text'][:100]}...")
            if session['qa_pairs']:
                print(f"  first question: {session['qa_pairs'][0]['question']}")
                print(f"  first answer: {session['qa_pairs'][0]['ground_truth']}")
        print(f"\nTotal sessions: {count}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: If you see 'dataset not found', you may need to:")
        print("1. Download the dataset from HuggingFace: ai-hyz/MemoryAgentBench")
        print("2. Or set the data_dir parameter to point to local dataset")
        import traceback
        traceback.print_exc()


