"""
HotpotQA数据集加载器
"""
import json
from typing import Dict, Any, List, Optional, Iterator
from pathlib import Path
import pandas as pd


def load_hotpotqa_parquet(parquet_path: str) -> List[Dict[str, Any]]:
    """
    加载HotpotQA parquet文件
    
    Args:
        parquet_path: parquet文件路径
        
    Returns:
        样本列表
    """
    df = pd.read_parquet(parquet_path)
    return df.to_dict('records')


def iter_sessions(
    data_dir: Optional[str] = None,
    split: str = "test",
    limit: Optional[int] = None
) -> Iterator[Dict[str, Any]]:
    """
    迭代HotpotQA数据集中的sessions
    
    统一接口：返回格式为
    {
        "session_id": str,
        "turns": [...],  # 历史对话（对于HotpotQA，这是context文本）
        "qa_pairs": [...]  # 需要回答的问题
    }
    
    Args:
        data_dir: 数据目录路径，默认使用 ./data/hotpot_qa
        split: 数据集split (train/validation/test)
        limit: 限制返回的样本数量（用于测试）

    Yields:
        Session字典
    """
    if data_dir is None:
        data_dir = "./data/hotpot_qa"
    
    # HotpotQA使用fullwiki split
    if split == "test":
        parquet_path = Path(data_dir) / "fullwiki" / "test-00000-of-00001.parquet"
    elif split == "validation":
        parquet_path = Path(data_dir) / "fullwiki" / "validation-00000-of-00001.parquet"
    elif split == "train":
        # train可能有多个文件，先处理第一个
        parquet_path = Path(data_dir) / "fullwiki" / "train-00000-of-00002.parquet"
    else:
        raise ValueError(f"Unknown split: {split}")
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"HotpotQA {split} data not found at {parquet_path}")
    
    samples = load_hotpotqa_parquet(str(parquet_path))
    
    if limit:
        samples = samples[:limit]
    
    for sample_idx, sample in enumerate(samples):
        # HotpotQA的格式：每个样本有一个context（长文本）和一个问题
        sample_id = sample.get("id", f"hotpotqa_{split}_{sample_idx}")
        context = sample.get("context", "")
        question = sample.get("question", "")
        answers = sample.get("answers", [])
        
        # 对于HotpotQA，context作为"历史对话"（需要写入memory）
        # 我们将context按段落或句子分割成多个turns
        # 简单实现：按双换行符分割
        context_chunks = [chunk.strip() for chunk in context.split("\n\n") if chunk.strip()]
        
        turns = []
        for chunk_idx, chunk in enumerate(context_chunks):
            turns.append({
                "speaker": "user",  # HotpotQA的context可以视为用户提供的背景信息
                "text": chunk,
                "chunk_id": chunk_idx
            })
        
        # QA pair
        qa_pairs = [{
            "query_id": f"{sample_id}_qa_0",
            "question": question,
            "ground_truth": answers,  # HotpotQA可能有多个正确答案
            "meta": {
                "sample_id": sample_id,
                "supporting_facts": sample.get("supporting_facts", []),
                "type": sample.get("type", "comparison")
            }
        }]
        
        yield {
            "session_id": sample_id,
            "turns": turns,
            "qa_pairs": qa_pairs,
            "meta": {
                "num_context_chunks": len(turns),
                "context_length": len(context)
            }
        }


if __name__ == "__main__":
    # 测试代码
    print("Testing HotpotQA data loader...")
    count = 0
    for session in iter_sessions(split="test", limit=2):
        count += 1
        print(f"\nSession {count}:")
        print(f"  session_id: {session['session_id']}")
        print(f"  num_turns: {len(session['turns'])}")
        print(f"  num_qa_pairs: {len(session['qa_pairs'])}")
        if session['turns']:
            print(f"  first turn: {session['turns'][0]['text'][:100]}...")
        if session['qa_pairs']:
            print(f"  question: {session['qa_pairs'][0]['question']}")
            print(f"  ground_truth: {session['qa_pairs'][0]['ground_truth']}")
    print(f"\nTotal sessions: {count}")


