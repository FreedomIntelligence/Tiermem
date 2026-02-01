#!/usr/bin/env python3
"""
在线评估 Router：对 QUERY action 真实执行搜索

特点：
- 按 benchmark + session_id 排序，避免重复加载数据库
- 缓存 mem0 实例，同一 session 复用
- QUERY action 真实执行：refined_query → search → LLM-as-judge

Usage:
    # 使用 vLLM 作为 Router (MAB)
    python scripts/router_training/5_eval_router_online.py \
      --router-type vllm \
      --vllm-url http://localhost:8000/v1 \
      --model Qwen3-4B-Thinking-2507 \
      --data data/router_offline_v2/ \
      --benchmark mab

    # 使用 OpenAI (gpt-4.1-mini) 作为 Router (MAB)
    python scripts/router_training/5_eval_router_online.py \
      --router-type openai \
      --model gpt-4.1-mini \
      --data data/router_offline_v2/ \
      --benchmark mab

    # 评估 locomo (使用 OpenAI)
    python scripts/router_training/5_eval_router_online.py \
      --router-type openai \
      --model gpt-4.1 \
      --data data/router_offline_locomo_v2/ \
      --benchmark locomo \
      --limit 200 \
      --output-dir results/router_online_eval_gpt_4.1

    # 评估 locomo (使用 vLLM)
    python scripts/router_training/5_eval_router_online.py \
      --router-type vllm \
      --vllm-url http://localhost:8000/v1 \
      --model Qwen3-0.6B \
      --data data/router_offline_locomo_v2/ \
      --benchmark locomo \
      --limit 1540 \
      --output-dir results/eval/router_online_eval_0.6B_grpo_600_test_true
      
    # 只评估前100条数据 (使用 OpenAI)
    python scripts/router_training/5_eval_router_online.py \
      --router-type openai \
      --model gpt-4.1-mini \
      --data data/router_offline_v2/ \
      --benchmark mab \
      --limit 100
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_thinking_output(text: str) -> str:
    """解析 Thinking 模型输出，提取 </think> 后面的内容"""
    if not text:
        return ""
    think_pattern = r'</think>\s*(.*)$'
    match = re.search(think_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if '<think>' in text.lower() and '</think>' not in text.lower():
        return ""
    return text.strip()


def extract_action(text: str) -> Tuple[str, str]:
    """从 Router 输出中提取 action 和 query"""
    text = parse_thinking_output(text) or text
    text = text.strip()

    if "{" in text:
        try:
            obj = json.loads(text)
            action = obj.get("action", "").upper().strip()
            query = obj.get("query", "").strip()
            if action in ("S", "R"):
                return action, ""
            elif action in ("QUERY", "REFINE"):
                return "QUERY", query
        except json.JSONDecodeError:
            pass

        start_idx = text.find("{")
        if start_idx >= 0:
            brace_count = 0
            end_idx = start_idx
            for i in range(start_idx, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break
            if brace_count == 0 and end_idx > start_idx:
                try:
                    json_str = text[start_idx:end_idx + 1]
                    obj = json.loads(json_str)
                    action = obj.get("action", "").upper().strip()
                    query = obj.get("query", "").strip()
                    if action in ("S", "R"):
                        return action, ""
                    elif action in ("QUERY", "REFINE"):
                        return "QUERY", query
                except json.JSONDecodeError:
                    pass

    text_upper = text.upper()
    if text_upper.startswith("QUERY:"):
        return "QUERY", text[6:].strip()
    if text_upper.startswith("REFINE:"):
        return "QUERY", text[7:].strip()
    if text_upper == "S":
        return "S", ""
    if text_upper == "R":
        return "R", ""

    for ch in text_upper:
        if ch == "S":
            return "S", ""
        if ch == "R":
            return "R", ""

    return "UNKNOWN", ""


def load_offline_data(data_path: str, benchmark: Optional[str] = None) -> List[Dict[str, Any]]:
    """加载离线数据集，按 benchmark + session_id 排序"""
    data_path = Path(data_path)
    samples = []

    if data_path.is_file():
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    elif data_path.is_dir():
        for jsonl_file in sorted(data_path.glob("*.jsonl")):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))

    # 过滤 benchmark
    #if benchmark:
        #samples = [s for s in samples if s.get("benchmark", s.get("split", "")).startswith(benchmark)]

    # 按 benchmark + session_id 排序
    samples.sort(key=lambda x: (x.get("benchmark", ""), x.get("session_id", "")))

    return samples


def build_router_prompt(sample: Dict[str, Any]) -> str:
    """构建 Router prompt"""
    question = sample.get("question", "")
    summaries = sample.get("original_summaries", [])
    scores = sample.get("original_scores", [])

    summaries_lines = []
    for i, (summary, score) in enumerate(zip(summaries[:5], scores[:5])):
        summaries_lines.append(f"[{i}] score={score:.3f}\n{summary}")

    summaries_block = "\n\n".join(summaries_lines) if summaries_lines else "(no summaries retrieved)"

    prompt1 = f"""You are an expert router for a memory-augmented QA system.

Your task: Analyze the retrieved summaries and decide the best action to answer the question.

Available actions:
1. "S" - Answer using current summaries only 
    Use when: Summaries contain the EXPLICIT answer to the specific question.
    CRITICAL: Do not infer causes from effects (e.g., "benefits of X" is NOT "reason for starting X"). 
    If the exact answer is not verbatim in the text, or the question needs efficient details do not use S.

2. "QUERY" - Search again with a better query
    Use when: Summaries contain partial clues or related keywords but lack the direct answer.
    Include a more targeted query that focuses on the missing specific detail.

3. "R" - Deep research mode (slow path)
    Use when: Summaries are ambiguous, only contextually related, or miss the answer entirely.
    If the question requires completeness (e.g., "how many times", "list all", "what are all", "Both of them"), prefer R to ensure comprehensive answers.


Question: {question}

Retrieved Summaries:
{summaries_block}

Output format (JSON only):
- If answering with summaries: {{"action": "S"}}
- If need better search: {{"action": "QUERY", "query": "your refined search query"}}
- If deep research needed: {{"action": "R"}}

Your response:"""
    prompt = f"""You are an expert router for a memory-augmented QA system.

Your task: Analyze the retrieved summaries and decide the best action to answer the question.

Available actions:
1. "S" - Answer using current summaries only 
    Use when: Summaries contain the EXPLICIT answer to the specific question.
    CRITICAL: Do not infer causes from effects (e.g., "benefits of X" is NOT "reason for starting X"). 
    If the exact answer is not verbatim in the text, or the question needs efficient details do not use S.

2. "R" - Deep research mode (slow path)
    Use when: Summaries are ambiguous, only contextually related, or miss the answer entirely.
    If the question requires completeness (e.g., "how many times", "list all", "what are all", "Both of them"), prefer R to ensure comprehensive answers.


Question: {question}

Retrieved Summaries:
{summaries_block}

Output format (JSON only):
- If answering with summaries: {{"action": "S"}}
- If deep research needed: {{"action": "R"}}

Your response:"""


    

    return prompt


class OnlineEvaluator:
    """在线评估器：支持真实 QUERY 评估"""

    def __init__(
        self,
        router_type: str = "vllm",
        vllm_url: Optional[str] = None,
        model_name: Optional[str] = None,
        judge_model: str = "gpt-4.1-mini",
        is_thinking_model: bool = True,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
    ):
        self.router_type = router_type
        self.vllm_url = vllm_url
        self.model_name = model_name
        self.judge_model = judge_model
        self.is_thinking_model = is_thinking_model
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port

        # Router client
        from openai import OpenAI
        if router_type == "vllm":
            if not vllm_url or not model_name:
                raise ValueError("vllm-url and model are required when router-type is vllm")
            self.router_client = OpenAI(
                api_key="vllm-api-key",
                base_url=vllm_url,
            )
        elif router_type == "openai":
            if not model_name:
                model_name = "gpt-4.1-mini"
            self.router_client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
            )
        else:
            raise ValueError(f"Unknown router_type: {router_type}")

        # OpenAI client for judge
        self.judge_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )

        # mem0 cache by session_id
        self._mem0_cache: Dict[str, Any] = {}
        self._current_benchmark = None

    def get_router_decision(self, prompt: str) -> str:
        """获取 Router 决策"""
        try:
            kwargs = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.5,
            }
            # 只有 vLLM 且是 thinking 模型时才添加 extra_body
            if self.router_type == "vllm" and self.is_thinking_model:
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": True},
                }
            response = self.router_client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"[Evaluator] Router call failed: {e}")
            return ""

    def get_mem0(self, session_id: str, benchmark: str) -> Any:
        """获取或创建 mem0 实例"""
        cache_key = f"{benchmark}:{session_id}"
        if cache_key in self._mem0_cache:
            return self._mem0_cache[cache_key]

        try:
            from src.mem0 import Memory
            from src.mem0.configs.base import MemoryConfig

            # 与 LinkedViewSystem 保持一致：将 session_id 中的 / 和 \ 替换为 _
            # 确保 collection_name 与写入时一致
            safe_session_id = session_id.replace("/", "_").replace("\\", "_")
            collection_name = f"TierMem_locomo_{safe_session_id}"

            config = MemoryConfig(
                llm={
                    "provider": "openai",
                    "config": {"model": "gpt-4.1-mini"},
                },
                vector_store={
                    "provider": "qdrant",
                    "config": {
                        "host": self.qdrant_host,
                        "port": self.qdrant_port,
                        "collection_name": collection_name,
                    },
                },
            )
            mem0 = Memory(config=config)
            self._mem0_cache[cache_key] = mem0
            return mem0
        except Exception as e:
            print(f"[Evaluator] Failed to create mem0 for {session_id}: {e}")
            return None

    def search_with_query(
        self,
        session_id: str,
        benchmark: str,
        query: str,
        top_k: int = 5,
    ) -> List[str]:
        """用 query 搜索数据库"""
        mem0 = self.get_mem0(session_id, benchmark)
        if mem0 is None:
            print(f"[Evaluator] Mem0 is not found for {session_id}: {benchmark}")
            return []

        try:
            # 生成 collection_name 用于调试
            safe_session_id = session_id.replace("/", "_").replace("\\", "_")
            collection_name = f"TierMem_{benchmark}_{safe_session_id}"
            
            user_id = f"{session_id}:user"
            results = mem0.search(query=query, user_id=user_id, limit=top_k)

            summaries = []
            print(f"[Evaluator] Search results: {results}")
            if isinstance(results, dict) and "results" in results:
                results = results["results"]

            # 如果搜索返回空，尝试诊断问题
            if not results:
                print(f"[Evaluator] Empty search results for session_id={session_id}, collection={collection_name}, user_id={user_id}")
                print(f"[Evaluator] Collection name: {collection_name}")
                print(f"[Evaluator] Qdrant host: {self.qdrant_host}:{self.qdrant_port}")
                # 尝试检查 collection 中是否有任何数据（不指定 user_id，但 mem0 可能要求至少一个 session id）
                # 注意：这个诊断可能失败，因为 mem0.search 要求至少一个 session id
                # 但我们可以尝试用不同的 user_id 格式搜索，或者直接检查 Qdrant
                try:
                    # 尝试检查 Qdrant collection 是否存在
                    from qdrant_client import QdrantClient
                    qdrant_client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
                    collections = qdrant_client.get_collections().collections
                    collection_names = [c.name for c in collections]
                    if collection_name in collection_names:
                        # collection 存在，检查是否有数据
                        collection_info = qdrant_client.get_collection(collection_name)
                        point_count = collection_info.points_count
                        print(f"[Evaluator] Collection exists with {point_count} points")
                        if point_count > 0:
                            print(f"[Evaluator] WARNING: Collection has data but search returned empty. Possible causes:")
                            print(f"[Evaluator]   1. user_id mismatch (searching with '{user_id}', but data may use different user_id)")
                            print(f"[Evaluator]   2. Query doesn't match any embeddings in the collection")
                    else:
                        print(f"[Evaluator] Collection does not exist in Qdrant")
                        print(f"[Evaluator] Available collections (first 10): {collection_names[:10]}")
                except Exception as diag_e:
                    print(f"[Evaluator] Diagnostic check failed: {diag_e}")
                    import traceback
                    print(f"[Evaluator] Traceback: {traceback.format_exc()}")

            for r in results:
                if isinstance(r, dict):
                    memory_text = r.get("memory", "") or r.get("text", "")
                else:
                    memory_text = str(r)
                if memory_text:
                    summaries.append(memory_text)

            return summaries
        except Exception as e:
            print(f"[Evaluator] Search failed for {session_id}: {e}")
            import traceback
            print(f"[Evaluator] Traceback: {traceback.format_exc()}")
            return []

    def llm_judge(
        self,
        question: str,
        summaries: List[str],
        ground_truth: List[str],
    ) -> bool:
        """LLM-as-Judge 评估 summaries 是否能回答问题"""
        summaries_text = "\n".join([f"- {s}" for s in summaries])
        
        # 确保 ground_truth 是列表
        if not isinstance(ground_truth, list):
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth]
            else:
                ground_truth = [str(ground_truth)]
        
        gt_text = ", ".join(ground_truth[:3]) if ground_truth else "(no ground truth)"

        prompt = f"""You are evaluating whether a QA system can correctly answer a question using retrieved memory summaries.

Question: {question}

Retrieved Summaries:
{summaries_text}

Ground Truth Answer(s): {gt_text}

Can the question be correctly answered using ONLY the retrieved summaries above?
Consider:
1. Do the summaries contain the key information needed?
2. Is the ground truth answer (or equivalent) derivable from the summaries?

Answer with ONLY "YES" or "NO"."""

        try:
            response = self.judge_client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            answer = response.choices[0].message.content.strip().upper()
            return answer == "YES"
        except Exception as e:
            print(f"[Evaluator] LLM judge failed: {e}")
            return False

    def evaluate_query_action(
        self,
        sample: Dict[str, Any],
        refined_query: str,
    ) -> Tuple[bool, List[str]]:
        """评估 QUERY action 的效果"""
        session_id = sample.get("session_id", "")
        benchmark = sample.get("benchmark", sample.get("split", "mab"))
        question = sample.get("question", "")
        ground_truth = sample.get("ground_truth", [])
        original_summaries = sample.get("original_summaries", [])

        # 确保 ground_truth 是列表
        if not isinstance(ground_truth, list):
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth]
            else:
                ground_truth = [str(ground_truth)]

        # 用 refined query 搜索
        new_summaries = self.search_with_query(session_id, benchmark, refined_query)

        if not new_summaries:
            # 搜索失败，fallback 到原始 summaries
            print(f"Search failed for {session_id}: {refined_query}")
            new_summaries = original_summaries
            print(f"Fallback to original summaries: {original_summaries}")

        # 合并 summaries（去重）
        all_summaries = list(set(original_summaries + new_summaries))

        # LLM-as-judge 评估
        is_correct = self.llm_judge(question, all_summaries, ground_truth)

        return is_correct, new_summaries


def evaluate_online(
    evaluator: OnlineEvaluator,
    samples: List[Dict[str, Any]],
    max_samples: Optional[int] = None,
    verbose: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """在线评估 Router"""
    if max_samples:
        samples = samples[:max_samples]

    results = {
        "total": len(samples),
        "actions": {"S": 0, "R": 0, "QUERY": 0, "UNKNOWN": 0},
        "correct_by_action": {"S": 0, "R": 0, "QUERY": 0},
        "optimal_match": 0,
        # 详细统计：应该选什么 vs 实际选了什么
        "should_s_actual_s": 0,          # S-S
        "should_s_actual_refine": 0,     # S-refine
        "should_s_actual_r": 0,          # S-R
        "should_r_actual_s": 0,          # R-S
        "should_r_actual_refine_correct": 0,  # R-refine-做对了
        "should_r_actual_refine_wrong": 0,    # R-refine-做错了
        "should_r_actual_r": 0,          # R-R
    }

    per_session_results = defaultdict(lambda: {
        "total": 0, "correct": 0, "s": 0, "r": 0, "query": 0,
    })

    # 准备 JSONL 输出文件并加载已处理的 query_id
    jsonl_file = None
    jsonl_path = None
    processed_query_ids = set()
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_path / "eval_details.jsonl"
        
        # 如果文件已存在，读取已处理的 query_id
        if jsonl_path.exists():
            print(f"发现已有评估记录: {jsonl_path}")
            print("加载已处理的 query_id...")
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            query_id = record.get("query_id", "")
                            if query_id:
                                processed_query_ids.add(query_id)
                        except json.JSONDecodeError:
                            continue
            print(f"已处理 {len(processed_query_ids)} 条记录，将跳过这些 query_id")
            # 使用追加模式
            jsonl_file = open(jsonl_path, "a", encoding="utf-8")
        else:
            # 创建新文件
            jsonl_file = open(jsonl_path, "w", encoding="utf-8")

    current_session = None
    skipped_count = 0

    try:
        for sample in tqdm(samples, desc="Evaluating"):
            query_id = sample.get("query_id", "")
            
            # 跳过已处理的 query_id
            if query_id in processed_query_ids:
                skipped_count += 1
                if verbose:
                    print(f"  跳过已处理的 query_id: {query_id}")
                continue
            
            session_id = sample.get("session_id", "")
            benchmark = sample.get("benchmark", sample.get("split", ""))
            s_correct = sample.get("s_path_correct", False)
            r_correct = sample.get("r_path_correct", False)
            
            # 跳过 s_correct 和 r_correct 都是 false 的案例
            if not s_correct and not r_correct:
                skipped_count += 1
                if verbose:
                    print(f"  跳过 s_correct 和 r_correct 都是 false 的案例: {query_id}")
                continue
            
            optimal = sample.get("optimal_action", "R")
            question = sample.get("question", "")
            ground_truth = sample.get("ground_truth", [])

            # 确保 ground_truth 是列表
            if not isinstance(ground_truth, list):
                if isinstance(ground_truth, str):
                    ground_truth = [ground_truth]
                else:
                    ground_truth = [str(ground_truth)]

            # Log session change
            if session_id != current_session:
                if verbose:
                    print(f"\n[Session] {session_id}")
                current_session = session_id

            # 获取 Router 决策
            prompt = build_router_prompt(sample)
            response = evaluator.get_router_decision(prompt)
            action, refined_query = extract_action(response)

            results["actions"][action] = results["actions"].get(action, 0) + 1

            # 评估
            new_summaries = []
            if action == "S":
                is_correct = s_correct
            elif action == "R":
                is_correct = r_correct
            elif action == "QUERY":
                # 如果 s_correct 是 true，说明应该选 S 而不是 QUERY，直接判定为错误
                if s_correct:
                    is_correct = False
                    new_summaries = []
                elif refined_query:
                    is_correct, new_summaries = evaluator.evaluate_query_action(
                        sample, refined_query
                    )
                else:
                    # 没有 query，fallback 到 r_correct
                    is_correct = r_correct
                    new_summaries = []
            else:
                is_correct = False

            if is_correct:
                results["correct_by_action"][action] = results["correct_by_action"].get(action, 0) + 1

            if action == optimal:
                results["optimal_match"] += 1

            # 详细统计：应该选什么 vs 实际选了什么
            # 确定应该选什么：如果 s_correct=True，应该选S；否则如果 r_correct=True，应该选R
            should_action = None
            if s_correct:
                should_action = "S"
            elif r_correct:
                should_action = "R"
            
            # 统计各种组合
            if should_action == "S":
                if action == "S":
                    results["should_s_actual_s"] += 1
                elif action == "QUERY":
                    results["should_s_actual_refine"] += 1
                elif action == "R":
                    results["should_s_actual_r"] += 1
            elif should_action == "R":
                if action == "S":
                    results["should_r_actual_s"] += 1
                elif action == "QUERY":
                    if is_correct:
                        results["should_r_actual_refine_correct"] += 1
                    else:
                        results["should_r_actual_refine_wrong"] += 1
                elif action == "R":
                    results["should_r_actual_r"] += 1

            # 按 session 统计
            per_session_results[session_id]["total"] += 1
            if is_correct:
                per_session_results[session_id]["correct"] += 1
            per_session_results[session_id][action.lower()] += 1

            # 保存详细记录到 JSONL
            if jsonl_file:
                # 获取 S/R response 和 ground_truth
                s_response = sample.get("s_path_answer", "")
                r_response = sample.get("r_path_answer", "")
                golden_response = sample.get("ground_truth", [])
                # 确保 golden_response 是列表
                if not isinstance(golden_response, list):
                    if isinstance(golden_response, str):
                        golden_response = [golden_response]
                    else:
                        golden_response = [str(golden_response)]

                record = {
                    "question": question,
                    "action": action,
                    "prompt": prompt,
                    "response": response,
                    "refined_query": refined_query if refined_query else "",
                    "new_summaries": new_summaries,
                    "is_correct": is_correct,
                    "s_correct": s_correct,
                    "r_correct": r_correct,
                    "s_response": s_response,
                    "r_response": r_response,
                    "golden_response": golden_response,
                    "session_id": session_id,
                    "benchmark": benchmark,
                    "query_id": query_id,
                }
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_file.flush()

            if verbose:
                print(f"  [{query_id}] {action} -> {'✓' if is_correct else '✗'}")
    finally:
        # 关闭 JSONL 文件
        if jsonl_file:
            jsonl_file.close()
            if jsonl_path:
                print(f"\n详细评估记录已保存到: {jsonl_path}")
                if skipped_count > 0:
                    print(f"跳过了 {skipped_count} 条已处理的记录")

    # 汇总：从 JSONL 文件读取所有记录进行统计
    total = results["total"]
    results["skipped"] = skipped_count
    new_processed_count = total - skipped_count  # 这一轮新处理的
    
    # 如果存在 JSONL 文件，从所有记录中统计
    if output_dir and jsonl_path and jsonl_path.exists():
        print("\n从所有已处理记录中统计结果...")
        all_results = {
            "total": 0,
            "actions": {"S": 0, "R": 0, "QUERY": 0, "UNKNOWN": 0},
            "correct_by_action": {"S": 0, "R": 0, "QUERY": 0},
            "optimal_match": 0,
            "should_s_actual_s": 0,
            "should_s_actual_refine": 0,
            "should_s_actual_r": 0,
            "should_r_actual_s": 0,
            "should_r_actual_refine_correct": 0,
            "should_r_actual_refine_wrong": 0,
            "should_r_actual_r": 0,
        }
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        all_results["total"] += 1
                        action = record.get("action", "")
                        is_correct = record.get("is_correct", False)
                        s_correct = record.get("s_correct", False)
                        r_correct = record.get("r_correct", False)
                        
                        # 统计 action 分布
                        if action in all_results["actions"]:
                            all_results["actions"][action] += 1
                        else:
                            all_results["actions"]["UNKNOWN"] += 1
                        
                        # 统计正确率
                        if is_correct and action in all_results["correct_by_action"]:
                            all_results["correct_by_action"][action] += 1
                        
                        # 统计详细组合
                        should_action = None
                        if s_correct:
                            should_action = "S"
                        elif r_correct:
                            should_action = "R"
                        
                        if should_action == "S":
                            if action == "S":
                                all_results["should_s_actual_s"] += 1
                            elif action == "QUERY":
                                all_results["should_s_actual_refine"] += 1
                            elif action == "R":
                                all_results["should_s_actual_r"] += 1
                        elif should_action == "R":
                            if action == "S":
                                all_results["should_r_actual_s"] += 1
                            elif action == "QUERY":
                                if is_correct:
                                    all_results["should_r_actual_refine_correct"] += 1
                                else:
                                    all_results["should_r_actual_refine_wrong"] += 1
                            elif action == "R":
                                all_results["should_r_actual_r"] += 1
                    except json.JSONDecodeError:
                        continue
        
        # 使用全部记录的统计结果
        results = all_results
        total = results["total"]
        total_correct = sum(results["correct_by_action"].values())
        results["accuracy"] = total_correct / max(1, total)
        results["decision_accuracy"] = results["optimal_match"] / max(1, total)
    else:
        # 只统计这一轮的
        total_correct = sum(results["correct_by_action"].values())
        results["accuracy"] = total_correct / max(1, new_processed_count)
        results["decision_accuracy"] = results["optimal_match"] / max(1, new_processed_count)

    # Baselines
    s_only_correct = sum(1 for s in samples if s.get("s_path_correct", False))
    r_only_correct = sum(1 for s in samples if s.get("r_path_correct", False))
    results["s_only_baseline"] = s_only_correct / max(1, len(samples))
    results["r_only_baseline"] = r_only_correct / max(1, len(samples))

    results["per_session"] = dict(per_session_results)
    results["new_processed"] = new_processed_count  # 这一轮新处理的
    results["skipped"] = skipped_count

    return results


def print_results(results: Dict[str, Any], save_to_file: Optional[Path] = None) -> str:
    """打印评估结果，可选保存到文件"""
    lines = []
    
    def add_line(s: str = ""):
        lines.append(s)
        print(s)
    
    add_line("\n" + "=" * 60)
    add_line("Online Evaluation Results")
    add_line("=" * 60)

    total = results["total"]
    skipped = results.get("skipped", 0)
    new_processed = results.get("new_processed", 0)
    
    add_line(f"\nTotal samples (all processed): {total}")
    if skipped > 0:
        add_line(f"Skipped in this run: {skipped}")
    if new_processed > 0:
        add_line(f"Newly processed in this run: {new_processed}")
    
    processed = total  # 使用全部记录进行统计

    add_line(f"\n--- Action Distribution ---")
    for action in ["S", "R", "QUERY", "UNKNOWN"]:
        count = results["actions"].get(action, 0)
        pct = 100 * count / max(1, processed)
        add_line(f"  {action}: {count} ({pct:.1f}%)")

    add_line(f"\n--- Accuracy ---")
    add_line(f"Answer Accuracy: {100*results['accuracy']:.1f}%")
    add_line(f"Decision Accuracy (vs optimal): {100*results['decision_accuracy']:.1f}%")

    add_line(f"\n--- Per-Action Accuracy ---")
    for action in ["S", "R", "QUERY"]:
        count = results["actions"].get(action, 0)
        correct = results["correct_by_action"].get(action, 0)
        if count > 0:
            acc = 100 * correct / count
            add_line(f"  {action}: {correct}/{count} ({acc:.1f}%)")

    add_line(f"\n--- Detailed Router Accuracy (Should vs Actual) ---")
    # 应该选S的情况
    should_s_total = (results.get("should_s_actual_s", 0) + 
                     results.get("should_s_actual_refine", 0) + 
                     results.get("should_s_actual_r", 0))
    if should_s_total > 0:
        add_line(f"\n  Should be S (total: {should_s_total}):")
        add_line(f"    S-S (correct): {results.get('should_s_actual_s', 0)} ({100*results.get('should_s_actual_s', 0)/should_s_total:.1f}%)")
        add_line(f"    S-refine (wrong): {results.get('should_s_actual_refine', 0)} ({100*results.get('should_s_actual_refine', 0)/should_s_total:.1f}%)")
        add_line(f"    S-R (wrong): {results.get('should_s_actual_r', 0)} ({100*results.get('should_s_actual_r', 0)/should_s_total:.1f}%)")
    
    # 应该选R的情况
    should_r_total = (results.get("should_r_actual_s", 0) + 
                     results.get("should_r_actual_refine_correct", 0) + 
                     results.get("should_r_actual_refine_wrong", 0) + 
                     results.get("should_r_actual_r", 0))
    if should_r_total > 0:
        add_line(f"\n  Should be R (total: {should_r_total}):")
        add_line(f"    R-S (wrong): {results.get('should_r_actual_s', 0)} ({100*results.get('should_r_actual_s', 0)/should_r_total:.1f}%)")
        add_line(f"    R-refine-correct: {results.get('should_r_actual_refine_correct', 0)} ({100*results.get('should_r_actual_refine_correct', 0)/should_r_total:.1f}%)")
        add_line(f"    R-refine-wrong: {results.get('should_r_actual_refine_wrong', 0)} ({100*results.get('should_r_actual_refine_wrong', 0)/should_r_total:.1f}%)")
        add_line(f"    R-R (correct): {results.get('should_r_actual_r', 0)} ({100*results.get('should_r_actual_r', 0)/should_r_total:.1f}%)")
    
    # 计算Router准确率（应该选S时选S，应该选R时选R或refine且做对）
    router_correct = (results.get("should_s_actual_s", 0) + 
                     results.get("should_r_actual_r", 0) + 
                     results.get("should_r_actual_refine_correct", 0))
    router_total = should_s_total + should_r_total
    if router_total > 0:
        add_line(f"\n  Router Decision Accuracy: {router_correct}/{router_total} ({100*router_correct/router_total:.1f}%)")

    add_line(f"\n--- Baselines ---")
    add_line(f"S-only baseline: {100*results['s_only_baseline']:.1f}%")
    add_line(f"R-only baseline: {100*results['r_only_baseline']:.1f}%")
    
    # 保存到文件
    if save_to_file:
        with open(save_to_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        add_line(f"\n结果已保存到: {save_to_file}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Online Router Evaluation")
    parser.add_argument("--router-type", type=str, default="vllm", choices=["vllm", "openai"],
                       help="Router type: vllm or openai (default: vllm)")
    parser.add_argument("--vllm-url", type=str, default=None, help="vLLM server URL (required if router-type is vllm)")
    parser.add_argument("--model", type=str, default=None, help="Model name (required if router-type is vllm, default: gpt-4.1-mini for openai)")
    parser.add_argument("--data", type=str, required=True, help="Path to offline dataset")
    parser.add_argument("--benchmark", type=str, default=None, help="Filter by benchmark (mab/locomo)")
    parser.add_argument("--judge-model", type=str, default="gpt-4.1-mini", help="LLM judge model")
    parser.add_argument("--max-samples", type=int, default=200, help="Max samples to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate (alias for --max-samples)")
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking mode")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--output-dir", type=str, default="./results/router_online_eval",
                       help="Directory to save detailed evaluation records (JSONL format)")
    parser.add_argument("--qdrant-host", type=str, default="localhost",
                       help="Qdrant server host (default: localhost)")
    parser.add_argument("--qdrant-port", type=int, default=6333,
                       help="Qdrant server port (default: 6333)")

    args = parser.parse_args()

    # 验证参数
    if args.router_type == "vllm":
        if not args.vllm_url or not args.model:
            print("Error: --vllm-url and --model are required when --router-type is vllm")
            return
    elif args.router_type == "openai":
        if not args.model:
            args.model = "gpt-4.1-mini"
        if not os.environ.get("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable is required when --router-type is openai")
            return

    # 加载数据
    print(f"Loading data from: {args.data}")
    samples = load_offline_data(args.data, args.benchmark)
    print(f"Loaded {len(samples)} samples")

    if not samples:
        print("No samples found!")
        return

    # 初始化评估器
    evaluator = OnlineEvaluator(
        router_type=args.router_type,
        vllm_url=args.vllm_url,
        model_name=args.model,
        judge_model=args.judge_model,
        is_thinking_model=not args.no_thinking,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
    )

    # 评估
    max_samples = args.limit if args.limit is not None else args.max_samples
    results = evaluate_online(
        evaluator=evaluator,
        samples=samples,
        max_samples=max_samples,
        verbose=args.verbose,
        output_dir=args.output_dir,
    )

    # 保存结果到 output_dir
    eval_txt_path = None
    eval_json_path = None
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        eval_txt_path = output_path / "eval.txt"
        eval_json_path = output_path / "eval.json"
    
    # 打印结果并保存文本格式
    print_results(results, save_to_file=eval_txt_path)
    
    # 保存 JSON 格式结果
    if eval_json_path:
        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"JSON 结果已保存到: {eval_json_path}")
    
    # 如果指定了 --output，也保存到那里
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results also saved to: {output_path}")


if __name__ == "__main__":
    main()
