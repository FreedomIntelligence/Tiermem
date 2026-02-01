#!/usr/bin/env python3
"""
使用 GPT-5 收集 Router 训练数据（改进版 V2 - S/R 二分类）

改进点：
1. LLM-as-Judge 评估 summary 覆盖度，而不是生成答案的正确性
2. 更系统的思维链 prompt（问题类型、完整性、精确性分析）
3. S/R 二分类：判断 summaries 是否足够回答问题
4. 兼容现有数据格式，可以增强已有数据

数据收集策略：
- 从所有 QA 样本中采样
- GPT-5 决策 + 严格 coverage judge
- 收集 S 和 R 两类数据

使用方式：
1. 新数据收集：--results-dir results/xxx --output data/router_training/new.jsonl
2. 增强已有数据：--enhance-existing data/router_distilled/locomo_s_fail_r_ok.jsonl --output data/router_training/enhanced.jsonl
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).parent.parent.parent  # scripts/router_training -> scripts -> TierMem
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI

# 项目模块
from src.memory.linked_view_system import LinkedViewSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ GPT-5 显式思维链 Prompt (只有 S/R 二分类) ============

ROUTER_THINKING_PROMPT_V2 = """You are a router for a memory-augmented QA system. Decide whether the retrieved summaries are sufficient to answer the question.

## Available Actions
1) "S" — Answer using current summaries only
Use when: The summaries contain a span that directly states the answer (or a direct paraphrase) WITHOUT needing to assume missing facts.
Rules of thumb:
- If you can point to a specific snippet that can be copied into the answer, choose S.

2) "R" — Deep research mode
Use when: Summaries are too ambiguous, contradictory, or unrelated; OR the question requires a comprehensive list and summaries are clearly incomplete.
R is more reliable than S.

## Input
QUESTION: {question}

SUMMARIES:
{summaries_block}

## Decision checklist
- What exact answer format is needed? (single value / list / date / identity)
- Do summaries explicitly contain it?
- Any ambiguity or need to infer?
- Any completeness requirement?

## Output (JSON only)
{{
  "thinking": "<your reasoning>",
  "action": "S/R"
}}
"""


# ============ 改进的 Coverage Judge ============

def judge_summary_coverage(
    client: OpenAI,
    question: str,
    gold_answer: str,
    summaries: List[str],
    model: str = "gpt-4.1",
) -> Tuple[bool, str, str]:
    """
    判断 summaries 是否包含足够信息来回答问题（严格标准）

    Returns: (has_sufficient_info, label, reason)
    """
    summaries_text = "\n".join(f"- {s}" for s in summaries) if summaries else "(No summaries)"

    prompt = f"""You are evaluating whether retrieved summaries contain sufficient information to answer a question.

Question: {question}
Gold answer: {gold_answer}

Retrieved summaries:
{summaries_text}

Your task: Determine if the summaries contain EXPLICIT information to answer the question correctly.

**Strict criteria** (be conservative):
1. The answer should be DIRECTLY stated or clearly derivable from the summaries
2. Do NOT count vague/related information as sufficient
3. Do NOT infer causes from effects (e.g., "benefits of X" does NOT tell us "reason for starting X")
4. For questions requiring completeness ("how many", "list all", "both"), summaries must provide COMPLETE information
5. For questions requiring exact details (dates, numbers, names), summaries must contain those exact details

If you have ANY doubt whether the summaries allow a factual answer without guessing, answer "false".

Output JSON (no extra text):
{{"has_sufficient_info": true/false, "reason": "brief explanation"}}
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        result = json.loads(resp.choices[0].message.content)
        has_info = result.get("has_sufficient_info", False)
        reason = result.get("reason", "")
        label = "SUFFICIENT" if has_info else "INSUFFICIENT"
        return has_info, label, reason
    except Exception as e:
        logger.error(f"Coverage judge failed: {e}")
        return False, "ERROR", str(e)


# ============ Data Classes ============

@dataclass
class RouterDecision:
    """GPT-5 的路由决策（S/R 二分类）"""
    # 连贯的思维链（自然语言段落）
    thinking: str = ""

    # 决策
    action: str = "R"  # S / R

    # 原始响应
    raw_response: str = ""


@dataclass
class CoverageJudgement:
    """Summary 覆盖度评估"""
    has_sufficient_info: bool
    label: str  # SUFFICIENT / INSUFFICIENT
    reason: str


@dataclass
class TrainingSample:
    """用于训练的样本"""
    query_id: str
    session_id: str
    question: str
    ground_truth: str

    # 检索结果
    summaries: List[str]
    scores: List[float]

    # GPT-5 决策（带思维链）
    gpt5_decision: RouterDecision

    # Coverage Judge（基于 summaries，不是生成的答案）
    coverage_judgement: CoverageJudgement

    # 训练标签（ground truth label）
    optimal_label: str = ""  # S / R

    # 辅助信息
    category: Optional[int] = None  # LoCoMo category


# ============ GPT-5 Router (显式思维链) ============

def gpt5_router_decide_v2(
    client: OpenAI,
    question: str,
    summaries: List[str],
    scores: List[float],
    model: str = "gpt-5",
) -> RouterDecision:
    """
    使用显式思维链 prompt 让 GPT-5 决策（S/R 二分类）

    输出格式：
    {
      "thinking": "...",
      "action": "S/R"
    }
    """
    import re

    summaries_block = ""
    for i, (summary, score) in enumerate(zip(summaries, scores)):
        summaries_block += f"[{i}] score={score:.3f}\n{summary}\n\n"

    if not summaries_block:
        summaries_block = "(no summaries retrieved)"

    prompt = ROUTER_THINKING_PROMPT_V2.format(
        question=question,
        summaries_block=summaries_block,
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = (resp.choices[0].message.content or "").strip()

        logger.info(f"GPT-5 raw response: {content[:300] if content else '(empty)'}...")

        if not content:
            print(resp)
            logger.warning("GPT-5 returned empty response")
            return RouterDecision(action="R", raw_response="")

        # 增强的 JSON 解析
        json_str = None

        # 策略1: 直接尝试解析（如果返回的就是纯 JSON）
        if content.startswith("{"):
            json_str = content

        # 策略2: 提取 ```json ... ``` 块
        if json_str is None:
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if match:
                json_str = match.group(1)

        # 策略3: 更宽松的提取（处理嵌套的情况）
        if json_str is None:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                json_str = match.group(0)

        if not json_str:
            logger.warning(f"No JSON found in response: {content[:100]}...")
            # 尝试从文本中提取 action
            action = "R"
            if '"action"' in content.lower():
                if '"s"' in content.lower() or "'s'" in content.lower():
                    action = "S"
            return RouterDecision(action=action, raw_response=content)

        # 清理 JSON 字符串
        json_str = json_str.strip()

        # 尝试解析
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}, trying to fix...")
            # 尝试修复常见问题
            # 1. 移除末尾逗号
            fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
            # 2. 修复单引号
            fixed = fixed.replace("'", '"')
            try:
                result = json.loads(fixed)
            except:
                logger.error(f"JSON parse failed even after fix: {json_str[:100]}...")
                return RouterDecision(action="R", raw_response=content)

        # 提取字段
        thinking = result.get("thinking", "")
        if isinstance(thinking, dict):
            # 兼容旧版嵌套格式，转换为字符串
            parts = []
            if thinking.get("question_needs"):
                parts.append(f"The question asks for {thinking['question_needs']}.")
            if thinking.get("summaries_have"):
                parts.append(f"The summaries contain {thinking['summaries_have']}.")
            if thinking.get("coverage"):
                parts.append(f"Coverage: {thinking['coverage']}.")
            if thinking.get("red_flags"):
                parts.append(f"Red flags: {', '.join(thinking['red_flags'])}.")
            thinking = " ".join(parts)

        action = result.get("action", "R")
        if isinstance(action, str):
            action = action.upper().strip()
        else:
            action = "R"

        # 规范化 action (只允许 S 或 R)
        if action not in ("S", "R"):
            if action in ("SUMMARY", "SUMMARIES"):
                action = "S"
            else:
                action = "R"

        decision = RouterDecision(
            thinking=thinking,
            action=action,
            raw_response=content,
        )

        return decision

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"GPT-5 router decision failed: {e}")
        return RouterDecision(
            action="R",
            thinking=f"Error: {e}",
            raw_response="",
        )


# ============ 数据加载 ============

def load_qa_results(results_dir: Path) -> Dict[str, Dict]:
    """加载 QA 结果"""
    results = {}
    sessions_dir = results_dir / "sessions"

    if sessions_dir.exists():
        for f in sessions_dir.glob("*_qa.jsonl"):
            with open(f) as fp:
                for line in fp:
                    data = json.loads(line)
                    results[data["query_id"]] = data
    else:
        qa_file = results_dir / "qa_logs.jsonl"
        if qa_file.exists():
            with open(qa_file) as f:
                for line in f:
                    data = json.loads(line)
                    results[data["query_id"]] = data

    return results


def sample_all_queries(
    qa_results: Dict[str, Dict],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    从所有 QA 样本中采样（不限定 S✗ R✓）

    返回：
    [
        {
            "query_id": str,
            "session_id": str,
            "question": str,
            "ground_truth": str,
            "category": int (可选),
        },
        ...
    ]
    """
    samples = []
    for qid, data in qa_results.items():
        samples.append({
            "query_id": qid,
            "session_id": data.get("session_id", ""),
            "question": data["question"],
            "ground_truth": data.get("ground_truth", [""])[0] if isinstance(data.get("ground_truth"), list) else data.get("ground_truth", ""),
            "category": data.get("category"),
        })

    if limit:
        samples = samples[:limit]

    return samples


# ============ 主处理流程 ============

def process_sample(
    sample: Dict[str, Any],
    system: LinkedViewSystem,
    client: OpenAI,
    gpt5_model: str = "gpt-5",
    judge_model: str = "gpt-4.1",
) -> TrainingSample:
    """
    处理单个样本（S/R 二分类版本）

    流程：
    1. 检索 summaries
    2. GPT-5 决策（带思维链）
    3. Coverage Judge 评估 summaries 是否足够
    4. 确定 optimal_label（ground truth）: S 或 R
    """
    query_id = sample["query_id"]
    session_id = sample["session_id"]
    question = sample["question"]
    ground_truth = sample["ground_truth"]
    category = sample.get("category")

    logger.info(f"Processing {query_id} (session: {session_id})")

    # 1. 加载 session
    try:
        system.load(session_id)
    except Exception as e:
        logger.error(f"Failed to load session {session_id}: {e}")
        return TrainingSample(
            query_id=query_id,
            session_id=session_id,
            question=question,
            ground_truth=ground_truth,
            summaries=[],
            scores=[],
            gpt5_decision=RouterDecision(action="R", thinking=f"Error: {e}"),
            coverage_judgement=CoverageJudgement(False, "ERROR", str(e)),
            category=category,
        )

    # 2. 检索
    user_id = f"{session_id}:user"
    try:
        hits = system.index.search(user_id=user_id, query=question, top_k=5)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return TrainingSample(
            query_id=query_id,
            session_id=session_id,
            question=question,
            ground_truth=ground_truth,
            summaries=[],
            scores=[],
            gpt5_decision=RouterDecision(action="R", thinking=f"Error: {e}"),
            coverage_judgement=CoverageJudgement(False, "ERROR", str(e)),
            category=category,
        )

    summaries = [h.summary_text for h in hits if h.summary_text]
    scores = [h.score for h in hits if h.score is not None]

    if not summaries:
        logger.warning(f"No summaries for {query_id}")
        return TrainingSample(
            query_id=query_id,
            session_id=session_id,
            question=question,
            ground_truth=ground_truth,
            summaries=[],
            scores=[],
            gpt5_decision=RouterDecision(action="R", thinking="No summaries"),
            coverage_judgement=CoverageJudgement(False, "NO_SUMMARIES", ""),
            category=category,
        )

    # 3. GPT-5 决策（带思维链）
    gpt5_decision = gpt5_router_decide_v2(
        client=client,
        question=question,
        summaries=summaries,
        scores=scores,
        model=gpt5_model,
    )
    logger.info(f"  GPT-5 decision: {gpt5_decision.action}")

    # 4. Coverage Judge（严格评估 summaries 是否足够）
    has_info, label, reason = judge_summary_coverage(
        client=client,
        question=question,
        gold_answer=ground_truth,
        summaries=summaries,
        model=judge_model,
    )
    coverage_judgement = CoverageJudgement(has_info, label, reason)
    logger.info(f"  Coverage: {label} - {reason}")

    # 5. 确定 optimal_label（ground truth）: S 或 R
    optimal_label = "S" if has_info else "R"

    # 构建最终结果
    result = TrainingSample(
        query_id=query_id,
        session_id=session_id,
        question=question,
        ground_truth=ground_truth,
        summaries=summaries,
        scores=scores,
        gpt5_decision=gpt5_decision,
        coverage_judgement=coverage_judgement,
        optimal_label=optimal_label,
        category=category,
    )

    return result


def determine_optimal_label(
    initial_coverage: CoverageJudgement,
) -> str:
    """
    确定训练标签（ground truth）- S/R 二分类

    规则：
    1. 如果 summaries 已经 SUFFICIENT → optimal = "S"
    2. 否则 → optimal = "R"
    """
    if initial_coverage.has_sufficient_info:
        return "S"
    return "R"


# ============ 主函数 ============

def enhance_existing_data(
    input_path: Path,
    output_path: Path,
    client: OpenAI,
    gpt5_model: str = "gpt-5",
    judge_model: str = "gpt-4.1",
    re_decide: bool = False,
    delay: float = 1.0,
    limit: Optional[int] = None,
    max_workers: int = 1,
) -> Dict[str, int]:
    """
    增强已有的训练数据（S/R 二分类版本）：
    1. 添加 coverage_judgement（基于 summaries 是否包含答案）
    2. 可选：用改进的思维链重新跑 GPT-5 决策
    3. 重新计算 optimal_label（S 或 R）

    Args:
        input_path: 已有数据文件路径
        output_path: 增强后的输出路径
        client: OpenAI 客户端
        gpt5_model: GPT-5 模型名
        judge_model: Judge 模型
        re_decide: 是否用新 prompt 重新跑 GPT-5 决策
        delay: API 调用间隔（单线程时使用）
        limit: 最多处理样本数
        max_workers: 并发线程数

    Returns:
        统计信息
    """
    logger.info(f"Enhancing existing data from {input_path}")
    logger.info(f"  Using {max_workers} workers")

    # 加载已有数据
    samples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except:
                    pass
    logger.info(f"  Loaded {len(samples)} samples")

    if limit:
        samples = samples[:limit]
        logger.info(f"  Limited to {len(samples)} samples")

    # 断点续跑
    processed_ids = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        processed_ids.add(data.get("query_id"))
                    except:
                        pass
        logger.info(f"  Already enhanced: {len(processed_ids)} samples")

    # 过滤掉已处理的样本
    samples_to_process = []
    for i, sample in enumerate(samples):
        qid = sample.get("query_id", f"unknown_{i}")
        if qid not in processed_ids:
            sample["_idx"] = i
            sample["_qid"] = qid
            samples_to_process.append(sample)

    logger.info(f"  Samples to process: {len(samples_to_process)}")

    if not samples_to_process:
        logger.info("  All samples already processed!")
        return {"total": 0, "coverage_sufficient": 0, "coverage_insufficient": 0,
                "optimal_S": 0, "optimal_R": 0, "re_decided": 0, "errors": 0}

    # 线程安全的统计和写入
    stats = {
        "total": 0,
        "coverage_sufficient": 0,
        "coverage_insufficient": 0,
        "optimal_S": 0,
        "optimal_R": 0,
        "re_decided": 0,
        "errors": 0,
    }
    stats_lock = threading.Lock()
    file_lock = threading.Lock()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def process_single_sample(sample: Dict) -> Optional[Dict]:
        """处理单个样本（线程安全）"""
        sample.pop("_idx", None)  # 移除临时字段
        qid = sample.pop("_qid")

        try:
            # 提取已有字段
            question = sample.get("question", "")
            ground_truth = sample.get("ground_truth", "")

            # 兼容两种数据格式
            summaries = sample.get("summaries") or sample.get("original_summaries", [])
            scores = sample.get("scores") or sample.get("original_scores", [])

            # 1. 对初始 summaries 做 coverage_judgement
            has_info, label, reason = judge_summary_coverage(
                client=client,
                question=question,
                gold_answer=ground_truth,
                summaries=summaries,
                model=judge_model,
            )
            coverage_judgement = {
                "has_sufficient_info": has_info,
                "label": label,
                "reason": reason,
            }

            # 2. 可选：重新跑 GPT-5 决策
            new_gpt5_decision = None
            if re_decide and summaries:
                new_decision = gpt5_router_decide_v2(
                    client=client,
                    question=question,
                    summaries=summaries,
                    scores=scores,
                    model=gpt5_model,
                )
                new_gpt5_decision = asdict(new_decision)

            # 3. 重新计算 optimal_label（S/R 二分类）
            optimal_label = "S" if has_info else "R"

            # 4. 构建增强后的样本
            enhanced_sample = dict(sample)
            enhanced_sample["coverage_judgement"] = coverage_judgement
            enhanced_sample["optimal_label_v2"] = optimal_label
            if new_gpt5_decision:
                enhanced_sample["gpt5_decision_v2"] = new_gpt5_decision

            return {
                "qid": qid,
                "sample": enhanced_sample,
                "has_info": has_info,
                "optimal_label": optimal_label,
                "re_decided": new_gpt5_decision is not None,
                "error": None,
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "qid": qid,
                "sample": None,
                "error": str(e),
            }

    # 并发处理
    with open(output_path, "a") as f:
        if max_workers <= 1:
            # 单线程模式
            for sample in tqdm(samples_to_process, desc="Enhancing"):
                result = process_single_sample(sample)

                if result["error"]:
                    stats["errors"] += 1
                    logger.error(f"  Error processing {result['qid']}: {result['error']}")
                else:
                    # 写入文件
                    f.write(json.dumps(result["sample"], ensure_ascii=False) + "\n")
                    f.flush()

                    # 更新统计
                    stats["total"] += 1
                    if result["has_info"]:
                        stats["coverage_sufficient"] += 1
                    else:
                        stats["coverage_insufficient"] += 1
                    stats[f"optimal_{result['optimal_label']}"] += 1
                    if result["re_decided"]:
                        stats["re_decided"] += 1

                if delay > 0:
                    time.sleep(delay)
        else:
            # 多线程模式
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_single_sample, sample): sample
                          for sample in samples_to_process}

                for future in tqdm(as_completed(futures), total=len(futures), desc="Enhancing"):
                    result = future.result()

                    with stats_lock:
                        if result["error"]:
                            stats["errors"] += 1
                            logger.error(f"  Error processing {result['qid']}: {result['error']}")
                        else:
                            # 写入文件（加锁）
                            with file_lock:
                                f.write(json.dumps(result["sample"], ensure_ascii=False) + "\n")
                                f.flush()

                            # 更新统计
                            stats["total"] += 1
                            if result["has_info"]:
                                stats["coverage_sufficient"] += 1
                            else:
                                stats["coverage_insufficient"] += 1
                            stats[f"optimal_{result['optimal_label']}"] += 1
                            if result["re_decided"]:
                                stats["re_decided"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="GPT-5 Router 数据收集（S/R 二分类版本）")

    # 模式选择
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--results-dir", type=str, help="实验结果目录（包含 QA logs）- 新数据收集模式")
    mode_group.add_argument("--enhance-existing", type=str, help="已有数据文件路径 - 增强模式")

    parser.add_argument("--output", type=str, required=True, help="输出文件路径")
    parser.add_argument("--gpt5-model", type=str, default="gpt-5", help="Router 决策模型（默认 gpt-5）")
    parser.add_argument("--judge-model", type=str, default="gpt-4.1", help="Judge 模型")
    parser.add_argument("--limit", type=int, default=None, help="最多处理样本数")
    parser.add_argument("--delay", type=float, default=0.0, help="API 调用间隔（秒，仅单线程模式）")
    parser.add_argument("--max-workers", type=int, default=10, help="并发线程数（默认 10）")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="系统 LLM 模型")
    parser.add_argument("--re-decide", action="store_true", help="增强模式下重新跑 GPT-5 决策（使用新 prompt）")
    args = parser.parse_args()

    # OpenAI 客户端
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    logger.info("OpenAI client created")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ============ 增强模式 ============
    if args.enhance_existing:
        logger.info("="*60)
        logger.info("增强模式：处理已有数据（S/R 二分类）")
        logger.info(f"并发线程数: {args.max_workers}")
        logger.info("="*60)

        stats = enhance_existing_data(
            input_path=Path(args.enhance_existing),
            output_path=output_path,
            client=client,
            gpt5_model=args.gpt5_model,
            judge_model=args.judge_model,
            re_decide=args.re_decide,
            delay=args.delay,
            limit=args.limit,
            max_workers=args.max_workers,
        )

        # 保存统计
        stats_path = output_path.parent / "enhance_stats_v2.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        # 打印总结
        print(f"\n{'='*60}")
        print("数据增强完成（S/R 二分类版本）")
        print(f"{'='*60}")
        print(f"总样本数: {stats['total']}")
        print(f"\nCoverage 分布（初始 summaries）:")
        print(f"  SUFFICIENT: {stats['coverage_sufficient']} ({100*stats['coverage_sufficient']/max(stats['total'],1):.1f}%)")
        print(f"  INSUFFICIENT: {stats['coverage_insufficient']} ({100*stats['coverage_insufficient']/max(stats['total'],1):.1f}%)")
        print(f"\nOptimal Label 分布:")
        print(f"  S: {stats['optimal_S']} ({100*stats['optimal_S']/max(stats['total'],1):.1f}%)")
        print(f"  R: {stats['optimal_R']} ({100*stats['optimal_R']/max(stats['total'],1):.1f}%)")
        if args.re_decide:
            print(f"\n重新决策: {stats['re_decided']} 样本")
        print(f"\n错误: {stats['errors']}")
        print(f"\n结果: {output_path}")
        print(f"统计: {stats_path}")
        return

    # ============ 新数据收集模式 ============
    logger.info("="*60)
    logger.info("新数据收集模式（S/R 二分类）")
    logger.info("="*60)

    # LinkedViewSystem 配置
    lv_cfg = {
        "benchmark_name": "memory_agent_bench",
        "mem0_config": {
            "backend": "mem0",
            "llm": {
                "provider": "openai",
                "config": {"model": args.model},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": "localhost",
                    "port": 6333,
                    "collection_name": "temp",
                },
            },
        },
        "page_size": 4000,
        "fast_model": args.model,
        "slow_model": args.model,
        "top_k": 5,
    }

    system = LinkedViewSystem(lv_cfg)
    logger.info("LinkedViewSystem created")

    # 加载数据
    logger.info(f"Loading QA results from {args.results_dir}...")
    qa_results = load_qa_results(Path(args.results_dir))
    logger.info(f"  Loaded {len(qa_results)} questions")

    # 采样（从所有 QA）
    samples = sample_all_queries(qa_results, limit=args.limit)
    logger.info(f"Sampled {len(samples)} queries")

    # 断点续跑
    processed_ids = set()
    if output_path.exists():
        logger.info(f"Found existing output {output_path}, loading...")
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        processed_ids.add(data.get("query_id"))
                    except:
                        pass
        logger.info(f"  Already processed: {len(processed_ids)} samples")

    # 统计
    stats = {
        "total": 0,
        "optimal_S": 0,
        "optimal_R": 0,
        "gpt5_S": 0,
        "gpt5_R": 0,
        "errors": 0,
    }

    file_mode = "a" if output_path.exists() else "w"
    with open(output_path, file_mode) as f:
        for i, sample in enumerate(tqdm(samples, desc="Processing")):
            qid = sample["query_id"]

            if qid in processed_ids:
                logger.info(f"[{i+1}/{len(samples)}] Skip {qid} (already processed)")
                continue

            logger.info(f"[{i+1}/{len(samples)}] Processing {qid}")

            try:
                result = process_sample(
                    sample=sample,
                    system=system,
                    client=client,
                    gpt5_model=args.gpt5_model,
                    judge_model=args.judge_model,
                )

                stats["total"] += 1
                stats[f"optimal_{result.optimal_label}"] += 1
                stats[f"gpt5_{result.gpt5_decision.action}"] += 1

                f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                f.flush()
                processed_ids.add(qid)

            except Exception as e:
                logger.error(f"  Error: {e}")
                stats["errors"] += 1

            if args.delay > 0:
                time.sleep(args.delay)

    # 保存统计
    stats_path = output_path.parent / "collection_stats_v2.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # 打印总结
    print(f"\n{'='*60}")
    print("数据收集完成（S/R 二分类版本）")
    print(f"{'='*60}")
    print(f"总样本数: {stats['total']}")
    print(f"\nOptimal Label 分布:")
    print(f"  S: {stats['optimal_S']} ({100*stats['optimal_S']/max(stats['total'],1):.1f}%)")
    print(f"  R: {stats['optimal_R']} ({100*stats['optimal_R']/max(stats['total'],1):.1f}%)")
    print(f"\nGPT-5 决策分布:")
    print(f"  S: {stats['gpt5_S']} ({100*stats['gpt5_S']/max(stats['total'],1):.1f}%)")
    print(f"  R: {stats['gpt5_R']} ({100*stats['gpt5_R']/max(stats['total'],1):.1f}%)")
    print(f"\n错误: {stats['errors']}")
    print(f"\n结果: {output_path}")
    print(f"统计: {stats_path}")


if __name__ == "__main__":
    main()
