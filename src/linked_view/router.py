"""
Binary Router：只看 (Q, Z) 决定走 S 还是 R。

实现目标：
- 线上：轻量推理，输出 {"S", "R"}
- 线下：支持使用 RouteExample 数据做监督训练（最小可用版）

在此基础上，增加一个基于 LLM 的简易 Router（LLMRouter），
直接让一个小模型读 query + summaries，输出 "S" 或 "R"。
"""

from typing import Any, List, Optional
import logging
logger = logging.getLogger(__name__)
import math

from .summary_index import SummaryHit, EvidenceSet


class RouteExample:
    """离线训练样本：用于 supervised router。"""

    def __init__(self, query: str, hits: List[SummaryHit], y_route_raw: int):
        self.query = query
        self.hits = hits
        self.y_route_raw = y_route_raw  # 1 表示应走 R（Raw），0 表示 S


class BinaryRouter:
    """
    最小可用 Router：
    - 支持基于 hand-crafted feature 的逻辑决策
    - 也支持注入一个轻量模型（例如小型 MLP / 逻辑回归），统一走 sigmoid + 阈值

    说明：
    - 这里不直接依赖 torch / sklearn，保持“无外部依赖可跑”
    - 若传入 model_fn，则应满足：p = model_fn(features: List[float])，返回 [0,1] 概率
    """

    def __init__(self, threshold: float = 0.5, model_fn: Optional[Any] = None):
        self.threshold = float(threshold)
        self.model_fn = model_fn

    # === 线上推理 ===
    def decide(self, evidence: EvidenceSet) -> str:
        """决策走 S 还是 R。"""

        feats = self._extract_features(evidence)

        if self.model_fn is not None:
            try:
                p_raw = float(self.model_fn(feats))
            except Exception:
                # 模型故障时回退到启发式
                p_raw = self._heuristic_p_raw(evidence, feats)
        else:
            p_raw = self._heuristic_p_raw(evidence, feats)

        return "R" if p_raw > self.threshold else "S"

    # === 特征抽取（只依赖 summaries） ===
    def _extract_features(self, evidence: EvidenceSet) -> List[float]:
        """线上特征：scores / coverage / 时间跨度 / query 长度等。"""

        hits = evidence.hits
        scores = [h.score for h in hits if h.score is not None]
        q_tokens = evidence.query.split()

        if scores:
            s_max = max(scores)
            s_mean = sum(scores) / len(scores)
            s_var = sum((s - s_mean) ** 2 for s in scores) / len(scores)
            s_std = math.sqrt(s_var)
            # 简单熵：归一化后计算
            s_sum = sum(scores) or 1.0
            probs = [s / s_sum for s in scores]
            entropy = -sum(p * math.log(p + 1e-8) for p in probs)
        else:
            s_max = s_mean = s_std = entropy = 0.0

        total_sum_len = sum(len(h.summary_text) for h in hits)
        hit_ratio = len(hits)

        # 时间跨度：这里暂时用是否跨 session / 是否有 timestamp 作为两个粗特征
        has_ts = any(h.timestamp for h in hits)
        multi_session = len({h.session_id for h in hits if h.session_id is not None}) > 1

        q_len = len(q_tokens)

        return [
            float(s_max),
            float(s_mean),
            float(s_std),
            float(entropy),
            float(total_sum_len),
            float(hit_ratio),
            1.0 if has_ts else 0.0,
            1.0 if multi_session else 0.0,
            float(q_len),
        ]

    # === 启发式概率（不依赖任何模型） ===
    def _heuristic_p_raw(self, evidence: EvidenceSet, feats: List[float]) -> float:
        """根据启发式估计走 R 的概率。"""

        hits = evidence.hits
        q = evidence.query.lower()

        s_max, s_mean, s_std, entropy, total_sum_len, hit_ratio, has_ts, multi_session, q_len = feats

        # 关键词：最近 / 上个月 / 变化 / 对比 / 总结 / 为什么
        temporal_keywords = ["最近", "上个月", "去年", "之前", "之后", "变化", "对比"]
        causal_keywords = ["为什么", "原因", "导致", "how did", "how do", "分析", "总结"]

        has_temporal = any(k in q for k in temporal_keywords)
        has_causal = any(k in q for k in causal_keywords)

        # 低分 / 高熵 / 记忆少 时更需要 R
        lack_evidence = (s_max < 0.3) or (len(hits) < 3)
        high_uncertainty = entropy > 1.5

        score = 0.0
        if q_len > 20:
            score += 0.2
        if has_temporal:
            score += 0.25
        if has_causal:
            score += 0.25
        if lack_evidence:
            score += 0.2
        if high_uncertainty:
            score += 0.1

        # 压到 [0,1]
        return max(0.0, min(1.0, score))


class QueryRewriter:
    """
    Query 改写器：维护一个 "summary的summary"（改写说明书），指导query改写。

    核心功能：
    1. build_guide_from_summaries: 从现有 memories 自动提取主题/关键词/实体，生成改写指南
    2. rewrite_query: 基于指南改写 query，使其更适合检索
    """

    def __init__(self, llm: Any, guide_text: str = ""):
        """
        llm: 需要实现 generate(prompt: str) -> str
        guide_text: 初始的改写指南（可选，也可后续通过 build_guide 生成）
        """
        self.llm = llm
        self.guide_text = guide_text

    def build_guide_from_summaries(self, summaries: List[str], max_guide_len: int = 500) -> str:
        """
        从一批 summaries 中提取关键主题/实体/关键词，生成一个简洁的改写指南。

        参数：
            summaries: 现有的 memory summaries 列表
            max_guide_len: 生成的指南最大字符数

        返回：
            生成的 guide_text，同时更新 self.guide_text
        """
        if not summaries:
            self.guide_text = ""
            return ""

        # 合并所有 summaries，限制总长度避免超token
        combined = "\n".join(summaries[:50])  # 最多取50条
        if len(combined) > 5000:
            combined = combined[:5000] + "..."

        prompt = (
            "You are an expert in building query rewriting guides for a memory system.\n"
            "Given a list of memory summaries below, extract:\n"
            "1. Main topics/themes that appear frequently\n"
            "2. Key entities (people, places, events, concepts)\n"
            "3. Common keywords and phrases\n"
            "4. Typical query patterns\n\n"
            f"Output a concise guide (max {max_guide_len} chars) in the following format:\n"
            "Topics: [topic1, topic2, ...]\n"
            "Entities: [entity1, entity2, ...]\n"
            "Keywords: [keyword1, keyword2, ...]\n"
            "Query patterns: [pattern1, pattern2, ...]\n\n"
            f"Memory summaries:\n{combined}\n\n"
            "Guide:"
        )

        try:
            guide = self.llm.generate(prompt) or ""
            guide = guide.strip()[:max_guide_len]
            self.guide_text = guide
            return guide
        except Exception as exc:
            print(f"[QueryRewriter] build_guide failed: {exc}")
            return ""

    def rewrite_query(self, query: str) -> str:
        """
        基于改写指南，将原始 query 改写成更适合检索的形式。

        策略：
        - 扩展关键词（基于 guide 中的 topics/entities）
        - 补充相关术语
        - 规范化表达

        返回：
            改写后的 query（如果改写失败则返回原始 query）
        """
        if not self.guide_text:
            return query  # 没有指南时直接返回原query

        prompt = (
            "You are a query rewriter for a memory retrieval system.\n"
            "Your job: rewrite the user's query to make it better for semantic search.\n\n"
            "You have access to a guide that describes the memory library content:\n"
            f"{self.guide_text}\n\n"
            "Rewriting rules:\n"
            "- Expand abbreviations or vague terms based on the guide\n"
            "- Add relevant keywords/entities mentioned in the guide if applicable\n"
            "- Keep the query concise (ideally 10-30 words)\n"
            "- Preserve the original intent and question structure\n"
            "- If the query is already good, return it as-is\n\n"
            f"Original query:\n{query}\n\n"
            "Rewritten query (output ONLY the rewritten query, no explanations):"
        )

        try:
            rewritten = self.llm.generate(prompt) or query
            rewritten = rewritten.strip()
            # 简单校验：如果改写后太长或太短，回退到原query
            if len(rewritten) < 3 or len(rewritten) > 200:
                return query
            return rewritten
        except Exception as exc:
            print(f"[QueryRewriter] rewrite_query failed: {exc}")
            return query


class LLMRouter:
    """
    简易 LLM Router：
    - 直接把 (Q, summaries) 丢给一个小模型（如 gpt-4o-mini），让它回答 "S" 或 "R"
    - 方便快速对路由策略做 prompt 级调参
    - 新增：支持 QueryRewriter 进行 query 改写，实现双路召回
    """

    def __init__(self, llm: Any, threshold: float = 0.5, query_rewriter: Optional[QueryRewriter] = None):
        """
        llm: 需要实现 generate(prompt: str) -> str
        threshold: 预留参数，目前 LLM 直接输出离散标签，不使用概率阈值
        query_rewriter: 可选的 QueryRewriter 实例，用于 query 改写
        """

        self.llm = llm
        self.threshold = float(threshold)
        self.query_rewriter = query_rewriter

    @staticmethod
    def parse_thinking_output(text: str) -> str:
        """
        解析 Thinking 模型输出，提取 </think> 后面的内容。

        支持格式：
        - <think>思考过程...</think>实际答案
        - 无 think 标签的普通输出

        Returns:
            提取后的实际答案文本
        """
        import re
        if not text:
            return ""

        # 尝试匹配 </think> 标签后的内容
        # 模式1：<think>...</think>答案
        think_pattern = r'</think>\s*(.*)$'
        match = re.search(think_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # 模式2：没有 think 标签，直接返回原文本
        # 检查是否有未闭合的 <think> 标签
        if '<think>' in text.lower() and '</think>' not in text.lower():
            # 未闭合的 think 标签，可能还在思考中，返回空
            return ""

        return text.strip()

    def get_rewritten_query(self, query: str) -> str:
        """
        使用 QueryRewriter 改写 query（如果配置了的话）。

        返回：
            改写后的 query，如果没有配置 query_rewriter 则返回原始 query
        """
        if self.query_rewriter is None:
            return query
        return self.query_rewriter.rewrite_query(query)

    def decompose_query(self, query: str, max_queries: int = 3) -> List[str]:
        """
        使用 LLM 将一个复杂 query 拆解成若干个子 query（用于多路检索）。

        返回值：
            - 最多 max_queries 条子 query
            - 出错时回退为 [query]
        """

        max_queries = max(1, int(max_queries))
        prompt = (
            "You are a query planner for a memory-augmented QA system.\n"
            "Given a user question, you should decompose it into at most "
            f"{max_queries} focused sub-questions that will be used for retrieval.\n"
            "Rules:\n"
            "- If the question is simple, you can just output the original question as a single line.\n"
            "- If it is complex or multi-part, split it into several concrete sub-questions.\n"
            "- Output ONLY the sub-questions, one per line, WITHOUT numbering or bullets.\n"
            "- Do NOT add any explanations.\n\n"
            f"User question:\n{query}\n\n"
            "Sub-questions (one per line, at most "
            f"{max_queries}"
            " lines):"
        )

        try:
            text = self.llm.generate(prompt) or ""
            lines = [ln.strip() for ln in text.splitlines()]
            # 过滤空行
            sub_queries: List[str] = [ln for ln in lines if ln]
            # 去掉可能的编号前缀（例如 "1. xxx", "- xxx"）
            cleaned: List[str] = []
            for q in sub_queries:
                # 简单去掉常见前缀
                q_strip = q.lstrip("-•* \t")
                # 去掉 "1. " / "2) " 之类
                if len(q_strip) > 2 and q_strip[0].isdigit() and q_strip[1] in {".", ")", "、"}:
                    q_strip = q_strip[2:].lstrip()
                if q_strip:
                    cleaned.append(q_strip)

            cleaned = cleaned[:max_queries]
            if not cleaned:
                return [query]
            return cleaned
        except Exception:
            # 出现异常时，直接回退为单一 query
            return [query]
    def decide(
        self,
        evidence: EvidenceSet,
        retrieval_round: int = 0,
        history: Optional[List[dict]] = None
    ) -> str:
        """
        让 LLM 读 query + summaries，判断走 S / R / 或用新 query 再检索。

        Args:
            evidence: 包含 query、hits 和 extra_context 的证据集
            retrieval_round: 当前检索轮次（0 表示第一次）
            history: 检索历史，格式为 [{"round": int, "query": str, "hits_count": int}, ...]

        Returns:
            - "S": Summary-only path (快速答案)
            - "R": Research path (深度研究) 

        出错时回退为 "R"（更安全）。
        """

        q = evidence.query
        summaries: List[str] = []
        for i, h in enumerate(evidence.hits):
            summaries.append(
                f"[{i}] score={h.score:.3f} ts={h.timestamp} s:{h.session_id or ''}\n{h.summary_text}"
            )
        summaries_block = "\n\n".join(summaries) if summaries else "(no summaries)"

        # 构建检索历史描述
        history = history or []
        history_text = ""
        if history:
            history_lines = [f"Round {h['round']}: Query=\"{h['query']}\", Found {h['hits_count']} hits" for h in history]
            history_text = "\n".join(history_lines)
        else:
            history_text = "(This is the first retrieval round)"

        # 判断是否还可以继续检索（最多 2 轮）
        max_rounds = 2
        can_query = retrieval_round < (max_rounds - 1)

        prompt = f"""You are a router for a memory-augmented QA system.

TASK: Decide the next action based on retrieved summaries.
- "S" = Answer using summaries only (fast path, when info is sufficient)
- "QUERY: <new_query>" = Reformulate query and retrieve again (when summaries are related but incomplete)
- "R" = Research full database (slow path, when summaries are irrelevant or question needs deep reasoning)

DECISION CRITERIA:
1. S: Summaries explicitly contain the answer → output "S"
2. QUERY: Summaries have related info but miss specific details → output "QUERY: <better query>"
3. R: Summaries are irrelevant OR question requires multi-hop reasoning → output "R"

Question: {q}

Retrieved Summaries:
{summaries_block}

Output your decision (S, QUERY: <query>, or R):"""

        try:
            text = self.llm.generate(prompt) or ""
            # 解析 thinking 模型输出（如果有 </think> 标签）
            t = self.parse_thinking_output(text) or text.strip()

            # 检查是否是 QUERY 格式
            t_upper = t.upper()
            if t_upper.startswith("QUERY:"):
                new_query = t[6:].strip()  # 去掉 "QUERY:" 前缀
                if new_query and can_query:
                    return f"QUERY: {new_query}"
                else:
                    # 如果已达到最大轮次或 query 为空，回退到 R
                    return "R"

            # 兼容旧格式 REFINE:
            if t_upper.startswith("REFINE:"):
                new_query = t[7:].strip()
                if new_query and can_query:
                    return f"QUERY: {new_query}"
                else:
                    return "R"

            # 检查 S/R
            label = ""
            for ch in t_upper:
                if ch.isalpha():
                    label = ch
                    break
            if label == "S":
                return "S"
            if label == "R":
                return "R"
        except Exception:
            # 出现异常时，保守起见走 R
            pass
        return "R"


class ThinkingLLMRouter:
    """
    支持 Thinking 模型的 Router（兼容 vLLM 和 OpenAI）。

    特性：
    1. 支持两种 action: S, R
    2. 自动解析 </think> 标签（Qwen3-Thinking 等模型）
    3. 兼容 OpenAI API 和 vLLM（通过 extra_body 参数）
    """

    def __init__(
        self,
        client: Any = None,
        model: str = "gpt-4.1-mini",
        is_thinking_model: bool = False,
        search_fn: Any = None,
        threshold: float = 0.5,
    ):
        """
        Args:
            client: OpenAI 客户端实例（支持 OpenAI API 和 vLLM）
            model: 模型名称
            is_thinking_model: 是否是 thinking 模型（如 Qwen3-Thinking），需要 extra_body
            search_fn: 搜索回调函数，签名: search_fn(query: str, top_k: int) -> List[SummaryHit]
            threshold: 预留参数
        """
        self.client = client
        self.model = model
        self.is_thinking_model = is_thinking_model
        self.search_fn = search_fn
        self.threshold = float(threshold)
        self.last_usage: dict = {}  # 存储最近一次调用的 token usage
        self.last_raw_response: str = ""  # 存储最近一次调用的原始响应（包含思维链）

    @staticmethod
    def parse_thinking_output(text: str) -> str:
        """
        解析 Thinking 模型输出，提取 </think> 后面的内容。
        """
        import re
        if not text:
            return ""

        # 匹配 </think> 后的内容
        think_pattern = r'</think>\s*(.*)$'
        match = re.search(think_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # 检查未闭合的 <think> 标签
        if '<think>' in text.lower() and '</think>' not in text.lower():
            return ""

        return text.strip()

    def _call_llm(self, prompt: str, max_tokens: int = 2048) -> tuple[str, dict]:
        """
        调用 LLM，自动处理 thinking 模型的 extra_body。

        Returns:
            (text, usage): 模型输出文本（已解析 </think>）和 token usage 信息
        """
        if self.client is None:
            return "", {}

        try:
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            }

            # thinking 模型需要 extra_body（vLLM）
            if self.is_thinking_model:
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": True},
                }

            response = self.client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content or ""
            logger.info(f"[ThinkingLLMRouter] Response: {text}")
            if text == "":
                logger.warning(f"[ThinkingLLMRouter] {response}")

            # 保存原始响应（包含思维链）
            self.last_raw_response = text

            # 提取 token usage
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(response.usage, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(response.usage, "total_tokens", 0) or 0),
                }

            # 解析 thinking 输出
            if self.is_thinking_model:
                parsed_text = self.parse_thinking_output(text)
                return parsed_text, usage
            return text.strip(), usage
        except Exception as e:
            print(f"[ThinkingLLMRouter] LLM call failed: {e}")
            return "", {}

    def _build_summaries_block(self, hits: List[SummaryHit]) -> str:
        """构建 summaries 文本块。"""
        if not hits:
            return "(no summaries)"

        summaries: List[str] = []
        for i, h in enumerate(hits):
            score = h.score if h.score is not None else 0.0
            summaries.append(
                f"[{i}] score={score:.3f} ts={h.timestamp or 'N/A'}\n{h.summary_text or ''}"
            )
        return "\n\n".join(summaries)

    def decide(
        self,
        evidence: EvidenceSet,
        retrieval_round: int = 0,
        history: Optional[List[dict]] = None
    ) -> str:
        """
        Router 决策（支持两阶段）。

        第一阶段：可以输出 S, R,


        Args:
            evidence: 包含 query、hits 和 extra_context 的证据集
            retrieval_round: 当前检索轮次（0 表示第一次）
            history: 检索历史

        Returns:
            - "S": Summary-only path
            - "R": Research path
        """
        q = evidence.query
        summaries_block = self._build_summaries_block(evidence.hits)

        prompt = f"""You are an expert router for a memory-augmented QA system.

Your task: Analyze the retrieved summaries and decide the best action to answer the question.

Available actions:
1. "S" - Answer using current summaries only 
    Use when: Summaries contain the EXPLICIT answer to the specific question.
    CRITICAL: Do not infer causes from effects (e.g., "benefits of X" is NOT "reason for starting X"). 
    If the exact answer is not verbatim in the text, or the question needs efficient details do not use S.

2. "R" - Deep research mode (slow path)
    Use when: Summaries are ambiguous, only contextually related, or miss the answer entirely.
    If you have ANY doubt whether the summaries allow a factual answer without guessing, choose R.
    If the question requires completeness (e.g., "how many times", "list all", "what are all"), prefer R to ensure comprehensive answers.


Question: {q}

Retrieved Summaries:
{summaries_block}

Output format (JSON only):
- If answering with summaries: {{"action": "S"}}
- If deep research needed: {{"action": "R"}}

Your response:"""

        # 记录 prompt
        logger.info(f"[ThinkingLLMRouter] Round {retrieval_round} - Prompt:\n{prompt}")
        
        text, usage = self._call_llm(prompt)
        
        # 存储 token usage（供外部访问）
        self.last_usage = usage
        
        # 记录生成的答案
        logger.info(f"[ThinkingLLMRouter] Round {retrieval_round} - Raw Response: {text}")
        
        if not text:
            logger.warning(f"[ThinkingLLMRouter] Round {retrieval_round} - Empty response, defaulting to S")
            return "S"

        # 尝试解析 JSON
        import json
        
        # 方法1: 尝试直接解析整个文本
        try:
            result = json.loads(text.strip())
            action = result.get("action", "").upper().strip()
            
            if action == "S":
                logger.info(f"[ThinkingLLMRouter] Round {retrieval_round} - Final Decision: S")
                return "S"
            elif action == "R":
                logger.info(f"[ThinkingLLMRouter] Round {retrieval_round} - Final Decision: R")
                return "R"
        except (json.JSONDecodeError, KeyError, AttributeError):
            # 方法2: 尝试提取 JSON 对象（查找第一个 { 到匹配的 }）
            start_idx = text.find('{')
            if start_idx >= 0:
                brace_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(text)):
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i
                            break
                
                if brace_count == 0 and end_idx > start_idx:
                    try:
                        json_str = text[start_idx:end_idx + 1]
                        result = json.loads(json_str)
                        action = result.get("action", "").upper().strip()
                        
                        if action == "S":
                            logger.info(f"[ThinkingLLMRouter] Round {retrieval_round} - Final Decision: S (extracted from JSON)")
                            return "S"
                        elif action == "R":
                            logger.info(f"[ThinkingLLMRouter] Round {retrieval_round} - Final Decision: R (extracted from JSON)")
                            return "R"
                    except (json.JSONDecodeError, KeyError, AttributeError):
                        pass  # JSON 解析失败，回退到文本解析
        
        # 回退到文本解析（向后兼容）
        t_upper = text.upper().strip()
        # 解析 S/R
        for ch in t_upper:
            if ch == "S":
                logger.info(f"[ThinkingLLMRouter] Round {retrieval_round} - Final Decision: S (parsed from text)")
                return "S"
            if ch == "R":
                logger.info(f"[ThinkingLLMRouter] Round {retrieval_round} - Final Decision: R (parsed from text)")
                return "R"

        logger.warning(f"[ThinkingLLMRouter] Round {retrieval_round} - Could not parse decision from response, defaulting to S")
        return "S"

    def decide_with_search(
        self,
        evidence: EvidenceSet,
        top_k: int = 5
    ) -> tuple:
        """
        完整的两阶段决策（带自动搜索）。

        如果第一阶段返回 QUERY，自动执行搜索并进行第二阶段决策。

        Args:
            evidence: 初始证据集
            top_k: 搜索时返回的结果数

        Returns:
            tuple: (final_action, all_hits, query_history)
            - final_action: "S" 或 "R"
            - all_hits: 所有检索到的 hits（包括 QUERY 后的）
            - query_history: 检索历史 [{"round": int, "query": str, "hits_count": int, "action": str}, ...]
        """
        query_history: List[dict] = []
        all_hits = list(evidence.hits)
        seen_ids = {h.raw_log_id for h in all_hits if h.raw_log_id}

        # 记录初始检索
        query_history.append({
            "round": 0,
            "query": evidence.query,
            "hits_count": len(evidence.hits),
            "action": "initial"
        })

        # 第一阶段决策
        action = self.decide(evidence, retrieval_round=0, history=query_history)

        if action.startswith("QUERY:") and self.search_fn is not None:
            # 提取新 query
            new_query = action[6:].strip()

            # 执行搜索
            new_hits = self.search_fn(new_query, top_k)

            # 合并结果（去重）
            for h in new_hits:
                if h.raw_log_id and h.raw_log_id not in seen_ids:
                    all_hits.append(h)
                    seen_ids.add(h.raw_log_id)

            # 记录 QUERY 检索
            query_history.append({
                "round": 1,
                "query": new_query,
                "hits_count": len(new_hits),
                "action": "QUERY"
            })

            # 构建合并后的 evidence
            merged_evidence = EvidenceSet(
                query=evidence.query,
                hits=all_hits,
                extra_context=evidence.extra_context
            )

            # 第二阶段决策（只能 S 或 R）
            final_action = self.decide(merged_evidence, retrieval_round=1, history=query_history)


            query_history.append({
                "round": 2,
                "query": evidence.query,
                "hits_count": len(all_hits),
                "action": final_action
            })

            return final_action, all_hits, query_history
        else:
            # 直接返回第一阶段结果
            query_history.append({
                "round": 1,
                "query": evidence.query,
                "hits_count": len(all_hits),
                "action": action
            })

            return action, all_hits, query_history


__all__ = ["RouteExample", "BinaryRouter", "QueryRewriter", "LLMRouter", "ThinkingLLMRouter"]
