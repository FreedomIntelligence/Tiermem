"""
GAM (General Agentic Memory) 适配器

参考实现：
- /share/home/qmzhu/AGMS/relatedwork/general-agentic-memory/eval/locomo_test.py
"""
import os
import time
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import tempfile
import shutil

from core.systems.base import MemorySystem, Turn, ObserveResult, AnswerResult

try:
    from gam import (
        MemoryAgent,
        ResearchAgent,
        InMemoryMemoryStore,
        InMemoryPageStore,
        IndexRetriever,
        OpenAIGenerator,
        OpenAIGeneratorConfig,
        IndexRetrieverConfig,
    )
    # 可选检索器（可能因为依赖问题无法导入）
    try:
        from gam import BM25Retriever, BM25RetrieverConfig
        BM25_AVAILABLE = True
    except ImportError:
        BM25Retriever = None
        BM25RetrieverConfig = None
        BM25_AVAILABLE = False
    
    try:
        from gam import DenseRetriever, DenseRetrieverConfig
        DENSE_AVAILABLE = True
    except ImportError:
        DenseRetriever = None
        DenseRetrieverConfig = None
        DENSE_AVAILABLE = False
    
    GAM_AVAILABLE = True
except ImportError as e:
    GAM_AVAILABLE = False
    print(f"Warning: GAM library not available: {e}")
    print("Install from /share/home/qmzhu/AGMS/relatedwork/general-agentic-memory")


# ========== LoCoMo专用Prompt函数（来自官方实现） ==========

def make_summary_prompt(summary: str, question: str) -> str:
    """
    生成基于研究摘要的答案prompt（官方GAM的方式）
    
    关键：要求生成短答案（short phrase），不是完整句子
    """
    return f"""\
Based on the summary below, write an answer in the form of **a short phrase** for the following question, not a sentence. Answer with exact words from the context whenever possible.
For questions that require answering a date or time, strictly follow the format \"15 July 2023\" and provide a specific date whenever possible. For example, if you need to answer \"last year,\" give the specific year of last year rather than just saying \"last year.\" Only provide one year, date, or time, without any extra responses.
If the question is about the duration, answer in the form of several years, months, or days.

QUESTION:
{question}

SUMMARY:
{summary}

Short answer:
"""


def make_summary_prompt_category3(summary: str, question: str) -> str:
    """
    Category 3专用prompt（需要分析和推理的问题）
    """
    return f"""\
Based on the summary below, write an answer in the form of **a short phrase** for the following question, not a sentence.
The question may need you to analyze and infer the answer from the summary.
    
QUESTION:
{question}

SUMMARY:
{summary}

Short answer:
"""


def answer_with_summary(category: Optional[int], summary: str, question: str, generator) -> str:
    """
    根据category选择不同的prompt，并使用generator生成短答案（官方GAM的方式）
    """
    if category == 3:
        prompt = make_summary_prompt_category3(summary, question)
    else:
        prompt = make_summary_prompt(summary, question)
    
    try:
        raw = generator.generate_single(prompt=prompt)
        if raw is None:
            print(f"⚠ Warning: generator.generate_single() returned None")
            return ""
        
        answer_text = raw.get("text", "")
        if not answer_text:
            print(f"⚠ Warning: generator.generate_single() returned empty text. Raw response: {raw}")
            print(f"  Prompt length: {len(prompt)}, Summary length: {len(summary)}, Question: {question[:50]}...")
            return str(raw)
        return answer_text.strip()
    except Exception as e:
        print(f"⚠ Error in answer_with_summary: {e}")
        import traceback
        traceback.print_exc()
        return ""


class GAMSystem(MemorySystem):
    """
    GAM系统适配器
    
    GAM的工作流程：
    1. 使用MemoryAgent.memorize()将对话内容存储为记忆
    2. 使用ResearchAgent.research()研究问题并生成摘要
    3. 基于研究摘要生成答案
    
    关键：GAM使用session_chunk粒度（官方方式），而不是逐句turn
    """
    # 指定使用session_chunks（粗粒度），而不是默认的turns（细粒度）
    preferred_turns_key = "session_chunks"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        if not GAM_AVAILABLE:
            raise ImportError("GAM library not available. Please install it first.")
        
        # 配置参数
        self.memory_model = config.get("memory_model", "gpt-5-mini")
        self.research_model = config.get("research_model", "gpt-5-mini")
        self.working_model = config.get("working_model", "gpt-5-mini")
        self.memory_api_key = config.get("memory_api_key") or os.getenv("OPENAI_API_KEY")
        self.research_api_key = config.get("research_api_key") or os.getenv("OPENAI_API_KEY")
        self.working_api_key = config.get("working_api_key") or os.getenv("OPENAI_API_KEY")
        self.memory_base_url = config.get("memory_base_url")
        self.research_base_url = config.get("research_base_url")
        self.working_base_url = config.get("working_base_url")
        self.max_research_iters = config.get("max_research_iters", 3)
        
        # 存储组件（在reset时初始化）
        self.memory_store = None
        self.page_store = None
        self.memory_agent = None
        self.research_agent = None
        self.retrievers = {}
        
        # 用于存储临时目录
        self.temp_dir = None
        self.session_dir = None
        
        # 用于记录cost
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_api_calls = 0
        
        # 累积turns（用于GAM的memorize）
        self.accumulated_turns = []
        
    def reset(self, session_id: str) -> None:
        """重置系统状态，开始新的会话"""
        super().reset(session_id)
        
        # 清理之前的资源
        if self.session_dir and os.path.exists(self.session_dir):
            try:
                shutil.rmtree(self.session_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up session dir: {e}")
        
        # 创建临时目录用于存储GAM的状态
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="gam_")
        
        self.session_dir = os.path.join(self.temp_dir, session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        
        # 重置cost统计
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_api_calls = 0
        
        # 重置累积的turns
        self.accumulated_turns = []
        
        # 初始化存储
        self.memory_store = InMemoryMemoryStore(dir_path=self.session_dir)
        self.page_store = InMemoryPageStore(dir_path=self.session_dir)
        
        # 创建Memory Generator
        memory_generator_config = OpenAIGeneratorConfig(
            model_name=self.memory_model,
            api_key=self.memory_api_key,
            base_url=self.memory_base_url,
            temperature=0.3,
            max_tokens=2048
        )
        memory_generator = OpenAIGenerator(memory_generator_config.__dict__)
        
        # 创建MemoryAgent
        self.memory_agent = MemoryAgent(
            memory_store=self.memory_store,
            page_store=self.page_store,
            generator=memory_generator
        )
        
        # 创建Research Generator和Working Generator
        research_generator_config = OpenAIGeneratorConfig(
            model_name=self.research_model,
            api_key=self.research_api_key,
            base_url=self.research_base_url,
            temperature=0.3,
            max_tokens=2048
        )
        research_generator = OpenAIGenerator(research_generator_config.__dict__)
        
        working_generator_config = OpenAIGeneratorConfig(
            model_name=self.working_model,
            api_key=self.working_api_key,
            base_url=self.working_base_url,
            temperature=0.3,
            max_tokens=2048
        )
        # 关键：保存working_generator到self，供answer方法使用
        self.working_generator = OpenAIGenerator(working_generator_config.__dict__)
        
        # 创建检索器（延迟构建，在第一次answer时构建）
        self.retrievers = {}
        self._retrievers_built = False
        
        # 创建ResearchAgent（检索器稍后设置）
        self.research_agent = ResearchAgent(
            page_store=self.page_store,
            memory_store=self.memory_store,
            retrievers={},  # 稍后设置
            generator=research_generator,
            max_iters=self.max_research_iters
        )
    
    def _build_retrievers(self):
        """构建检索器（只在第一次answer时调用）"""
        if self._retrievers_built:
            return
        
        try:
            # Index检索器
            page_index_dir = os.path.join(self.session_dir, "page_index")
            if os.path.exists(page_index_dir):
                shutil.rmtree(page_index_dir)
            
            index_config = IndexRetrieverConfig(index_dir=page_index_dir)
            index_retriever = IndexRetriever(index_config.__dict__)
            index_retriever.build(self.page_store)
            self.retrievers["page_index"] = index_retriever
        except Exception as e:
            print(f"Warning: Failed to build IndexRetriever: {e}")
        
        # BM25检索器（可选，需要Java/pyserini）
        if BM25_AVAILABLE and BM25Retriever is not None:
            try:
                bm25_index_dir = os.path.join(self.session_dir, "bm25_index")
                if os.path.exists(bm25_index_dir):
                    shutil.rmtree(bm25_index_dir)
                
                bm25_config = BM25RetrieverConfig(index_dir=bm25_index_dir, threads=1)
                bm25_retriever = BM25Retriever(bm25_config.__dict__)
                bm25_retriever.build(self.page_store)
                self.retrievers["keyword"] = bm25_retriever
                print("  ✓ BM25Retriever built successfully")
            except Exception as e:
                print(f"  ⚠ Failed to build BM25Retriever: {e}")
        else:
            print("  ⚠ BM25Retriever not available (Java/pyserini dependencies missing)")
        
        # Dense检索器（可选，需要FlagEmbedding）
        if DENSE_AVAILABLE and DenseRetriever is not None:
            try:
                dense_index_dir = os.path.join(self.session_dir, "dense_index")
                if os.path.exists(dense_index_dir):
                    shutil.rmtree(dense_index_dir)
                
                dense_config = DenseRetrieverConfig(
                    index_dir=dense_index_dir,
                    model_name="BAAI/bge-m3"
                )
                dense_retriever = DenseRetriever(dense_config.__dict__)
                dense_retriever.build(self.page_store)
                self.retrievers["vector"] = dense_retriever
                print("  ✓ DenseRetriever built successfully")
            except Exception as e:
                print(f"  ⚠ Failed to build DenseRetriever: {e}")
        else:
            print("  ⚠ DenseRetriever not available (FlagEmbedding dependencies missing)")
        
        # 更新ResearchAgent的检索器
        self.research_agent.retrievers = self.retrievers
        self._retrievers_built = True
        
        print(f"  ✓ Built {len(self.retrievers)} retriever(s): {list(self.retrievers.keys())}")
    
    def observe(self, turn: Turn) -> ObserveResult:
        """
        写入阶段：使用MemoryAgent存储对话内容
        
        GAM的memorize方法可以接收单个文本块，我们将每个turn作为独立的memorize调用
        """
        start_time = time.time()
        
        # 累积turn
        self.accumulated_turns.append(turn.text)
        
        # 执行memorize（将当前turn作为独立的记忆块）
        # 注意：GAM的memorize会处理文本并提取记忆
        try:
            memory_update = self.memory_agent.memorize(turn.text)
            # 记录API调用（memorize会调用LLM）
            self.total_api_calls += 1
        except Exception as e:
            print(f"Warning: GAM memorize failed for turn: {e}")
            memory_update = None
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # 估算token数（简化实现）
        tokens_in = len(turn.text.split())
        tokens_out = 50  # memorize通常会产生一些输出
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        
        # 获取当前存储状态
        storage_stats = self._get_storage_stats()
        
        return ObserveResult(
            cost_metrics={
                "total_latency_ms": latency_ms,
                "total_tokens_in": tokens_in,
                "total_tokens_out": tokens_out,
                "api_calls_count": 1
            },
            storage_stats=storage_stats
        )
    
    def _get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        try:
            if self.memory_store:
                final_state = self.memory_store.load()
                return {
                    "total_items_count": len(final_state.abstracts) if final_state else 0,
                    "current_context_window": sum(len(ab) for ab in (final_state.abstracts if final_state else []))
                }
        except Exception:
            pass
        return {
            "total_items_count": 0,
            "current_context_window": 0
        }
    
    def answer(self, query: str, meta: Optional[Dict[str, Any]] = None) -> AnswerResult:
        """
        QA阶段：使用ResearchAgent研究问题并生成答案（官方GAM的方式）
        
        关键改进：
        1. 使用research_agent.research()获取研究摘要
        2. 使用working_generator + 专用prompt生成短答案（不是直接截断summary）
        3. 根据category选择不同的prompt（category 3有特殊prompt）
        """
        start_time = time.time()
        retrieval_start = time.time()
        
        # 构建检索器（如果还没构建）
        self._build_retrievers()
        
        # 从meta中提取category（用于选择prompt）
        category = None
        if meta is not None:
            category = meta.get("category")
        
        # 使用ResearchAgent进行研究
        try:
            research_result = self.research_agent.research(query)
            research_summary = research_result.integrated_memory or ""
            
            # 获取研究过程的迭代信息
            raw_memory = research_result.raw_memory if hasattr(research_result, 'raw_memory') else {}
            iterations = raw_memory.get('iterations', [])
            loop_count = len(iterations)
            
            # 提取thought trace
            thought_trace = []
            for i, iteration in enumerate(iterations):
                if isinstance(iteration, dict):
                    plan = iteration.get('plan', '')
                    reflection = iteration.get('reflection', '')
                    if plan:
                        thought_trace.append(f"Iteration {i+1} Plan: {plan}")
                    if reflection:
                        thought_trace.append(f"Iteration {i+1} Reflection: {reflection}")
            
            retrieval_latency_ms = int((time.time() - retrieval_start) * 1000)
            generation_start = time.time()
            
            # 关键改进：使用working_generator + 专用prompt生成短答案（官方GAM的方式）
            if not research_summary:
                answer = "I need more information to answer this question."
            else:
                # 使用官方prompt函数生成短答案
                answer = answer_with_summary(
                    category=category,
                    summary=research_summary,
                    question=query,
                    generator=self.working_generator
                )
                
                # 检查答案是否为空，并记录警告
                if not answer:
                    print(f"⚠ Warning: Empty answer generated for query: {query[:50]}...")
                    print(f"  Research summary length: {len(research_summary)}")
                    print(f"  Category: {category}")
                    # 如果答案为空，使用research_summary的前100个字符作为fallback
                    answer = research_summary[:100].strip() if research_summary else "I need more information to answer this question."
            
            generation_latency_ms = int((time.time() - generation_start) * 1000)
            total_latency_ms = int((time.time() - start_time) * 1000)
            
            # 估算token数（包括prompt和生成的答案）
            prompt_tokens = len(query.split()) + len(research_summary.split()) if research_summary else len(query.split())
            tokens_in = prompt_tokens
            tokens_out = len(answer.split())
            
            return AnswerResult(
                answer=answer,
                cost_metrics={
                    "online_tokens_in": tokens_in,
                    "online_tokens_out": tokens_out,
                    "online_retrieval_latency_ms": retrieval_latency_ms,
                    "online_generation_latency_ms": generation_latency_ms,
                    "online_total_latency_ms": total_latency_ms,
                    "online_api_calls": loop_count + 1  # research iterations + final generation
                },
                mechanism_trace={
                    "retrieved_contexts": [research_summary[:200]] if research_summary else [],
                    "gam_thought_trace": thought_trace,
                    "gam_loop_count": loop_count,
                    "research_summary": research_summary[:500] if research_summary else "",
                    "category": category
                }
            )
        except Exception as e:
            print(f"Error in GAM answer: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回错误结果
            return AnswerResult(
                answer=f"Error: {str(e)}",
                cost_metrics={
                    "online_tokens_in": 0,
                    "online_tokens_out": 0,
                    "online_retrieval_latency_ms": 0,
                    "online_generation_latency_ms": 0,
                    "online_total_latency_ms": int((time.time() - start_time) * 1000),
                    "online_api_calls": 0
                },
                mechanism_trace={
                    "error": str(e)
                }
            )
    
    def get_system_name(self) -> str:
        return "gam"
    
    def __del__(self):
        """清理临时目录"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass
