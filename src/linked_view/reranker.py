"""
Reranker module using Qwen3-Reranker for document re-ranking.

Usage:
    reranker = Qwen3Reranker(model_path="/path/to/Qwen3-Reranker-0.6B")
    reranked_hits = reranker.rerank(query, hits, top_k=5)
"""

import logging
from typing import List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Lazy imports to avoid loading heavy dependencies at module import time
_torch = None
_AutoTokenizer = None
_AutoModelForCausalLM = None


def _ensure_imports():
    """Lazily import torch and transformers."""
    global _torch, _AutoTokenizer, _AutoModelForCausalLM
    if _torch is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        _torch = torch
        _AutoTokenizer = AutoTokenizer
        _AutoModelForCausalLM = AutoModelForCausalLM

        # 打印版本信息用于诊断
        try:
            import transformers
            import accelerate
            logger.info(f"[Qwen3Reranker] torch version: {torch.__version__}")
            logger.info(f"[Qwen3Reranker] transformers version: {transformers.__version__}")
            logger.info(f"[Qwen3Reranker] accelerate version: {accelerate.__version__}")
            logger.info(f"[Qwen3Reranker] CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"[Qwen3Reranker] CUDA device: {torch.cuda.get_device_name(0)}")
                logger.info(f"[Qwen3Reranker] CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        except Exception as e:
            logger.warning(f"[Qwen3Reranker] Failed to get version info: {e}")


class Qwen3Reranker:
    """
    Reranker using Qwen3-Reranker model.

    Based on: relatedwork/Qwen3-Embedding/examples/qwen3_reranker_transformers.py
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-Reranker-0.6B",
        max_length: int = 2048,
        instruction: Optional[str] = None,
        device: str = "cuda",
    ) -> None:
        """
        Initialize the Qwen3 Reranker.

        Args:
            model_name_or_path: Path to the Qwen3-Reranker model
            max_length: Maximum sequence length
            instruction: Custom instruction for reranking
            device: Device to run the model on ("cuda" or "cpu")
        """
        _ensure_imports()

        self.max_length = max_length
        self.device = device

        logger.info(f"[Qwen3Reranker] Loading model from {model_name_or_path}")

        self.tokenizer = _AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side='left'
        )

        # Check if flash_attn is available
        attn_impl = "eager"
        if device == "cuda":
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
                logger.info("[Qwen3Reranker] Using flash_attention_2")
            except ImportError:
                attn_impl = "eager"
                logger.info("[Qwen3Reranker] flash_attn not installed, using eager attention")

        # Load model with appropriate dtype
        # 参考原始 Qwen3-Embedding 仓库的加载方式
        # 注意：模型 config.json 中 torch_dtype 是 bfloat16
        # transformers 4.57+ 默认使用 meta tensor，需要显式禁用

        # 清除 CUDA 缓存
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

        # 禁用 accelerate 的自动优化
        import os
        os.environ["ACCELERATE_DISABLE_META_TENSORS"] = "1"

        load_success = False
        last_error = None

        # 检测 GPU 是否支持 bfloat16
        use_bf16 = _torch.cuda.is_available() and _torch.cuda.is_bf16_supported()
        dtype = _torch.bfloat16 if use_bf16 else _torch.float16
        logger.info(f"[Qwen3Reranker] Using dtype: {dtype} (bf16 supported: {use_bf16})")

        # 策略 1: 使用 device_map="auto" 让 accelerate 完整处理（避免手动 .to()）
        if not load_success:
            try:
                logger.info("[Qwen3Reranker] Trying loading strategy 1: device_map='auto'")
                self.lm = _AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    attn_implementation=attn_impl,
                    device_map="auto",  # 让 accelerate 自动处理设备分配
                )
                self.lm.eval()
                load_success = True
                logger.info("[Qwen3Reranker] Strategy 1 succeeded (device_map=auto)")
            except Exception as e:
                last_error = e
                logger.warning(f"[Qwen3Reranker] Strategy 1 failed: {e}")

        # 策略 2: 使用 device_map 指定具体设备
        if not load_success:
            try:
                logger.info("[Qwen3Reranker] Trying loading strategy 2: device_map={'': device}")
                target_device = device if device == "cuda" and _torch.cuda.is_available() else "cpu"
                self.lm = _AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    attn_implementation="eager",
                    device_map={"": target_device},
                )
                self.lm.eval()
                load_success = True
                logger.info(f"[Qwen3Reranker] Strategy 2 succeeded (device_map={{'': {target_device}}})")
            except Exception as e:
                last_error = e
                logger.warning(f"[Qwen3Reranker] Strategy 2 failed: {e}")

        # 策略 3: 完全在 CPU 上加载（禁用所有优化），然后用 to_empty + load_state_dict
        if not load_success:
            try:
                logger.info("[Qwen3Reranker] Trying loading strategy 3: load on CPU with materialization")
                # 先创建模型配置
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

                # 直接在目标设备上初始化模型（而不是先 meta 再移动）
                with _torch.device(device if device == "cuda" and _torch.cuda.is_available() else "cpu"):
                    self.lm = _AutoModelForCausalLM.from_pretrained(
                        model_name_or_path,
                        config=config,
                        trust_remote_code=True,
                        torch_dtype=dtype,
                        attn_implementation="eager",
                        low_cpu_mem_usage=False,
                        device_map=None,
                    )
                self.lm.eval()
                load_success = True
                logger.info("[Qwen3Reranker] Strategy 3 succeeded")
            except Exception as e:
                last_error = e
                logger.warning(f"[Qwen3Reranker] Strategy 3 failed: {e}")

        # 策略 4: 使用 safetensors 直接加载权重
        if not load_success:
            try:
                logger.info("[Qwen3Reranker] Trying loading strategy 4: manual safetensors loading")
                from transformers import AutoConfig
                from safetensors.torch import load_file
                import os as os_module

                config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

                # 在目标设备上创建空模型
                target_device = device if device == "cuda" and _torch.cuda.is_available() else "cpu"

                # 加载 safetensors 文件
                safetensors_path = os_module.path.join(model_name_or_path, "model.safetensors")
                state_dict = load_file(safetensors_path, device=target_device)

                # 创建模型并加载权重
                self.lm = _AutoModelForCausalLM.from_config(
                    config,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    attn_implementation="eager",
                )
                self.lm.load_state_dict(state_dict)
                self.lm = self.lm.to(target_device)
                self.lm.eval()
                load_success = True
                logger.info("[Qwen3Reranker] Strategy 4 succeeded (manual safetensors)")
            except Exception as e:
                last_error = e
                logger.warning(f"[Qwen3Reranker] Strategy 4 failed: {e}")

        # 策略 5: 原始简单方式（最后尝试）
        if not load_success:
            try:
                logger.info("[Qwen3Reranker] Trying loading strategy 5: original simple style")
                self.lm = _AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    attn_implementation=attn_impl,
                ).cuda().eval()
                load_success = True
                logger.info("[Qwen3Reranker] Strategy 5 succeeded (original style)")
            except Exception as e:
                last_error = e
                logger.warning(f"[Qwen3Reranker] Strategy 5 failed: {e}")

        if not load_success:
            logger.error(f"[Qwen3Reranker] All loading strategies failed. Last error: {last_error}")
            raise RuntimeError(f"Failed to load reranker model after trying all strategies: {last_error}")

        # Token IDs for yes/no
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        # Prompt template
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

        # Default instruction for memory retrieval
        self.instruction = instruction or "Given the user query, retrieve the relevant memory passages that can help answer the question"

        logger.info(f"[Qwen3Reranker] Model loaded successfully on {device}")

    def _format_instruction(self, instruction: str, query: str, doc: str) -> str:
        """Format the input for the reranker."""
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _process_inputs(self, pairs: List[str]):
        """Tokenize and prepare inputs for the model."""
        out = self.tokenizer(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )

        for i, ele in enumerate(out['input_ids']):
            out['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens

        out = self.tokenizer.pad(
            out,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length
        )

        for key in out:
            out[key] = out[key].to(self.lm.device)

        return out

    def _compute_logits(self, inputs) -> List[float]:
        """Compute relevance scores using the model."""
        _ensure_imports()

        # 使用 with 语句代替装饰器，避免类定义时 _torch 为 None 的问题
        with _torch.no_grad():
            batch_scores = self.lm(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = _torch.stack([false_vector, true_vector], dim=1)
            batch_scores = _torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()

        return scores

    def compute_scores(
        self,
        pairs: List[Tuple[str, str]],
        instruction: Optional[str] = None,
        batch_size: int = 8,
    ) -> List[float]:
        """
        Compute relevance scores for query-document pairs.

        Args:
            pairs: List of (query, document) tuples
            instruction: Optional custom instruction
            batch_size: Batch size for inference

        Returns:
            List of relevance scores (0-1)
        """
        if not pairs:
            return []

        instruction = instruction or self.instruction
        formatted_pairs = [
            self._format_instruction(instruction, query, doc)
            for query, doc in pairs
        ]

        # Process in batches
        all_scores = []
        for i in range(0, len(formatted_pairs), batch_size):
            batch = formatted_pairs[i:i + batch_size]
            inputs = self._process_inputs(batch)
            scores = self._compute_logits(inputs)
            all_scores.extend(scores)

        return all_scores

    def rerank(
        self,
        query: str,
        hits: List[Any],
        top_k: int = 5,
        text_field: str = "summary_text",
        instruction: Optional[str] = None,
        force_score: bool = False,
    ) -> List[Any]:
        """
        Rerank hits by relevance to the query.

        Args:
            query: The user query
            hits: List of hit objects (must have text_field attribute or be dict)
            top_k: Number of top results to return
            text_field: Field name to extract text from hits
            instruction: Optional custom instruction
            force_score: If True, always compute scores even if len(hits) <= top_k

        Returns:
            Reranked list of top_k hits (with updated scores)
        """
        if not hits:
            return []

        # 如果不需要强制计算分数，且 hits 数量 <= top_k，直接返回
        if not force_score and len(hits) <= top_k:
            logger.info(f"[Qwen3Reranker] Only {len(hits)} hits, returning all without reranking")
            return hits

        # Extract text from hits
        pairs = []
        valid_indices = []

        for i, hit in enumerate(hits):
            # Try to get text from different sources
            text = None
            if hasattr(hit, text_field):
                text = getattr(hit, text_field)
            elif isinstance(hit, dict) and text_field in hit:
                text = hit[text_field]
            elif hasattr(hit, "content"):
                text = hit.content
            elif isinstance(hit, dict) and "content" in hit:
                text = hit["content"]

            if text:
                pairs.append((query, str(text)[:1500]))  # Truncate long texts
                valid_indices.append(i)

        if not pairs:
            logger.warning("[Qwen3Reranker] No valid text found in hits, returning original")
            return hits[:top_k]

        # Compute scores
        logger.info(f"[Qwen3Reranker] Reranking {len(pairs)} hits")
        scores = self.compute_scores(pairs, instruction)

        # Sort by score and get top_k
        scored_hits = list(zip(valid_indices, scores))
        scored_hits.sort(key=lambda x: x[1], reverse=True)

        # Return top_k hits with updated scores
        reranked = []
        for idx, score in scored_hits[:top_k]:
            hit = hits[idx]
            # Update score if possible
            if hasattr(hit, "score"):
                hit.score = score
            elif isinstance(hit, dict):
                hit["rerank_score"] = score
            reranked.append(hit)

        logger.info(f"[Qwen3Reranker] Reranked to top {len(reranked)} hits")
        return reranked


# Singleton instance for reuse
_reranker_instance: Optional[Qwen3Reranker] = None


def get_reranker(
    model_path: str = "Qwen/Qwen3-Reranker-0.6B",
    **kwargs
) -> Qwen3Reranker:
    """
    Get or create a singleton Qwen3Reranker instance.

    Args:
        model_path: Path to the model
        **kwargs: Additional arguments for Qwen3Reranker

    Returns:
        Qwen3Reranker instance
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Qwen3Reranker(model_name_or_path=model_path, **kwargs)
    return _reranker_instance
