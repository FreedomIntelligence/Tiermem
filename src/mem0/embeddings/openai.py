import os
import time
import warnings
from typing import Literal, Optional

from openai import OpenAI

from ..configs.embeddings.base import BaseEmbedderConfig
from ..embeddings.base import EmbeddingBase


class OpenAIEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "text-embedding-3-small"
        self.config.embedding_dims = self.config.embedding_dims or 1536

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        base_url = (
            self.config.openai_base_url
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        if os.environ.get("OPENAI_API_BASE"):
            warnings.warn(
                "The environment variable 'OPENAI_API_BASE' is deprecated and will be removed in the 0.1.80. "
                "Please use 'OPENAI_BASE_URL' instead.",
                DeprecationWarning,
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        # 最近一次 embedding 调用的真实 usage（OpenAI embeddings 返回 usage.total_tokens）
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.last_latency_ms: int = 0

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using OpenAI.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ")
        t0 = time.time()
        resp = self.client.embeddings.create(
            input=[text],
            model=self.config.model,
            dimensions=self.config.embedding_dims,
        )
        self.last_latency_ms = int((time.time() - t0) * 1000)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            # embeddings 一般只有 total_tokens（输入 token）
            tt = getattr(usage, "total_tokens", None) or getattr(usage, "prompt_tokens", None) or 0
            self.last_usage = {"prompt_tokens": int(tt), "completion_tokens": 0, "total_tokens": int(tt)}
        else:
            self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return resp.data[0].embedding
