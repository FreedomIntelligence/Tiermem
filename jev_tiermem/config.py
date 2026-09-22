from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Config:
    model: str = "gpt-4.1-mini"
    jev_model: str = "jev-latest"
    jev_api_url: str | None = None
    sufficient_threshold: float = 0.8
    writeback_threshold: float = 0.9
    raw_chunk_chars: int = 3000
    summary_batch_chars: int = 12000
    note_max_chars: int = 1600
    memory_max_chars: int = 24000
    summary_top_k: int = 5
    raw_top_k: int = 4
    max_raw_pages: int = 12
    max_search_rounds: int = 3
    evidence_max_chars: int = 24000
    timeout: float = 30.0
    writeback: bool = True
    max_output_tokens: int = 2048
    reasoning_effort: str | None = None

    def __post_init__(self):
        if self.jev_api_url is not None:
            url = urlsplit(self.jev_api_url)
            if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.fragment:
                raise ValueError("jev_api_url must be a complete HTTP(S) endpoint without embedded credentials")
        for name in ("sufficient_threshold", "writeback_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in (
            "raw_chunk_chars", "summary_batch_chars", "note_max_chars",
            "memory_max_chars", "summary_top_k", "raw_top_k", "max_raw_pages",
            "max_search_rounds", "evidence_max_chars", "max_output_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.timeout < float("inf"):
            raise ValueError("timeout must be positive and finite")
        if self.raw_chunk_chars > min(self.summary_batch_chars, self.evidence_max_chars):
            raise ValueError("raw_chunk_chars must fit the summary and evidence budgets")
        if self.note_max_chars > self.memory_max_chars:
            raise ValueError("note_max_chars must fit memory_max_chars")
        if self.reasoning_effort not in {None, "minimal", "low", "medium", "high", "xhigh"}:
            raise ValueError("Unsupported reasoning_effort")
