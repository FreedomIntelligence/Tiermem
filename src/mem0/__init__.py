try:
    import importlib.metadata as _importlib_metadata  # py>=3.8
except Exception:  # pragma: no cover
    _importlib_metadata = None

if _importlib_metadata is None:
    __version__ = "local"
else:
    try:
        __version__ = _importlib_metadata.version("mem0ai")
    except Exception:
        # 本仓库内置的 mem0（非 pip 安装）可能没有 dist metadata
        __version__ = "local"

try:
    from .client.main import AsyncMemoryClient, MemoryClient  # type: ignore # noqa
except Exception:  # pragma: no cover
    # 允许在缺少 httpx 等依赖时仍能使用本地 Memory（用于离线实验/最小化环境）
    AsyncMemoryClient = None  # type: ignore
    MemoryClient = None  # type: ignore

from .memory.main import AsyncMemory, Memory  # noqa
