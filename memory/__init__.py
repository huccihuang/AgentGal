"""记忆系统模块。"""

from .file_ops import get_allowed_fields, _append_section_file, _update_section_file

__all__ = [
    "memory_consolidator",
    "vector_store",
    "get_allowed_fields",
    "_update_section_file",
    "_append_section_file",
]


def __getattr__(name: str):
    """延迟暴露实例，避免覆盖同名子模块。"""
    if name == "memory_consolidator":
        from .consolidator import memory_consolidator

        return memory_consolidator
    if name == "vector_store":
        from .vector_store import vector_store

        return vector_store
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
