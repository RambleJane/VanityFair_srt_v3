"""Context-preserving batching for LLM stages."""
from __future__ import annotations

from typing import Iterator, Sequence, TypeVar

T = TypeVar("T")


def iter_batches(items: Sequence[T], batch_size: int) -> Iterator[list[T]]:
    """Yield consecutive batches of at most ``batch_size`` items."""
    size = max(1, int(batch_size))
    for start in range(0, len(items), size):
        yield list(items[start : start + size])
