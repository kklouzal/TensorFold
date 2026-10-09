"""Transient projection metadata for an internally owned forward operation.

Generic calls have no retained derived-array authority. An operation's caller
owns the produced inputs and borrows parameter transforms without mutation
until the synchronous operation exits. At most four sums/rotation entries are
retained together; generic calls retain none. Identity keys serve only this
private no-mutation borrow and never establish freshness across operations.
Native lazy graphs retain their arrays after this context retires. ContextVar
keeps synchronous forwards and independent threads separate. Callers must not
spawn Python tasks inside a scope; nested scopes deliberately share that operation.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator


_current: ContextVar[dict[tuple[Any, ...], tuple[Any, ...]] | None] = ContextVar(
    "tensorfold_projection_operation", default=None)


@contextmanager
def operation() -> Iterator[None]:
    existing = _current.get()
    if existing is not None:
        yield
        return
    token = _current.set({})
    try:
        yield
    finally:
        _current.reset(token)


def _remember(key: tuple[Any, ...], value: tuple[Any, ...]) -> None:
    cache = _current.get()
    if cache is None:
        return
    cache[key] = value
    while len(cache) > 4:
        cache.pop(next(iter(cache)))


def remember(x: Any, sums: Any, group: int = 64) -> None:
    _remember(("sums", id(x), group), (x, sums))


def sums_of(x: Any, group: int) -> Any | None:
    cache = _current.get()
    if cache is None:
        return None
    found = cache.get(("sums", id(x), group))
    return found[1] if found is not None and found[0] is x else None


def remember_rotation(owner: Any, x: Any, signs: Any, result: Any) -> None:
    """Publish one transform under the enclosing operation's no-mutation borrow."""
    _remember(("rotation", id(owner), id(x), id(signs)), (owner, x, signs, result))


def rotation_of(owner: Any, x: Any, signs: Any) -> Any | None:
    """Read a borrowed operation-local transform; generic calls always miss."""
    cache = _current.get()
    if cache is None:
        return None
    found = cache.get(("rotation", id(owner), id(x), id(signs)))
    return (found[3] if found is not None and found[0] is owner and found[1] is x and found[2] is signs else None)
