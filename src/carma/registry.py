"""Generic string-keyed registry.

Every package that offers swappable implementations keeps one of these. Config
files name implementations by their registry key, and ``carma.config.build``
is the only place that turns a key into an object. See ``AGENTS.md`` section 4.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from carma.types import ConfigError

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps short string keys to factory callables."""

    def __init__(self, what: str) -> None:
        self._what = what
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, key: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator registering a factory under ``key``."""

        def wrap(factory: Callable[..., T]) -> Callable[..., T]:
            if key in self._items:
                msg = f"{self._what} {key!r} is already registered"
                raise ConfigError(msg)
            self._items[key] = factory
            return factory

        return wrap

    def create(self, key: str, **params: object) -> T:
        """Instantiate the implementation registered under ``key``."""
        if key not in self._items:
            known = ", ".join(sorted(self._items)) or "none"
            msg = f"unknown {self._what} {key!r}; registered: {known}"
            raise ConfigError(msg)
        return self._items[key](**params)

    def keys(self) -> tuple[str, ...]:
        """Every registered key, sorted."""
        return tuple(sorted(self._items))
