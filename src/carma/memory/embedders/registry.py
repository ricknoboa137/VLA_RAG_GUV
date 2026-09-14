"""Embedder registry."""

from __future__ import annotations

from carma.registry import Registry
from carma.types import Embedder

EMBEDDERS: Registry[Embedder] = Registry("embedder")
