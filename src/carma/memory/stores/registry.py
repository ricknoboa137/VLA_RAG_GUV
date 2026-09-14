"""Store registry."""

from __future__ import annotations

from carma.registry import Registry
from carma.types import Store

STORES: Registry[Store] = Registry("store")
