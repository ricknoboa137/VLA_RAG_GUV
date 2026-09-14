"""Operator-derived memory: embedding, storage, retrieval and hygiene.

This package must never import ``carma.backbone``. The two are joined only by
``carma.arbiter``; keeping them apart is what makes the no-memory baseline a
config change rather than a code path. See ``AGENTS.md`` section 3.
"""

from __future__ import annotations

from carma.memory.embedders import EMBEDDERS
from carma.memory.hygiene import HYGIENE
from carma.memory.retriever import RETRIEVERS
from carma.memory.stores import STORES

__all__ = ["EMBEDDERS", "HYGIENE", "RETRIEVERS", "STORES"]
