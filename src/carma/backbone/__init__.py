"""Vision-language-action navigation policies.

This package must never import ``carma.memory``. A backbone accepts retrieved
context as an argument; it does not know where that context came from, and a
backbone that ignores it must still accept it so that conditions differ only by
config. See ``AGENTS.md`` section 3.
"""

from __future__ import annotations

from carma.backbone.alley_expert import AlleyExpert
from carma.backbone.chunked import ChunkedBackbone, ChunkSampler
from carma.backbone.heuristic import HeuristicBackbone
from carma.backbone.registry import BACKBONES
from carma.backbone.remote import RemoteBackbone
from carma.backbone.xvla import XvlaBackbone

__all__ = [
    "BACKBONES",
    "AlleyExpert",
    "ChunkSampler",
    "ChunkedBackbone",
    "HeuristicBackbone",
    "RemoteBackbone",
    "XvlaBackbone",
]
