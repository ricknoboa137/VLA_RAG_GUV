"""A dependency-free hashing embedder.

Not a good text representation. It exists so that the whole pipeline, the test
suite and CI run with no model download at all, which keeps the default test
path offline as required by ``AGENTS.md`` section 2. Swap in
:class:`~carma.memory.embedders.sentence.SentenceTransformerEmbedder` for real
experiments.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from carma.memory.embedders.registry import EMBEDDERS
from carma.types.core import Embedding

_TOKEN = re.compile(r"[a-z0-9]+")


@EMBEDDERS.register("hashing")
class HashingEmbedder:
    """Hashes tokens into a fixed-width bag, then L2-normalises.

    Deterministic across processes and platforms: it uses ``blake2b`` rather
    than Python's randomised ``hash``.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            msg = "dim must be positive"
            raise ValueError(msg)
        self._dim = dim

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return self._dim

    def embed(self, texts: list[str]) -> Embedding:
        """Embed a batch, returning shape ``(len(texts), dim)``, float32."""
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN.findall(text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                idx = int.from_bytes(digest, "big") % self._dim
                out[row, idx] += 1.0
            norm = float(np.linalg.norm(out[row]))
            if norm > 0.0:
                out[row] /= norm
        return out
