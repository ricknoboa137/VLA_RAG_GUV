"""Adapter for open sentence-transformer checkpoints.

Requires the optional ``retrieval`` extra and a model already present in
``assets/``. It never downloads: fetching is an explicit, auditable step run by
``scripts/fetch_models.py``, so that no test or experiment silently reaches the
network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from carma.memory.embedders.registry import EMBEDDERS
from carma.types import ConfigError
from carma.types.core import Embedding


@EMBEDDERS.register("sentence_transformer")
class SentenceTransformerEmbedder:
    """Wraps a locally present sentence-transformers model.

    Args:
        model_dir: Directory holding the model, normally under ``assets/``.
    """

    def __init__(self, model_dir: str) -> None:
        path = Path(model_dir)
        if not path.exists():
            msg = (
                f"model directory {path} not found; "
                "run scripts/fetch_models.py before using this embedder"
            )
            raise ConfigError(msg)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional extra
            msg = "install the 'retrieval' extra to use SentenceTransformerEmbedder"
            raise ConfigError(msg) from exc
        self._model: Any = SentenceTransformer(str(path), device="cpu")
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return self._dim

    def embed(self, texts: list[str]) -> Embedding:
        """Embed a batch, returning shape ``(len(texts), dim)``, float32."""
        vecs = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)
