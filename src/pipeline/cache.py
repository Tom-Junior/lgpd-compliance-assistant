"""Cache em 2 niveis: exact-match (SHA256) + semantic (cosine similarity).

Versão 100% gratuita: usa sentence-transformers local (sem OpenAI/Gemini).
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


class ExactCache:
    """Cache por hash SHA256 da query. Captura replays exatos (~10-15% das queries)."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()

    def get(self, query: str) -> str | None:
        return self._store.get(self._key(query))

    def put(self, query: str, answer: str) -> None:
        self._store[self._key(query)] = answer

    def stats(self) -> dict[str, int]:
        return {"size": len(self._store)}


class SemanticCache:
    """Cache por similaridade de embedding. Captura parafrases (~20% adicional).
    
    Usa sentence-transformers local (100% gratuito, sem API externa).
    """

    def __init__(self, threshold: float = 0.93, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        self.threshold = threshold
        self.model_name = model_name
        self._queries: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._answers: list[str] = []
        self._model: SentenceTransformer | None = None  # Lazy load

    def _get_model(self) -> SentenceTransformer:
        """Lazy load do modelo (só carrega quando necessário)."""
        if self._model is None:
            print(f"📥 SemanticCache: carregando modelo {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            print("✅ SemanticCache: modelo carregado!")
        return self._model

    def _embed(self, text: str) -> np.ndarray:
        """Gera embedding local usando sentence-transformers."""
        model = self._get_model()
        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(embedding)

    def get(self, query: str) -> str | None:
        """Retorna resposta cacheada se similar a query alguma anterior, OU None."""
        if not self._queries:
            return None

        # 1. Embedar a query localmente
        query_embedding = self._embed(query)
        
        # 2. Calcular similaridade cosseno contra todos os embeddings cacheados
        similarities = []
        for cached_embedding in self._embeddings:
            cos_sim = np.dot(query_embedding, cached_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
            )
            similarities.append(cos_sim)
        
        # 3. Pegar índice do maior valor de similaridade
        similarities_array = np.array(similarities)
        best_idx = int(np.argmax(similarities_array))
        best_similarity = float(similarities_array[best_idx])
        
        # 4. Se similaridade >= threshold, retornar resposta cacheada
        if best_similarity >= self.threshold:
            return self._answers[best_idx]
        
        # 5. Caso contrário, retornar None
        return None

    def put(self, query: str, answer: str) -> None:
        self._queries.append(query)
        self._embeddings.append(self._embed(query))
        self._answers.append(answer)

    def stats(self) -> dict[str, Any]:
        return {"size": len(self._queries), "threshold": self.threshold}