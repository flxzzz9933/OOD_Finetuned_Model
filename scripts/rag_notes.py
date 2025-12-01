# scripts/rag_notes.py
"""
Simple RAG over CS3500 lecture-note chunks.

- Loads chunks from data/interim/context_chunks.jsonl
- Loads precomputed embeddings from data/interim/context_embeddings.npz
- Uses the SAME encoder as you used originally (all-MiniLM-L6-v2)
- Retrieves top-k chunks by cosine similarity.
- Formats a <Context> block like the one in fim_dataset.jsonl.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple, Optional

import json
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class NoteChunk:
    chunk_id: str
    source: str
    text: str
    topics: List[str]


class NoteRAG:
    def __init__(
        self,
        data_dir: str | Path = "data",
        k_default: int = 3,
        device: str = "cuda",
    ) -> None:
        base = Path(data_dir)
        chunks_path = base / "interim" / "context_chunks.jsonl"
        emb_path = base / "interim" / "context_embeddings.npz"
        meta_path = base / "interim" / "context_meta.json"

        # Load note chunks
        self.chunks: List[NoteChunk] = []
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.chunks.append(
                    NoteChunk(
                        chunk_id=obj["chunk_id"],
                        source=obj["source"],
                        text=obj["text"],
                        topics=obj.get("topics", []),
                    )
                )

        # Load embeddings (N x D)
        npz = np.load(emb_path)
        if "emb" in npz.files:
            emb = npz["emb"]
        elif "embeddings" in npz.files:
            emb = npz["embeddings"]
        else:
            # Fallback: take the first array in the file
            emb = npz[npz.files[0]]
        self.embeddings = self._normalize(emb.astype("float32"))

        # Load meta to get encoder name/dim if needed
        with meta_path.open("r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.model_name: str = self.meta.get(
            "model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.dim: int = int(self.meta.get("dim", self.embeddings.shape[1]))
        self.k_default = k_default

        # Query encoder – same as you used for indexing
        self.encoder = SentenceTransformer(self.model_name, device=device)

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        """Row-wise L2 normalization (for cosine similarity)."""
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a query string into a normalized embedding."""
        q = self.encoder.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        return q.astype("float32")

    def _topic_mask(self, topics: Optional[Iterable[str]]) -> np.ndarray:
        """Optional topic filter: keep chunks whose topics intersect."""
        if not topics:
            return np.ones(len(self.chunks), dtype=bool)
        topic_set = {t.lower() for t in topics}
        mask = []
        for c in self.chunks:
            mask.append(bool(topic_set.intersection({t.lower() for t in c.topics})))
        return np.array(mask, dtype=bool)

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        topics: Optional[Iterable[str]] = None,
    ) -> List[Tuple[NoteChunk, float]]:
        """
        Retrieve top-k chunks for a query.

        Args:
            query: text query; for FIM you can build this from code_prefix+code_suffix.
            k: number of chunks (defaults to self.k_default).
            topics: optional list of coarse topics to restrict search.

        Returns:
            List of (NoteChunk, cosine_similarity_score), highest first.
        """
        k = k or self.k_default
        q = self.encode_query(query)

        mask = self._topic_mask(topics)
        emb = self.embeddings[mask]
        if emb.size == 0:
            # No topic match – fall back to all chunks
            emb = self.embeddings
            idx_global = np.arange(len(self.chunks))
        else:
            idx_global = np.where(mask)[0]

        scores = emb @ q  # cosine similarity
        k = min(k, len(scores))
        # Argpartition for efficiency, then sort these k
        top_local = np.argpartition(-scores, k - 1)[:k]
        top_local = top_local[np.argsort(-scores[top_local])]

        results: List[Tuple[NoteChunk, float]] = []
        for i in top_local:
            j = int(idx_global[i])
            results.append((self.chunks[j], float(scores[i])))
        return results

    @staticmethod
    def format_context(hits: List[Tuple[NoteChunk, float]]) -> str:
        """
        Format retrieved chunks into the <Context> block used in fim_dataset.jsonl.
        """
        blocks: List[str] = []
        for chunk, score in hits:
            # score is included only for debugging; you can drop it.
            header = f"[{chunk.chunk_id}]"
            blocks.append(f"{header} {chunk.text.strip()}")
        return "<Context>\n" + "\n---\n".join(blocks) + "\n</Context>"

    # Convenience for FIM examples -----------------------------------------

    def build_query_from_fim_example(self, ex: Dict[str, Any]) -> str:
        """
        Build a retrieval query from a FIM JSONL example.

        Heuristic: last 200 chars of code_prefix + first 200 chars of code_suffix
        + topic tags. You can tweak this if you like.
        """
        prefix_tail = ex["code_prefix"][-200:]
        suffix_head = ex["code_suffix"][:200]
        topics = " ".join(ex.get("meta", {}).get("topics", []))
        return f"{topics}\n\n{prefix_tail}\n\n{suffix_head}"

    def context_for_fim_example(
        self, ex: Dict[str, Any], k: Optional[int] = None
    ) -> str:
        """
        Run retrieval for a FIM example and return a <Context> string.
        """
        query = self.build_query_from_fim_example(ex)
        topics = ex.get("meta", {}).get("topics", [])
        hits = self.retrieve(query, k=k, topics=topics)
        return self.format_context(hits)


if __name__ == "__main__":
    rag = NoteRAG(data_dir="../data", k_default=3, device="cpu")
    example_query = "adapter pattern: IntSet2ToIntSet1 adapter iterator implementation"
    hits = rag.retrieve(example_query, k=3, topics=["adapter", "decorator"])
    print(rag.format_context(hits))