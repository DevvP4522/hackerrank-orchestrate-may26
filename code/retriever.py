# pip install faiss-cpu sentence-transformers numpy
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_chunks(filepath: str) -> List[Dict[str, str]]:
    """
    Load JSON chunks and validate entries.
    Returns list of dicts: {text, source, product_area}
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to load JSON from {path}: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data).__name__}")

    chunks: List[Dict[str, str]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        text = item.get("text", "")
        source = item.get("source", "")
        product_area = item.get("product_area", "")

        if not isinstance(text, str) or not isinstance(source, str) or not isinstance(product_area, str):
            continue

        cleaned = text.strip()
        if not cleaned:
            continue

        chunks.append({"text": cleaned, "source": source.strip(), "product_area": product_area.strip()})

    return chunks


def _l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


def build_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generate L2-normalized embeddings for texts.
    Returns array of shape (N, D), dtype float32.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # sentence-transformers returns np.ndarray (float32/float64 depending on backend)
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # L2 normalization
    embeddings = _l2_normalize_rows(embeddings).astype(np.float32, copy=False)
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build FAISS IndexFlatL2 and add embeddings.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    if embeddings.shape[0] == 0:
        raise ValueError("Cannot build FAISS index with 0 embeddings")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    # FAISS expects float32 contiguous arrays
    vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
    index.add(vectors)
    return index


def save_artifacts(index: faiss.Index, texts: List[str], metadata: List[Dict[str, str]]) -> None:
    project_root = Path(__file__).resolve().parent.parent
    artifacts_dir = project_root
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    index_path = artifacts_dir / "faiss.index"
    texts_path = artifacts_dir / "texts.pkl"
    metadata_path = artifacts_dir / "metadata.pkl"

    try:
        faiss.write_index(index, str(index_path))
        with texts_path.open("wb") as f:
            pickle.dump(texts, f, protocol=pickle.HIGHEST_PROTOCOL)
        with metadata_path.open("wb") as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise RuntimeError(f"Failed to save artifacts: {exc}") from exc


def load_artifacts() -> Tuple[faiss.Index, List[str], List[Dict[str, str]]]:
    project_root = Path(__file__).resolve().parent.parent
    artifacts_dir = project_root

    index_path = artifacts_dir / "faiss.index"
    texts_path = artifacts_dir / "texts.pkl"
    metadata_path = artifacts_dir / "metadata.pkl"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing {index_path}")
    if not texts_path.exists():
        raise FileNotFoundError(f"Missing {texts_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}")

    try:
        index = faiss.read_index(str(index_path))
        with texts_path.open("rb") as f:
            texts = pickle.load(f)
        with metadata_path.open("rb") as f:
            metadata = pickle.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to load artifacts: {exc}") from exc

    if not isinstance(texts, list) or not isinstance(metadata, list):
        raise ValueError("Loaded texts/metadata are not lists")

    if len(texts) != len(metadata):
        raise ValueError(f"texts and metadata length mismatch: {len(texts)} != {len(metadata)}")

    return index, texts, metadata


def setup_index(force_rebuild: bool = False) -> Tuple[faiss.Index, List[str], List[Dict[str, str]]]:
    project_root = Path(__file__).resolve().parent.parent
    chunks_path = project_root / "processed_chunks.json"
    artifacts_dir = project_root

    index_exists = (artifacts_dir / "faiss.index").exists()

    if index_exists and not force_rebuild:
        print("Loading existing index...")
        try:
            index, texts, metadata = load_artifacts()
            print("Loaded successfully")
            print("Index ready")
            return index, texts, metadata
        except Exception as exc:
            print(f"[warn] Failed to load existing index (rebuilding). Reason: {exc}")

    if index_exists and force_rebuild:
        print("Force rebuild enabled; rebuilding index...")

    print("Building new index...")
    chunks = load_chunks(str(chunks_path))
    if not chunks:
        raise ValueError(f"No valid chunks found in {chunks_path}")

    texts = [c["text"] for c in chunks]
    metadata = [{"source": c["source"], "product_area": c["product_area"]} for c in chunks]

    embeddings = build_embeddings(texts)
    if embeddings.shape[0] != len(texts):
        raise ValueError("Embedding count does not match chunk count")

    index = build_faiss_index(embeddings)
    save_artifacts(index=index, texts=texts, metadata=metadata)

    print("Index ready")
    return index, texts, metadata


class Retriever:
    def __init__(self, force_rebuild: bool = False):
        self.index, self.texts, self.metadata = setup_index(force_rebuild=force_rebuild)
        # Keep the embedding model only in memory for query-time efficiency.
        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most relevant chunks for a query.
        """
        if not isinstance(query, str):
            return []
        query = query.strip()
        if not query:
            return []
        if top_k <= 0:
            return []

        q_emb = self._model.encode([query], batch_size=1, show_progress_bar=False)
        q_emb = np.asarray(q_emb, dtype=np.float32)
        q_emb = _l2_normalize_rows(q_emb)

        distances, indices = self.index.search(np.asarray(q_emb, dtype=np.float32), top_k)
        distances = distances[0]
        indices = indices[0]

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances, indices):
            if idx < 0 or idx >= len(self.texts):
                continue

            similarity = float(1.0 - float(dist))
            meta = self.metadata[idx]
            results.append(
                {
                    "text": self.texts[idx],
                    "source": meta.get("source", ""),
                    "product_area": meta.get("product_area", ""),
                    "score": similarity,
                }
            )
        return results


if __name__ == "__main__":
    retriever = Retriever(force_rebuild=True)
    print("Index built and ready.")
