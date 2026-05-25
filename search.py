"""
search.py
---------
Semantic search engine for the Document AI Pipeline.

HOW IT WORKS:
  1. Each document's text is encoded into a dense vector using
     SentenceTransformers (all-MiniLM-L6-v2 — small, fast, offline-capable).
  2. Vectors are stored in a FAISS index for fast cosine-similarity search.
  3. At query time the query string is encoded with the same model and the
     top-k nearest document vectors are returned.

WHY THESE CHOICES:
  - sentence-transformers/all-MiniLM-L6-v2  : ~80 MB, runs fully on CPU,
    no internet needed after the first download, strong semantic quality.
  - faiss-cpu                                : Meta's library for fast
    nearest-neighbour search; no GPU required.
  - IndexFlatIP (inner product on L2-normed  : equivalent to cosine similarity;
    vectors)                                   simple and exact (no approximation).

OFFLINE USE:
  On first run the model is downloaded to ~/.cache/huggingface/hub.
  After that it works fully offline. To pre-bundle the model, run:
      python -c "from sentence_transformers import SentenceTransformer; \
                 SentenceTransformer('all-MiniLM-L6-v2')"
  before zipping the project.

DEPENDENCIES (add to requirements.txt):
    sentence-transformers>=2.2.2
    faiss-cpu>=1.7.4
"""

from __future__ import annotations

import re
import textwrap
from typing import Optional

import numpy as np

import os
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"


from dotenv import load_dotenv
load_dotenv()  

# ── Lazy imports so the rest of the pipeline still works if these are
#    missing (helpful during setup / testing without GPU-heavy installs).
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

MODEL_NAME    = "all-MiniLM-L6-v2"   # 384-dim, ~80 MB, CPU-friendly
PREVIEW_CHARS = 200                   # Characters shown in search results
CHUNK_SIZE    = 512                   # Words per chunk for long documents


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Split a long document into overlapping word-level chunks so that
    very long PDFs are represented by multiple vectors rather than one
    truncated embedding.

    Overlap of 10 % prevents context loss at chunk boundaries.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    overlap  = max(1, chunk_size // 10)
    step     = chunk_size - overlap
    chunks   = []
    start    = 0

    while start < len(words):
        chunk = words[start : start + chunk_size]
        chunks.append(" ".join(chunk))
        start += step

    return chunks


def _build_preview(text: str, query: str) -> str:
    """
    Return a short snippet of the document that is near the query terms.
    Falls back to the document start if no term is found.
    """
    query_words = [w.lower() for w in query.split() if len(w) > 3]
    text_lower  = text.lower()

    best_pos = -1
    for word in query_words:
        pos = text_lower.find(word)
        if pos != -1:
            best_pos = pos
            break

    if best_pos == -1:
        snippet = text[:PREVIEW_CHARS]
    else:
        start   = max(0, best_pos - 40)
        snippet = text[start : start + PREVIEW_CHARS]

    # Collapse whitespace and trim
    snippet = re.sub(r"\s+", " ", snippet).strip()
    return textwrap.shorten(snippet, width=PREVIEW_CHARS, placeholder="…")


# ─────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────

class SemanticSearchEngine:
    """
    Build a FAISS index from document texts and answer free-text queries.

    Usage
    -----
    engine = SemanticSearchEngine()
    engine.build_index(documents)          # documents from main.py load_documents()
    results = engine.search("payment due in January", top_k=3)
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self._check_dependencies()
        print(f"\nLoading embedding model: '{model_name}' …")
        self.model      = SentenceTransformer(model_name)
        self.index      = None          # FAISS index (built lazily)
        self.metadata   : list[dict] = []   # parallel list to FAISS vectors
        print("  Model loaded.\n")

    # ── dependency guard ────────────────────────────────────────────────────

    @staticmethod
    def _check_dependencies():
        missing = []
        if not _ST_AVAILABLE:
            missing.append("sentence-transformers")
        if not _FAISS_AVAILABLE:
            missing.append("faiss-cpu")
        if missing:
            raise ImportError(
                f"Missing required packages: {', '.join(missing)}\n"
                f"Install with:  pip install {' '.join(missing)}"
            )

    # ── index construction ──────────────────────────────────────────────────

    def build_index(
        self,
        documents   : list[dict],
        class_map   : Optional[dict] = None,
    ) -> None:
        """
        Encode all documents and build a FAISS cosine-similarity index.

        Parameters
        ----------
        documents : list of dicts with at least 'filename' and 'text' keys
                    (exactly what main.py's load_documents() returns).
        class_map : optional {filename: class} mapping so search results
                    can show the document class alongside the filename.
                    If None, class is shown as 'Unknown'.
        """
        if not documents:
            print("[WARNING] build_index called with empty document list.")
            return

        class_map = class_map or {}

        # Build flat list of (chunk_text, metadata) pairs
        all_texts : list[str]  = []
        all_meta  : list[dict] = []

        for doc in documents:
            filename = doc["filename"]
            text     = doc.get("text", "").strip()
            doc_class = class_map.get(filename, doc.get("class", "Unknown"))

            if not text:
                # Image-based / empty PDF — index the filename so it at
                # least shows up in results (won't be semantically strong)
                text = f"Document: {filename}"

            chunks = _chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_texts.append(chunk)
                all_meta.append({
                    "filename"  : filename,
                    "class"     : doc_class,
                    "chunk_idx" : i,
                    "full_text" : text,       # kept for preview generation
                })

        print(f"Encoding {len(all_texts)} chunk(s) from {len(documents)} document(s)…")

        # Encode — returns (N, dim) float32 numpy array
        embeddings: np.ndarray = self.model.encode(
            all_texts,
            batch_size      = 32,
            show_progress_bar = False,
            convert_to_numpy  = True,
            normalize_embeddings = True,   # required for inner-product ≡ cosine
        )

        dim = embeddings.shape[1]

        # FAISS index: inner product on L2-normalised vectors = cosine similarity
        self.index    = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = all_meta

        print(f"  Index built: {self.index.ntotal} vector(s), dim={dim}")

    # ── search ───────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search for documents semantically similar to *query*.

        Returns a list of dicts (sorted by relevance) with keys:
            filename, class, score, preview, chunk_idx
        """
        if self.index is None or self.index.ntotal == 0:
            print("[WARNING] Index is empty — call build_index() first.")
            return []

        # Encode query with the same normalisation flag
        q_vec: np.ndarray = self.model.encode(
            [query],
            convert_to_numpy     = True,
            normalize_embeddings = True,
        ).astype(np.float32)

        # Retrieve more than top_k so we can de-duplicate by filename
        fetch_k  = min(top_k * 5, self.index.ntotal)
        scores, indices = self.index.search(q_vec, fetch_k)

        seen_files : set[str]  = set()
        results    : list[dict] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:          # FAISS returns -1 for padding
                continue
            meta     = self.metadata[idx]
            filename = meta["filename"]

            # One result per document (best-scoring chunk wins)
            if filename in seen_files:
                continue
            seen_files.add(filename)

            results.append({
                "filename"  : filename,
                "class"     : meta["class"],
                "score"     : round(float(score), 4),
                "preview"   : _build_preview(meta["full_text"], query),
                "chunk_idx" : meta["chunk_idx"],
            })

            if len(results) >= top_k:
                break

        return results

    # ── interactive shell ────────────────────────────────────────────────────

    def interactive_search(self) -> None:
        """
        Drop into a simple REPL so the user can type queries interactively.
        Exits on empty input or 'quit' / 'exit' / 'q'.
        """
        if self.index is None or self.index.ntotal == 0:
            print("[ERROR] No index available. Run build_index() first.")
            return

        print("\n" + "="*50)
        print("   SEMANTIC SEARCH  (type 'quit' to exit)")
        print("="*50)
        print("Example queries:")
        print("  • Find all documents mentioning payments due in January")
        print("  • Invoices from technology companies")
        print("  • Resumes with Python experience\n")

        while True:
            try:
                query = input("Search > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting search.")
                break

            if not query or query.lower() in ("quit", "exit", "q"):
                print("Exiting search.")
                break

            results = self.search(query, top_k=5)

            if not results:
                print("  No results found.\n")
                continue

            print(f"\n  Top {len(results)} result(s) for: '{query}'")
            print("  " + "-"*46)
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r['class']}] {r['filename']}  (score: {r['score']})")
                print(f"     {r['preview']}\n")