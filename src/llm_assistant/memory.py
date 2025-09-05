"""Advanced FAISS-backed user memory module.

This module stores user-specific facts as normalized embeddings and
upgrades from a flat index to an IVF index when the number of stored
vectors exceeds a threshold.  The index and raw facts can be persisted
and reloaded from disk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Set

import faiss
from sentence_transformers import SentenceTransformer


class UserMemoryModule:
    """Stores and retrieves user memories using a FAISS index."""

    UPGRADE_THRESHOLD = 1024
    IVF_NLIST = 100

    def __init__(self, model: SentenceTransformer, file_path: Optional[str | Path] = None) -> None:
        self.model = model
        self.embedding_dim = model.get_sentence_embedding_dimension()
        self.file_path: Optional[Path] = Path(file_path) if file_path else None

        self.memory_facts: List[str] = []
        self.memory_set: Set[str] = set()
        self._initialize_flat_index()

        if self.file_path:
            self.file_path.mkdir(parents=True, exist_ok=True)
            self.load()

    def _initialize_flat_index(self) -> None:
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        logging.info("Initialized with a simple flat index (IndexFlatIP).")

    def _upgrade_to_ivf_index(self) -> None:
        logging.info(
            f"Memory size ({self.index.ntotal}) reached upgrade threshold. Upgrading to IndexIVFFlat."
        )
        existing_vectors = self.index.reconstruct_n(0, self.index.ntotal)
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        new_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.IVF_NLIST, faiss.METRIC_INNER_PRODUCT)
        new_index.train(existing_vectors)
        new_index.add(existing_vectors)
        self.index = new_index
        logging.info("Index upgrade complete.")

    def add_memory(self, facts: List[str]) -> None:
        """Add new facts to memory, upgrading the index if needed."""
        unique_facts = [fact for fact in facts if fact and fact not in self.memory_set]
        if not unique_facts:
            return

        is_flat = isinstance(self.index, faiss.IndexFlatIP)
        if is_flat and self.index.ntotal >= self.UPGRADE_THRESHOLD:
            self._upgrade_to_ivf_index()

        logging.info(f"Adding {len(unique_facts)} new facts to memory.")
        embeddings = self.model.encode(unique_facts, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.memory_facts.extend(unique_facts)
        self.memory_set.update(unique_facts)

    def retrieve_memory(self, query: str, top_k: int = 10) -> List[str]:
        """Return the most similar stored facts for ``query``."""
        if self.index.ntotal == 0:
            return []

        if hasattr(self.index, "nprobe"):
            self.index.nprobe = min(20, self.IVF_NLIST)

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        k = min(top_k, self.index.ntotal)
        _, indices = self.index.search(query_embedding, k)
        return [self.memory_facts[i] for i in indices[0] if i != -1]

    def save(self) -> None:
        """Persist the FAISS index and memory facts to disk."""
        if not self.file_path:
            logging.warning("No file_path specified. Cannot save memory.")
            return

        index_path = self.file_path / "memory.index"
        facts_path = self.file_path / "memory_facts.json"
        logging.info(f"Saving memory to {self.file_path}...")
        faiss.write_index(self.index, str(index_path))
        with open(facts_path, "w", encoding="utf-8") as fh:
            json.dump(self.memory_facts, fh)
        logging.info("Save complete.")

    def load(self) -> None:
        """Load the FAISS index and memory facts from disk."""
        if not self.file_path:
            return

        index_path = self.file_path / "memory.index"
        facts_path = self.file_path / "memory_facts.json"
        if index_path.exists() and facts_path.exists():
            logging.info(f"Loading memory from {self.file_path}...")
            self.index = faiss.read_index(str(index_path))
            with open(facts_path, "r", encoding="utf-8") as fh:
                self.memory_facts = json.load(fh)
            self.memory_set = set(self.memory_facts)
            logging.info(
                f"Loaded {len(self.memory_facts)} facts. Index type: {type(self.index).__name__}"
            )
        else:
            logging.info("No existing memory found at path. Starting fresh.")

    def clear(self) -> None:
        """Clear all stored facts and reset the index."""
        self._initialize_flat_index()
        self.memory_facts = []
        self.memory_set = set()
        logging.info("Memory cleared.")

    def __len__(self) -> int:
        return self.index.ntotal
