# src/hybrid_index.py

import os
import json
import pickle
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class HybridIndex:
    """
    Hybrid retrieval index:
    - Dense retrieval: FAISS + sentence-transformer embeddings
    - Sparse retrieval: BM25 keyword search
    - Hybrid retrieval: weighted merge of dense + sparse scores
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ):
        self.embedding_model_name = embedding_model_name
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        self.embedder = SentenceTransformer(embedding_model_name)

        self.chunks: List[Dict[str, Any]] = []
        self.faiss_index = None
        self.bm25 = None
        self.tokenized_corpus = None

    # ---------------------------------------------------------
    # Basic tokenizer for BM25
    # ---------------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenizer for BM25.
        Later you can improve this with:
        - regex cleaning
        - stopword removal
        - legal term normalization
        """
        return text.lower().split()

    # ---------------------------------------------------------
    # Build index
    # ---------------------------------------------------------
    def build(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build both dense FAISS index and sparse BM25 index.

        Expected chunk format:
        {
            "text": "...",
            "doc_name": "contract.pdf",
            "page": 4,
            "chunk_id": "contract_p4_c0"
        }
        """

        if not chunks:
            raise ValueError("No chunks provided to build index.")

        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]

        # ----------------------------
        # Build BM25 sparse index
        # ----------------------------
        self.tokenized_corpus = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # ----------------------------
        # Build FAISS dense index
        # ----------------------------
        embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        embeddings = embeddings.astype("float32")

        # Normalize vectors so inner product behaves like cosine similarity
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)

    # ---------------------------------------------------------
    # Dense search only
    # ---------------------------------------------------------
    def dense_search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        if self.faiss_index is None:
            raise ValueError("FAISS index not built.")

        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(query_embedding)

        scores, indices = self.faiss_index.search(query_embedding, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            chunk = self.chunks[idx].copy()
            chunk["dense_score"] = float(score)
            chunk["sparse_score"] = 0.0
            chunk["hybrid_score"] = float(score)
            results.append(chunk)

        return results

    # ---------------------------------------------------------
    # Sparse search only
    # ---------------------------------------------------------
    def sparse_search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            raise ValueError("BM25 index not built.")

        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)

        top_indices = np.argsort(bm25_scores)[::-1][:top_k]

        results = []

        max_score = float(np.max(bm25_scores)) if np.max(bm25_scores) > 0 else 1.0

        for idx in top_indices:
            raw_score = float(bm25_scores[idx])
            normalized_score = raw_score / max_score

            chunk = self.chunks[idx].copy()
            chunk["dense_score"] = 0.0
            chunk["sparse_score"] = normalized_score
            chunk["hybrid_score"] = normalized_score
            results.append(chunk)

        return results

    # ---------------------------------------------------------
    # Hybrid search
    # ---------------------------------------------------------
    def hybrid_search(
        self,
        query: str,
        top_k: int = 8,
        dense_k: int = 12,
        sparse_k: int = 12,
    ) -> List[Dict[str, Any]]:
        """
        Runs dense + sparse search, merges scores, and returns top_k chunks.

        dense_weight and sparse_weight decide contribution:

        hybrid_score = dense_weight * dense_score + sparse_weight * sparse_score
        """

        if self.faiss_index is None:
            raise ValueError("FAISS index not built.")

        if self.bm25 is None:
            raise ValueError("BM25 index not built.")

        combined_scores: Dict[int, Dict[str, float]] = {}

        # ----------------------------
        # Dense retrieval
        # ----------------------------
        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(query_embedding)

        dense_scores, dense_indices = self.faiss_index.search(query_embedding, dense_k)

        for score, idx in zip(dense_scores[0], dense_indices[0]):
            if idx == -1:
                continue

            idx = int(idx)
            dense_score = float(score)

            if idx not in combined_scores:
                combined_scores[idx] = {
                    "dense_score": 0.0,
                    "sparse_score": 0.0,
                }

            combined_scores[idx]["dense_score"] = dense_score

        # ----------------------------
        # Sparse BM25 retrieval
        # ----------------------------
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)

        sparse_indices = np.argsort(bm25_scores)[::-1][:sparse_k]

        max_bm25 = float(np.max(bm25_scores)) if np.max(bm25_scores) > 0 else 1.0

        for idx in sparse_indices:
            idx = int(idx)

            raw_sparse_score = float(bm25_scores[idx])
            sparse_score = raw_sparse_score / max_bm25

            if idx not in combined_scores:
                combined_scores[idx] = {
                    "dense_score": 0.0,
                    "sparse_score": 0.0,
                }

            combined_scores[idx]["sparse_score"] = sparse_score

        # ----------------------------
        # Merge weighted scores
        # ----------------------------
        ranked_results: List[Tuple[int, float]] = []

        for idx, scores in combined_scores.items():
            dense_score = scores["dense_score"]
            sparse_score = scores["sparse_score"]

            hybrid_score = (
                self.dense_weight * dense_score
                + self.sparse_weight * sparse_score
            )

            ranked_results.append((idx, hybrid_score))

        ranked_results.sort(key=lambda x: x[1], reverse=True)

        # ----------------------------
        # Return chunks with scores
        # ----------------------------
        final_results = []

        for idx, hybrid_score in ranked_results[:top_k]:
            chunk = self.chunks[idx].copy()

            chunk["dense_score"] = combined_scores[idx]["dense_score"]
            chunk["sparse_score"] = combined_scores[idx]["sparse_score"]
            chunk["hybrid_score"] = hybrid_score

            final_results.append(chunk)

        return final_results

    # Alias for convenience
    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        return self.hybrid_search(query=query, top_k=top_k)

    # ---------------------------------------------------------
    # Save index to disk
    # ---------------------------------------------------------
    def save(self, index_dir: str) -> None:
        """
        Save FAISS index, BM25 index, chunks, and config.
        """

        os.makedirs(index_dir, exist_ok=True)

        if self.faiss_index is None:
            raise ValueError("Cannot save. FAISS index not built.")

        if self.bm25 is None:
            raise ValueError("Cannot save. BM25 index not built.")

        faiss.write_index(
            self.faiss_index,
            os.path.join(index_dir, "faiss.index")
        )

        with open(os.path.join(index_dir, "bm25.pkl"), "wb") as f:
            pickle.dump(self.bm25, f)

        with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        config = {
            "embedding_model_name": self.embedding_model_name,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
        }

        with open(os.path.join(index_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    # ---------------------------------------------------------
    # Load index from disk
    # ---------------------------------------------------------
    @classmethod
    def load(cls, index_dir: str) -> "HybridIndex":
        """
        Load FAISS index, BM25 index, chunks, and config.
        """

        config_path = os.path.join(index_dir, "config.json")

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {
                "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "dense_weight": 0.6,
                "sparse_weight": 0.4,
            }

        obj = cls(
            embedding_model_name=config["embedding_model_name"],
            dense_weight=config["dense_weight"],
            sparse_weight=config["sparse_weight"],
        )

        obj.faiss_index = faiss.read_index(
            os.path.join(index_dir, "faiss.index")
        )

        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            obj.bm25 = pickle.load(f)

        with open(os.path.join(index_dir, "chunks.json"), "r", encoding="utf-8") as f:
            obj.chunks = json.load(f)

        return obj