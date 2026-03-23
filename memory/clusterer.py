"""Context-based document clustering using DBSCAN on ChromaDB embeddings.

Clusters similar chunks together and writes cluster labels back to metadata,
enabling filtered search like where={"cluster_label": "learning_react"}.

Designed to run twice daily via daily_loop.py or rag_scheduler.py.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from dataclasses import dataclass, field

from config import CHROMA_COLLECTION


@dataclass
class Cluster:
    cluster_id: int
    label: str = ""
    chunk_ids: list[str] = field(default_factory=list)
    sample_texts: list[str] = field(default_factory=list)
    size: int = 0


class DocumentClusterer:
    """Cluster ChromaDB documents by embedding similarity."""

    def __init__(self, vector_store):
        self.store = vector_store

    def cluster_all(
        self,
        eps: float = 0.35,
        min_samples: int = 3,
        batch_size: int = 5000,
    ) -> list[Cluster]:
        """Run DBSCAN clustering on all document embeddings.

        Args:
            eps: Maximum distance between two samples in the same cluster.
                 For cosine distance on MiniLM embeddings, 0.3-0.4 works well.
            min_samples: Minimum number of samples to form a cluster.
            batch_size: How many docs to fetch from ChromaDB at once.
        """
        # Fetch all embeddings from ChromaDB
        total = self.store.count()
        if total < min_samples * 2:
            print(f"Too few documents ({total}) for clustering. Skipping.")
            return []

        print(f"Fetching {total} embeddings for clustering...")
        all_ids = []
        all_embeddings = []
        all_texts = []

        offset = 0
        while offset < total:
            result = self.store.collection.get(
                limit=batch_size,
                offset=offset,
                include=["embeddings", "documents"],
            )
            if not result["ids"]:
                break
            all_ids.extend(result["ids"])
            all_embeddings.extend(result["embeddings"])
            all_texts.extend(result["documents"])
            offset += len(result["ids"])

        if not all_embeddings:
            return []

        # Normalize embeddings for cosine distance
        embeddings = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        print(f"Running DBSCAN (eps={eps}, min_samples={min_samples}) on {len(embeddings)} docs...")
        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="cosine",
            n_jobs=-1,
        ).fit(embeddings)

        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        print(f"Found {n_clusters} clusters, {n_noise} noise points.")

        # Build cluster objects
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue  # skip noise
            if label not in clusters:
                clusters[label] = Cluster(cluster_id=label)
            clusters[label].chunk_ids.append(all_ids[i])
            if len(clusters[label].sample_texts) < 5:
                clusters[label].sample_texts.append(all_texts[i][:200])
            clusters[label].size += 1

        return list(clusters.values())

    def label_clusters(self, clusters: list[Cluster]) -> list[Cluster]:
        """Use LLM to generate human-readable labels for each cluster."""
        if not clusters:
            return clusters

        try:
            from twin.llm_client import chat_completion
        except ImportError:
            # Fallback: use numeric labels
            for c in clusters:
                c.label = f"cluster_{c.cluster_id}"
            return clusters

        system = (
            "You are labeling document clusters. For each cluster, I'll give you sample texts. "
            "Generate a short, descriptive label (2-4 words, snake_case) that captures the theme.\n\n"
            "Examples: learning_react, fitness_tracking, career_goals, mindset_confidence, "
            "code_architecture, nutrition_habits, weekly_planning, relationship_advice\n\n"
            "Respond with ONLY the labels, one per line, in order."
        )

        # Batch clusters into groups to reduce LLM calls
        batch_size = 15
        for i in range(0, len(clusters), batch_size):
            batch = clusters[i:i + batch_size]
            user_msg = ""
            for j, c in enumerate(batch):
                samples = " | ".join(c.sample_texts[:3])
                user_msg += f"Cluster {j + 1} ({c.size} docs): {samples}\n\n"

            try:
                raw = chat_completion(
                    system=system,
                    messages=[{"role": "user", "content": user_msg}],
                    max_tokens=500,
                )
                labels = [line.strip().lower().replace(" ", "_").replace("-", "_")
                          for line in raw.strip().split("\n") if line.strip()]
                for j, c in enumerate(batch):
                    if j < len(labels):
                        # Clean up: remove numbering prefixes like "1. " or "cluster_1: "
                        label = labels[j]
                        for prefix in ["cluster_", f"{j + 1}.", f"{j + 1}:"]:
                            if label.startswith(prefix):
                                label = label[len(prefix):].strip("_").strip()
                        c.label = label or f"cluster_{c.cluster_id}"
                    else:
                        c.label = f"cluster_{c.cluster_id}"
            except Exception:
                for c in batch:
                    c.label = f"cluster_{c.cluster_id}"

        return clusters

    def update_metadata_with_clusters(self, clusters: list[Cluster]) -> int:
        """Write cluster_id and cluster_label back to ChromaDB metadata.

        Returns total number of chunks updated.
        """
        total_updated = 0
        batch_size = 100

        for cluster in clusters:
            if not cluster.label:
                continue

            ids = cluster.chunk_ids
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                # Get existing metadata to merge
                existing = self.store.collection.get(
                    ids=batch_ids,
                    include=["metadatas"],
                )
                if not existing["metadatas"]:
                    continue

                new_metadatas = []
                for meta in existing["metadatas"]:
                    meta["cluster_id"] = str(cluster.cluster_id)
                    meta["cluster_label"] = cluster.label
                    new_metadatas.append(meta)

                self.store.update_metadata(batch_ids, new_metadatas)
                total_updated += len(batch_ids)

        print(f"Updated {total_updated} chunks with cluster labels.")
        return total_updated

    def run(self, eps: float = 0.35, min_samples: int = 3) -> dict:
        """Full clustering pipeline: cluster → label → update metadata.

        Returns summary dict.
        """
        clusters = self.cluster_all(eps=eps, min_samples=min_samples)
        if not clusters:
            return {"clusters": 0, "labeled": 0, "updated": 0}

        clusters = self.label_clusters(clusters)
        updated = self.update_metadata_with_clusters(clusters)

        cluster_summary = {c.label: c.size for c in clusters}
        return {
            "clusters": len(clusters),
            "labeled": sum(1 for c in clusters if c.label),
            "updated": updated,
            "cluster_sizes": cluster_summary,
        }
