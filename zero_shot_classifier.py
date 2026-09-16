"""
Zero-shot embedding classifier for the Hermes Router Proxy.

Pre-computes embeddings for each category description and classifies
new messages by cosine similarity — no LLM calls needed.

Usage:
    from zero_shot_classifier import ZeroShotClassifier
    clf = ZeroShotClassifier(cfg)
    category = clf.classify("write a Python function")  # → "code"

Config (router_config.yaml):
    classifier:
      zero_shot:
        enabled: true
        model_name: "all-MiniLM-L6-v2"    # sentence-transformer model
        confidence_threshold: 0.35        # min cosine similarity to accept
        descriptions:                      # per-category seed phrases
          chat: ["hello", "how are you", "thanks", "what is", "trivia", "greeting"]
          code: ["write code", "debug", "function", "python", "refactor", "bug", "API"]
          devops: ["deploy", "docker", "kubernetes", "nginx", "systemctl", "LXC"]
          research: ["research", "analyze", "compare", "paper", "study", "evaluate"]
          homeassistant: ["home assistant", "automation", "zigbee", "MQTT", "smart home"]

If descriptions are not in config, the category labels + names are used.
The model is loaded once and cached (singleton).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

log = logging.getLogger("hermes-router")

# ── Singleton ────────────────────────────────────────────────────────────────
_model: Any = None  # SentenceTransformer instance
_model_name: str = ""


def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load the SentenceTransformer model (singleton)."""
    global _model, _model_name
    if _model is None or _model_name != model_name:
        try:
            from sentence_transformers import SentenceTransformer
            t0 = time.time()
            _model = SentenceTransformer(model_name, device="cpu")
            _model_name = model_name
            log.info("Zero-shot model loaded: %s (%.1fs)", model_name, time.time() - t0)
        except ImportError:
            log.warning("sentence-transformers not installed — zero-shot disabled")
            return None
        except Exception as exc:
            log.warning("Failed to load zero-shot model '%s': %s", model_name, exc)
            return None
    return _model


class ZeroShotClassifier:
    """
    Zero-shot category classifier using sentence embeddings + cosine similarity.

    Pre-computes embeddings for category seed phrases on first use,
    then classifies new messages by finding the nearest category.
    No LLM calls, no training data needed.
    """

    def __init__(self, cfg: dict):
        zs_cfg = cfg.get("classifier", {}).get("zero_shot", {})
        self.enabled = zs_cfg.get("enabled", False)
        self.model_name = zs_cfg.get("model_name", "all-MiniLM-L6-v2")
        self.confidence_threshold = zs_cfg.get("confidence_threshold", 0.35)

        # Build category seed phrases
        cats = cfg.get("categories", {})
        descriptions = zs_cfg.get("descriptions", {})

        self.category_names: list[str] = []
        self.seed_texts: list[str] = []
        self._category_embeddings: np.ndarray | None = None

        for name, cat_cfg in cats.items():
            self.category_names.append(name)
            label = cat_cfg.get("label", name)
            # Use custom descriptions if provided, otherwise use label + name
            if name in descriptions:
                seeds = descriptions[name]
            else:
                seeds = [label, name, f"{name} task"]
            self.seed_texts.extend(seeds)

        # Track which seed index maps to which category
        self._seed_to_category: list[str] = []
        idx = 0
        for name in self.category_names:
            if name in descriptions:
                count = len(descriptions[name])
            else:
                count = 3  # label, name, "name task"
            self._seed_to_category.extend([name] * count)
            idx += count

    def _compute_embeddings(self) -> np.ndarray:
        """Compute embeddings for all seed texts. Called once on first classify."""
        model = _get_model(self.model_name)
        if model is None:
            raise RuntimeError("SentenceTransformer model not available")

        embeddings = model.encode(
            self.seed_texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings)

    def classify(self, text: str) -> tuple[str, float]:
        """
        Classify text into a category using cosine similarity.

        Returns (category_name, confidence) where confidence is the
        maximum cosine similarity score (0.0 to 1.0).
        Falls back to (first_category, 0.0) on any error.
        """
        if not self.enabled or not self.category_names:
            return self.category_names[0] if self.category_names else "chat", 0.0

        try:
            if self._category_embeddings is None:
                self._category_embeddings = self._compute_embeddings()

            model = _get_model(self.model_name)
            if model is None:
                return self.category_names[0], 0.0

            # Embed the input text
            query_emb = model.encode(
                [text], normalize_embeddings=True, show_progress_bar=False
            )
            query_emb = np.array(query_emb)[0]

            # Cosine similarity (embeddings are already normalized)
            similarities = self._category_embeddings @ query_emb

            # Find the best seed match, then map to its category
            # Compute per-category average for confidence
            cat_scores: dict[str, list[float]] = {}
            for i, score in enumerate(similarities):
                cat = self._seed_to_category[i]
                cat_scores.setdefault(cat, []).append(float(score))

            cat_avg = {c: sum(s) / len(s) for c, s in cat_scores.items()}
            best_avg_category = max(cat_avg, key=cat_avg.get)
            best_avg_score = cat_avg[best_avg_category]

            # Use the average-based category (more stable than single best seed)
            return best_avg_category, best_avg_score

        except Exception as exc:
            log.warning("Zero-shot classify error: %s", exc)
            return self.category_names[0] if self.category_names else "chat", 0.0


# ── Singleton instance ───────────────────────────────────────────────────────
_instance: ZeroShotClassifier | None = None


def get_zero_shot(cfg: dict) -> ZeroShotClassifier | None:
    """Return the singleton ZeroShotClassifier, or None if disabled."""
    global _instance
    if _instance is None:
        _instance = ZeroShotClassifier(cfg)
    if not _instance.enabled:
        return None
    return _instance