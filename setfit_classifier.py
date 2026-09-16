"""
SetFit classifier for the Hermes Router Proxy.

Tier-1 classifier: fine-tuned sentence-transformer (all-MiniLM-L6-v2) with
classification head. ~10ms inference on CPU. More accurate than zero-shot
cosine similarity because the model is trained on our actual category data.

Usage:
    from setfit_classifier import get_setfit
    clf = get_setfit(cfg)
    if clf is not None:
        label, confidence = clf.classify("write a python function")

Config (router_config.yaml):
    classifier:
      setfit:
        enabled: true
        model_path: .router/setfit/           # saved SetFit model directory
        confidence_threshold: 0.55            # min probability to accept
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("hermes-router")

# ── Singleton ────────────────────────────────────────────────────────────────
_model: Any = None  # SetFitModel instance
_model_dir: str = ""


class SetFitClassifier:
    """
    SetFit-based text classifier.

    Loads a fine-tuned SetFit model (all-MiniLM-L6-v2 base) and provides
    ~10ms classification on CPU. The model is trained on the same pretrain
    traces as the surrogate, but with contrastive fine-tuning for better
    accuracy than zero-shot cosine similarity.
    """

    def __init__(self, cfg: dict):
        sf_cfg = cfg.get("classifier", {}).get("setfit", {})
        self.enabled = sf_cfg.get("enabled", False)
        self.confidence_threshold = sf_cfg.get("confidence_threshold", 0.55)

        model_path = sf_cfg.get("model_path", ".router/setfit")
        p = Path(model_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / model_path
        self.model_path = p

        self._model: Any = None
        self._labels: list[str] = []

        if self.enabled and (p / "config.json").exists():
            self._load()
        elif self.enabled:
            log.warning("SetFit enabled but model not found at %s — disabling", p)
            self.enabled = False

    def _load(self) -> None:
        """Load the SetFit model."""
        try:
            from setfit import SetFitModel

            t0 = time.time()
            self._model = SetFitModel.from_pretrained(str(self.model_path))

            # Load label mapping if available
            labels_file = self.model_path / "labels.json"
            if labels_file.exists():
                meta = json.loads(labels_file.read_text())
                self._labels = meta.get("labels", [])
            else:
                # Try to get labels from model config
                if hasattr(self._model, "labels") and self._model.labels:
                    self._labels = list(self._model.labels)
                else:
                    self._labels = []

            log.info(
                "SetFit model loaded: %s (%d labels, %.1fs)",
                self.model_path,
                len(self._labels),
                time.time() - t0,
            )
        except Exception as exc:
            log.warning("Failed to load SetFit model from %s: %s", self.model_path, exc)
            self.enabled = False
            self._model = None

    def classify(self, text: str) -> tuple[str, float]:
        """
        Classify text into a category.

        Returns (category_name, confidence) where confidence is the
        max softmax probability (0.0 to 1.0).
        Returns ("", 0.0) on any error.
        """
        if not self.enabled or self._model is None:
            return "", 0.0

        try:
            # SetFit predict returns a list of label strings
            preds = self._model.predict([text])
            label = str(preds[0])

            # Get probabilities for confidence
            confidence = 0.9  # fallback
            if hasattr(self._model, "predict_proba"):
                try:
                    probs = self._model.predict_proba([text])
                    # probs can be a torch tensor or numpy array
                    import numpy as np
                    if hasattr(probs, "cpu"):
                        # PyTorch tensor
                        probs = probs.cpu().numpy()
                    if hasattr(probs, "tolist"):
                        probs = probs.tolist()
                    confidence = float(max(probs[0]))
                except Exception:
                    pass

            return label, confidence
        except Exception as exc:
            log.warning("SetFit predict error: %s", exc)
            return "", 0.0

    @property
    def is_available(self) -> bool:
        return self.enabled and self._model is not None


# ── Singleton instance ───────────────────────────────────────────────────────
_instance: SetFitClassifier | None = None


def get_setfit(cfg: dict) -> SetFitClassifier | None:
    """Return the singleton SetFitClassifier, or None if disabled."""
    global _instance
    if _instance is None:
        _instance = SetFitClassifier(cfg)
    if not _instance.is_available:
        return None
    return _instance