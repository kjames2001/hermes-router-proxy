#!/usr/bin/env python3
"""
TRACER-inspired offline surrogate fitting tool.

Reads classification traces from traces/router-trace-*.jsonl (new format) and
.router/traces.jsonl (legacy format), fits traditional ML models (TF-IDF + LR/SGD)
AND sentence-transformer embedding models (all-MiniLM-L6-v2 + LR/SGD), cross-validates
them against the LLM teacher labels, and saves the best model with a calibrated
acceptor gate.

Usage:
    python3 fit_surrogate.py                    # fit with defaults
    python3 fit_surrogate.py --target 0.95     # target 95% teacher agreement
    python3 fit_surrogate.py --output DIR       # custom output directory

Output (.router/surrogate/):
    manifest.json       - Method, coverage, teacher agreement, label space
    pipeline.joblib     - Surrogate + vectorizer pipeline (sklearn Pipeline)
    acceptor.joblib     - Calibrated acceptor gate (per-input confidence)

Based on: https://github.com/adrida/tracer (TRACER: Trace-Based Adaptive
Cost-Efficient Routing for LLM Classification)
"""

from __future__ import annotations

import argparse
import json
import glob
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# sklearn components
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import joblib
except ImportError:
    import pickle as joblib  # fallback

# Shared model classes (used for joblib deserialization path consistency)
from surrogate_models import SentenceTransformerVectorizer, HAS_SENTENCE_TRANSFORMERS


# ── Trace Loading ──────────────────────────────────────────────────────

def load_traces_all(script_dir: Path) -> list[dict]:
    """Load trace events from all available sources (new + legacy format).

    Returns a flat list of trace dicts with normalized fields:
      text (str)   — user message content
      label (str)  — 'simple' or 'complex'
      source (str) — 'classifier' or 'surrogate' or 'keyword' or 'cache'
      model (str|None)
    """
    events: list[dict] = []

    # 1. New format: traces/router-trace-YYYYMMDD.jsonl
    trace_dir = script_dir / "traces"
    if trace_dir.exists():
        for tf in sorted(glob.glob(str(trace_dir / "router-trace-*.jsonl"))):
            try:
                with open(tf, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            events.append(json.loads(line))
                        except (json.JSONDecodeError, ValueError):
                            pass
            except OSError:
                pass

    # 2. Legacy format: .router/traces.jsonl
    legacy_path = script_dir / ".router" / "traces.jsonl"
    if legacy_path.exists():
        try:
            with open(legacy_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except (json.JSONDecodeError, ValueError):
                        pass
        except OSError:
            pass

    return events


def extract_classifier_traces(events: list[dict]) -> list[dict]:
    """Filter and normalize events to classifier-source training examples.

    Handles both new format (event='classify') and legacy format (source='classifier').
    Returns list of dicts with {text, label, source, model}.
    """
    results = []
    for ev in events:
        etype = ev.get("event", "") or ev.get("source", "")

        # New format: event == "classify"
        if ev.get("event") == "classify":
            text = ev.get("user_message_preview", "").strip()
            label = ev.get("classifier_result", "").strip().lower()
            model = ev.get("model", "")
            source = "surrogate" if model.startswith("surrogate/") else "classifier"
        # Legacy format: source == "classifier"
        elif ev.get("source") == "classifier":
            text = ev.get("input", "").strip()
            label = ev.get("label", "").strip().lower()
            model = ev.get("model", None)
            source = "classifier"
        else:
            # Skip route, cache_hit, deviation, stream_error events
            continue

        if not text or label not in ("simple", "complex"):
            continue

        # Only use LLM-classifier traces as teacher labels (not surrogate's own predictions)
        if source != "classifier":
            continue

        results.append({"text": text, "label": label, "source": source, "model": model})

    return results


# ── Dataset Preparation ────────────────────────────────────────────────

def prepare_dataset(traces: list[dict]) -> tuple[list[str], list[str]]:
    """Extract (input_text, label) pairs from normalized classifier traces."""
    texts = [t["text"] for t in traces]
    labels = [t["label"] for t in traces]
    return texts, labels


# ── Candidate Fitting ───────────────────────────────────────────────────

def fit_candidates(
    texts: list[str],
    labels: list[str],
    target_agreement: float = 0.95,
    teacher_model: str = "unknown",
    prefer_embeddings: bool = False,
) -> dict:
    """
    Fit candidate surrogates (TF-IDF + embedding-based) and evaluate against teacher labels.
    Returns the best candidate dict with pipeline, metrics, and acceptor.
    """
    X = np.array(texts)
    y = np.array(labels)

    label_counts = {}
    for label in y:
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\n  Dataset: {len(texts)} traces from LLM classifier source")
    print(f"  Labels: {label_counts}")

    if len(label_counts) < 2:
        print("  ERROR: Need at least 2 classes (simple + complex) to train.")
        print("  Collect more traces with diverse queries and re-run.")
        sys.exit(1)

    if len(texts) < 20:
        print("  WARNING: Very few traces (< 20). Model quality will be limited.")
        print("  Collect more traces for better results.")
    if len(texts) < 10:
        print("  ERROR: Need at least 10 classifier-source traces to fit a surrogate.")
        print("  Current traces:", len(texts))
        sys.exit(1)

    # ── Candidate models ───────────────────────────────────────────────────
    candidates = [
        ("tfidf_lr", Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                stop_words="english",
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
            )),
        ])),
        ("tfidf_sgd", Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                stop_words="english",
            )),
            ("clf", SGDClassifier(
                loss="modified_huber",
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ])),
    ]

    if HAS_SENTENCE_TRANSFORMERS:
        print("  Sentence transformers available — adding embedding candidates...")
        # MiniLM embeddings (384 dims, normalized) + LR
        candidates.append(("embeddings_lr", Pipeline([
            ("emb", SentenceTransformerVectorizer(model_name="all-MiniLM-L6-v2")),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                C=1.0,
            )),
        ])))
        # MiniLM embeddings + SGD
        candidates.append(("embeddings_sgd", Pipeline([
            ("emb", SentenceTransformerVectorizer(model_name="all-MiniLM-L6-v2")),
            ("clf", SGDClassifier(
                loss="modified_huber",
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ])))
    else:
        print("  sentence-transformers not installed — skipping embedding candidates.")

    actual_splits = max(2, min(5, len(texts) // 4))
    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)

    best_name = None
    best_pipeline = None
    best_score = 0.0
    best_report = ""
    results = []

    for name, pipeline in candidates:
        print(f"\n  Fitting {name}...")
        t_fit = time.time()

        # Cross-validate teacher agreement
        cv_scores = cross_val_score(
            pipeline, X, y,
            cv=actual_splits,
            scoring="accuracy",
            n_jobs=1,  # sentence-transformers doesn't support parallel pickling well
        )
        mean_score = cv_scores.mean()

        # Fit on full dataset
        pipeline.fit(X, y)

        # Get predictions on training set for detailed report
        y_pred = pipeline.predict(X)
        report = classification_report(y, y_pred, zero_division=0)
        accuracy = accuracy_score(y, y_pred)

        elapsed = time.time() - t_fit
        results.append({
            "name": name,
            "cv_mean": round(mean_score, 4),
            "cv_std": round(cv_scores.std(), 4),
            "train_accuracy": round(accuracy, 4),
            "fit_time_sec": round(elapsed, 1),
        })

        print(f"    CV accuracy: {mean_score:.4f} ± {cv_scores.std():.4f}")
        print(f"    Train accuracy: {accuracy:.4f}")
        print(f"    Fit time: {elapsed:.1f}s")

        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_pipeline = pipeline
            best_report = report

    if best_name is None:
        print("  ERROR: No candidate model succeeded.")
        sys.exit(1)

    # If prefer_embeddings and best is TF-IDF, check if an embedding candidate is within 5%
    if prefer_embeddings and not best_name.startswith("embeddings_"):
        emb_candidates = [r for r in results if r["name"].startswith("embeddings_")]
        if emb_candidates:
            best_emb = max(emb_candidates, key=lambda r: r["cv_mean"])
            if best_emb["cv_mean"] >= best_score - 0.05:
                # Swap to embedding candidate
                for name, pipeline in candidates:
                    if name == best_emb["name"]:
                        best_name = name
                        best_score = best_emb["cv_mean"]
                        best_pipeline = pipeline
                        best_report = classification_report(y, pipeline.predict(X), zero_division=0)
                        print(f"  Prefer-embeddings: using {best_name} (CV={best_score:.4f}) "
                              f"over best TF-IDF (CV={results[0]['cv_mean']:.4f})")
                        break

    # Build acceptor gate — calibrated classifier for per-input confidence
    print(f"\n  Best candidate: {best_name} (CV accuracy: {best_score:.4f})")

    # Calibrate with the best pipeline
    cal_cv = max(2, min(3, len(texts) // 4))
    acceptor = CalibratedClassifierCV(
        best_pipeline,
        cv=cal_cv,
        method="sigmoid",
    )
    acceptor.fit(X, y)

    # Compute acceptor thresholds at different confidence levels
    probas = acceptor.predict_proba(X)
    max_probas = probas.max(axis=1)

    thresholds = {}
    for target in [0.80, 0.85, 0.90, 0.95, 0.99]:
        sorted_probs = np.sort(max_probas)
        idx = max(0, int((1 - target) * len(sorted_probs)) - 1)
        thresholds[str(target)] = round(float(sorted_probs[idx]), 4)

    # Compute coverage at target agreement
    threshold_at_target = thresholds[str(target_agreement)]
    above_threshold = max_probas >= threshold_at_target
    coverage = above_threshold.sum() / len(max_probas) if len(max_probas) > 0 else 0.0
    # Teacher agreement on the accepted partition
    y_pred_full = best_pipeline.predict(X)
    accepted_mask = above_threshold
    if accepted_mask.sum() > 0:
        accepted_agreement = float(accuracy_score(y[accepted_mask], y_pred_full[accepted_mask]))
    else:
        accepted_agreement = 0.0

    # Save results
    output_dir = Path(__file__).resolve().parent / ".router" / "surrogate"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect if best pipeline uses embeddings
    uses_embeddings = best_name.startswith("embeddings_")

    # Manifest
    manifest = {
        "teacher_model": teacher_model,
        "surrogate_method": best_name,
        "uses_embeddings": uses_embeddings,
        "target_teacher_agreement": target_agreement,
        "cv_accuracy": round(best_score, 4),
        "train_accuracy": round(accuracy_score(y, best_pipeline.predict(X)), 4),
        "accepted_agreement": accepted_agreement,
        "coverage": round(float(coverage), 4),
        "label_space": sorted(set(y)),
        "label_counts": label_counts,
        "n_traces": len(texts),
        "thresholds": thresholds,
        "acceptor_threshold": float(threshold_at_target),
        "candidates": results,
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Pipeline
    joblib.dump(best_pipeline, output_dir / "pipeline.joblib")

    # Acceptor
    joblib.dump(acceptor, output_dir / "acceptor.joblib")

    print(f"\n  {'='*60}")
    print(f"  Surrogate Fitting Results")
    print(f"  {'='*60}")
    print(f"  Teacher model:      {manifest['teacher_model']}")
    print(f"  Surrogate method:   {best_name}")
    print(f"  Uses embeddings:    {uses_embeddings}")
    print(f"  CV accuracy:        {best_score:.4f}")
    print(f"  Target agreement:   {target_agreement}")
    print(f"  Accepted agreement: {accepted_agreement:.4f}")
    print(f"  Coverage:           {coverage:.2%} of traffic handled locally")
    print(f"  Acceptor threshold: {threshold_at_target:.4f} (at {target_agreement} agreement)")
    print(f"  Thresholds:         {thresholds}")
    print(f"  {'='*60}")
    print(f"\n  Classification report (full training set):")
    print(best_report)
    print(f"\n  Saved to: {output_dir}/")
    print(f"    manifest.json  — metadata and metrics")
    print(f"    pipeline.joblib — surrogate model")
    print(f"    acceptor.joblib — calibrated acceptor gate")

    return manifest


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TRACER-inspired surrogate fitting for the Hermes router proxy"
    )
    parser.add_argument(
        "--target", type=float, default=0.95,
        help="Target teacher agreement (default: 0.95)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (default: .router/surrogate/)",
    )
    parser.add_argument(
        "--prefer-embeddings", action="store_true", default=False,
        help="Prefer embedding-based model over TF-IDF if within 5% CV accuracy",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = args.output or script_dir / ".router" / "surrogate"

    print(f"\n  TRACER Surrogate Fitting")
    print(f"  Trace sources: traces/router-trace-*.jsonl + .router/traces.jsonl")
    print(f"  Target teacher agreement: {args.target}")
    if HAS_SENTENCE_TRANSFORMERS:
        print(f"  Sentence transformers: AVAILABLE (all-MiniLM-L6-v2)")
    else:
        print(f"  Sentence transformers: NOT AVAILABLE (pip install sentence-transformers)")

    events = load_traces_all(script_dir)
    print(f"  Loaded {len(events)} raw trace events")

    classifier_traces = extract_classifier_traces(events)
    print(f"  Found {len(classifier_traces)} classifier-source traces (teacher labels)")

    if not classifier_traces:
        print("  No classifier-source traces found. Collect more data and re-run.")
        sys.exit(1)

    texts, labels = prepare_dataset(classifier_traces)
    if not texts:
        print("  No valid classifier traces found.")
        sys.exit(1)

    # Detect teacher model from traces
    teacher_model = "unknown"
    for t in classifier_traces:
        if t.get("model"):
            teacher_model = t["model"]
            break

    manifest = fit_candidates(
        texts, labels,
        target_agreement=args.target,
        teacher_model=teacher_model,
        prefer_embeddings=args.prefer_embeddings,
    )

    # Cost projection
    if manifest["n_traces"] > 0:
        coverage = manifest["coverage"]
        teacher_agreement = manifest["accepted_agreement"]
        print(f"\n  Cost Projection (10k queries/day):")
        print(f"    Without surrogate: 10,000 LLM classifier calls/day")
        print(f"    With surrogate:     {int(10000 * (1 - coverage))} LLM calls/day "
              f"({coverage:.0%} handled locally)")
        print(f"    Teacher agreement on handled traffic: {teacher_agreement:.1%}")


if __name__ == "__main__":
    main()
