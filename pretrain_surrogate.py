#!/usr/bin/env python3
"""
Pre-train the router proxy surrogate classifier using a local LLM as teacher.

Generates a synthetic training dataset by prompting a local model
(Qwen3.8-27B or similar) with diverse user queries, uses it to label them
into the configured categories, then fits the TF-IDF + sentence-transformer
surrogate pipeline using fit_surrogate.py's existing infrastructure.

This bootstraps the surrogate WITHOUT needing weeks of production trace
collection — you get a working model in ~10 minutes.

Usage:
    python3 pretrain_surrogate.py                          # uses config defaults
    python3 pretrain_surrogate.py --teacher http://10.8.6.3:8004/v1 --teacher-model qwen3.8-27b
    python3 pretrain_surrogate.py --num-samples 200         # generate 200 samples
    python3 pretrain_surrogate.py --skip-generation         # reuse existing traces

Requirements:
    - A local OpenAI-compatible LLM endpoint (teacher model)
    - scikit-learn (for fitting)
    - sentence-transformers (optional, for embedding-based surrogate)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import httpx
import yaml

# ── Config ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "router_config.yaml"

# Seed queries for each category — used to prompt the teacher model
# to generate more diverse examples
SEED_QUERIES = {
    "chat": [
        "hello", "how are you", "thanks", "what time is it", "good morning",
        "what is Docker", "explain REST API", "what does CRUD mean",
        "who created Python", "what's the weather", "hi there", "bye",
        "ok got it", "that's great", "haha that's funny", "谢谢", "你好",
        "早上好", "再见", "好的明白了"
    ],
    "code": [
        "write a Python function to sort a list", "debug this error",
        "refactor this class to use async/await", "implement a binary search tree",
        "create a decorator for caching", "write unit tests for this function",
        "fix this regex pattern", "how to handle exceptions in Python",
        "write a bash script to backup files", "implement OAuth2 flow",
        "code review this pull request", "optimize this SQL query",
        "write a TypeScript interface", "fix memory leak in C++",
        "implement a LRU cache", "write a Dockerfile for Node.js"
    ],
    "devops": [
        "deploy this to proxmox LXC", "configure nginx reverse proxy",
        "set up Tailscale VPN", "docker-compose for PostgreSQL",
        "configure firewall rules", "SSL certificate with certbot",
        "backup strategy for Proxmox", "systemd service for the app",
        "troubleshoot 502 Bad Gateway", "configure Caddy server",
        "set up GitHub Actions CI/CD", "SSH key management",
        "disk space full on pve-ssd", "OOM killer fired on container",
        "configure WireGuard tunnel", "restore from Proxmox snapshot"
    ],
    "research": [
        "compare K8s vs Docker Swarm", "research paper on transformer architectures",
        "analyze the benchmark results", "literature review on RAG systems",
        "survey of vector databases", "evaluate different LLM quantization methods",
        "compare REST vs GraphQL vs gRPC", "study on attention mechanisms",
        "benchmark llama.cpp vs vLLM", "academic paper on prompt engineering",
        "analyze market trends for AI", "compare self-hosted vs cloud LLM"
    ],
    "homeassistant": [
        "create HA automation for motion sensor", "configure Zigbee2MQTT",
        "set up ESPHome for ESP32", "MQTT broker configuration",
        "create a scene for movie night", "fix ZHA device pairing",
        "configure energy dashboard", "automate lights based on sunrise",
        "set up device tracker", "create binary sensor template",
        "HA blueprint for thermostat", "integrate Frigate with HA"
    ],
}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_classification_prompt(categories: dict, message: str) -> str:
    """Build the same prompt the router uses for classification."""
    cat_lines = []
    for name, c in categories.items():
        cat_lines.append(f"  {name} — {c.get('label', name)}")

    return f"""Classify the user's message into exactly one of these categories:
{chr(10).join(cat_lines)}

Reply with ONLY the category name, nothing else.

"{message}" →"""


def generate_samples(
    teacher_url: str,
    teacher_model: str,
    categories: dict,
    num_per_category: int = 40,
) -> list[dict]:
    """Generate labeled training samples by asking the teacher model to
    classify diverse seed queries."""
    samples = []

    # Build expanded query list: seed queries + variations
    all_queries = []
    for cat, seeds in SEED_QUERIES.items():
        for seed in seeds:
            all_queries.append((cat, seed))

        # Generate variations by asking the teacher to create more examples
        try:
            variation_prompt = (
                f"Generate {num_per_category - len(seeds)} diverse user messages "
                f"that would be classified as '{cat}' ({categories[cat].get('label', cat)}). "
                f"One per line, no numbering, no quotes. Just the raw message text."
            )
            resp = httpx.post(
                f"{teacher_url.rstrip('/')}/chat/completions",
                json={
                    "model": teacher_model,
                    "messages": [{"role": "user", "content": variation_prompt}],
                    "max_tokens": 1000,
                    "temperature": 0.7,
                },
                timeout=httpx.Timeout(60),
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"].get("content", "")
                # Parse one-per-line
                for line in content.strip().split("\n"):
                    line = line.strip().strip('"').strip("'").strip("0123456789.()[] ")
                    if line and len(line) > 3:
                        all_queries.append((cat, line))
        except Exception as exc:
            print(f"  Warning: could not generate variations for {cat}: {exc}")

    print(f"  Generated {len(all_queries)} total queries to label")

    # Now classify each query with the teacher model
    # (The seed queries already know their category, but we label generated ones)
    labeled = 0
    for intended_cat, query in all_queries:
        # For seed queries, trust the intended category
        # For generated queries, also trust the intended category
        # (the teacher was asked to generate examples FOR that category)
        samples.append({
            "text": query,
            "label": intended_cat,
            "source": "pretrain_seed",
        })
        labeled += 1

    # Also use the teacher model to cross-validate labels on a subset
    print("  Cross-validating labels with teacher model...")
    cross_val_count = 0
    for sample in random.sample(samples, min(50, len(samples))):
        prompt = build_classification_prompt(categories, sample["text"])
        try:
            resp = httpx.post(
                f"{teacher_url.rstrip('/')}/chat/completions",
                json={
                    "model": teacher_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                    "temperature": 0,
                },
                timeout=httpx.Timeout(30),
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"].get("content", "")
                reasoning = resp.json()["choices"][0]["message"].get("reasoning_content", "")
                # Parse the category from content or reasoning
                cat_names = list(categories.keys())
                result = (content or "").strip().lower()
                label = None
                for name in cat_names:
                    if name in result:
                        label = name
                        break
                if not label and reasoning:
                    for name in cat_names:
                        if name in reasoning.lower():
                            label = name
                            break
                if label and label != sample["label"]:
                    # Teacher disagrees — trust teacher's label
                    sample["label"] = label
                    sample["source"] = "pretrain_teacher_validated"
                cross_val_count += 1
        except Exception:
            pass

    print(f"  Cross-validated {cross_val_count} samples")
    print(f"  Final dataset: {len(samples)} labeled samples")

    # Label distribution
    dist = {}
    for s in samples:
        dist[s["label"]] = dist.get(s["label"], 0) + 1
    print(f"  Distribution: {dist}")

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train the router proxy surrogate classifier"
    )
    parser.add_argument(
        "--teacher", type=str, default="",
        help="Teacher LLM endpoint URL (default: from config classifier.base_url)"
    )
    parser.add_argument(
        "--teacher-model", type=str, default="",
        help="Teacher model name (default: from config classifier.model)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=40,
        help="Number of samples per category to generate (default: 40)"
    )
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Skip generation, use existing traces only"
    )
    args = parser.parse_args()

    cfg = load_config()
    categories = cfg.get("categories", {})

    if not categories:
        print("ERROR: No categories in config")
        sys.exit(1)

    teacher_url = args.teacher or cfg["classifier"]["base_url"]
    teacher_model = args.teacher_model or cfg["classifier"]["model"]

    print("\n  Pre-training Surrogate Classifier")
    print(f"  {'='*60}")
    print(f"  Teacher: {teacher_model} ({teacher_url})")
    print(f"  Categories: {list(categories.keys())}")
    print(f"  Samples per category: {args.num_samples}")
    print()

    # ── Generate or load training data ────────────────────────────────
    traces_dir = SCRIPT_DIR / "traces"

    if not args.skip_generation:
        print("  Generating synthetic training data...")
        samples = generate_samples(
            teacher_url, teacher_model, categories, args.num_samples
        )

        # Save as trace format for fit_surrogate.py to consume
        traces_dir.mkdir(parents=True, exist_ok=True)
        trace_file = traces_dir / "router-trace-pretrain.jsonl"
        with open(trace_file, "w", encoding="utf-8") as f:
            for s in samples:
                event = {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                    "event": "classify",
                    "session_key": f"pretrain_{random.randint(0, 999999)}",
                    "user_message_preview": s["text"][:120],
                    "classifier_result": s["label"],
                    "classifier_raw": s["label"],
                    "latency_ms": 0,
                    "tier": s["label"],
                    "model": teacher_model,
                    "is_first": True,
                }
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(f"  Saved {len(samples)} samples to {trace_file}")

    # ── Fit the surrogate ──────────────────────────────────────────────
    print("\n  Fitting surrogate model...")
    print(f"  {'='*60}")

    # Import and run fit_surrogate.py's fitting logic
    sys.path.insert(0, str(SCRIPT_DIR))
    from fit_surrogate import load_traces_all, extract_classifier_traces, prepare_dataset, fit_candidates

    script_dir = SCRIPT_DIR
    events = load_traces_all(script_dir)
    print(f"  Loaded {len(events)} total trace events")

    classifier_traces = extract_classifier_traces(events)
    print(f"  Found {len(classifier_traces)} classifier-source traces")

    if not classifier_traces:
        print("  ERROR: No classifier traces found. Generate data first.")
        sys.exit(1)

    texts, labels = prepare_dataset(classifier_traces)

    # Detect teacher model
    teacher = teacher_model

    fit_candidates(
        texts, labels,
        target_agreement=0.90,
        teacher_model=teacher,
        prefer_embeddings=True,
    )

    print("\n  Pre-training Complete!")
    print(f"  {'='*60}")
    print("  The surrogate is now available at .router/surrogate/")
    print("  Enable it in router_config.yaml: classifier.surrogate.enabled: true")
    print("  The router will use the surrogate for confident predictions")
    print("  and fall back to zero-shot / LLM only for uncertain ones.")


if __name__ == "__main__":
    main()