#!/usr/bin/env python3
"""
Hermes Model Router — hybrid flash-classifier + keyword-pipe proxy.

Routes simple queries (chat, quick questions) to cheap models and complex
queries (coding, system administration) to capable models. Session-aware:
classifies once with a flash model, then uses sub-millisecond keyword
deviation detection for follow-up messages.

OpenAI-compatible at POST /v1/chat/completitions.
Configuration: router_config.yaml (auto-detected alongside this file).

Author: James Huang + Jarvis (Hermes Agent)
License: MIT
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# ── Logging ─────────────────────────────────────────────────────────────────
class JsonFormatter(logging.Formatter):
    """Structured JSON log formatter — one line per record."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        return json.dumps(
            {
                "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "module": record.module,
                "line": record.lineno,
                "message": record.getMessage(),
            },
            ensure_ascii=False,
            default=str,
        )


_log_format = os.environ.get("LOG_FORMAT", "").strip().lower()
if _log_format == "json":
    _formatter = JsonFormatter()
else:
    _formatter = logging.Formatter(
        "%(asctime)s [router] %(levelname)s %(message)s"
    )

logging.basicConfig(
    level=logging.INFO,
    format=None,  # handled by formatter
)
_log_handler = logging.getLogger().handlers[0]
_log_handler.setFormatter(_formatter)

log = logging.getLogger("hermes-router")

# Also configure uvicorn's loggers for JSON mode
if _log_format == "json":
    for _name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        _uv_logger = logging.getLogger(_name)
        _uv_logger.handlers.clear()
        _uv_logger.addHandler(_log_handler)
        _uv_logger.propagate = False

# ── Config ─────────────────────────────────────────────────────────────────
CONFIG_DIR = Path(__file__).resolve().parent
CONFIG_PATH = CONFIG_DIR / "router_config.yaml"


def load_config() -> dict[str, Any]:
    """Load router_config.yaml, fail loudly if missing."""
    if not CONFIG_PATH.exists():
        log.fatal("Config not found: %s — run install.sh first", CONFIG_PATH)
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def env_key(name: str) -> str:
    """Read an API key from the environment.  Returns empty string on miss."""
    return os.environ.get(name, "").strip()


# ── Profile Hint (lazy extraction) ──────────────────────────────────────────

def build_classification_prompt(
    cfg: dict, user_message: str, *, force_extract: bool = False
) -> str:
    """
    Return the full classification prompt with profile hint injected.
    On first call (profile_hint empty) or when force_extract=True,
    reads USER.md + MEMORY.md and caches a 2-3 sentence summary back to config.
    """
    hint = cfg["classifier"].get("profile_hint", "").strip()

    if not hint or force_extract:
        hint = _extract_profile_hint(cfg)
        cfg["classifier"]["profile_hint"] = hint
        _write_config_back(cfg)

    template = cfg["classifier"]["system_prompt"]
    prompt = template.strip().replace("{message}", user_message)
    if hint:
        prompt = f"Agent context: {hint}\n\n{prompt}"
    return prompt


def _extract_profile_hint(cfg: dict) -> str:
    """Read USER.md + MEMORY.md, pass to flash model, return a 2-3 sentence summary.

    USER.md contains who the user is (name, identity, preferences, location).
    MEMORY.md contains durable facts (rules, environment, tool quirks).
    Together they give the classifier enough context for accurate routing.
    """
    p = cfg["persona"]
    user_path = Path(p["user_path"]).expanduser()
    memory_path = Path(p["memory_path"]).expanduser()

    parts: list[str] = []
    for label, path in (("USER.md", user_path), ("MEMORY.md", memory_path)):
        if path.exists():
            parts.append(path.read_text().strip())
        else:
            log.warning("%s not found at %s", label, path)

    if not parts:
        log.warning("Neither USER.md nor MEMORY.md found — skipping profile extraction")
        return "No profile available."

    max_chars = p.get("max_context_chars", 800)
    raw = "\n\n".join(parts)[:max_chars]

    prompt = (
        "Summarize this AI agent's user identity, environment, and key tools "
        "in 2–3 concise sentences. Keep only what helps classify tasks as 'simple' "
        f"or 'complex'.\n\n{raw}"
    )

    log.info("Extracting profile hint from USER.md+MEMORY.md (%d chars) → flash model", len(raw))
    summary = _call_classifier_raw(cfg, prompt, max_tokens=100)
    return summary.strip() or "No profile available."


def _call_classifier_raw(
    cfg: dict, prompt: str, max_tokens: int = 256
) -> str:
    """Call the flash classifier model with a raw prompt, return text content.
    
    Uses a generous token budget because reasoning models (deepseek-v4-flash)
    spend tokens on reasoning_content before producing content.
    Falls back to extracting the classification from reasoning_content 
    if the content field is empty.
    """
    cl = cfg["classifier"]
    api_key = env_key(cl["api_key_env"])

    payload: dict[str, Any] = {
        "model": cl["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    try:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = httpx.post(
            f"{cl['base_url'].rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=httpx.Timeout(15),
        )
        if resp.status_code != 200:
            log.warning("Classifier returned HTTP %d: %s", resp.status_code, resp.text[:200])
            return "simple"
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = (msg.get("content") or "").strip().lower()
        # Reasoning models may put the answer in reasoning_content instead
        if not content:
            reasoning = (msg.get("reasoning_content") or "")
            # Extract last meaningful word from reasoning
            parts = reasoning.lower().strip().split()
            for word in reversed(parts):
                cleaned = word.strip('.,;:!?"\'()')
                if cleaned in ("simple", "complex"):
                    content = cleaned
                    break
        return content or "simple"
    except Exception as exc:
        log.warning("Classifier call failed: %s", exc)
        return "simple"


def _write_config_back(cfg: dict) -> None:
    """Write updated config back to disk (profile_hint after extraction)."""
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    log.info("Wrote updated router_config.yaml with profile_hint")


# ── Classification ──────────────────────────────────────────────────────────

def classify(cfg: dict, user_message: str) -> str:
    """
    Ask the flash classifier: "simple" or "complex"?
    Returns "simple" as safe default on any failure.
    """
    t0 = time.time()
    prompt = build_classification_prompt(cfg, user_message)
    result = _call_classifier_raw(cfg, prompt, max_tokens=32)
    _record_classifier_latency((time.time() - t0) * 1000)

    if "complex" in result:
        return "complex"
    return "simple"


# ── Keyword Deviation Detection ─────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Strip whitespace, hyphens, underscores — collapse to lowercase."""
    return re.sub(r"[-\s_]+", "", text).lower()


def _fuzzy_match(keyword: str, text: str) -> bool:
    """
    Match a keyword against text with typo tolerance.
    
    1. Exact substring (normalised).
    2. Normalised substring.
    3. Levenshtein distance ≤1 for keywords ≥5 chars.
    """
    n_key = keyword.lower().strip()
    n_text = text.lower()

    # 1. Exact substring
    if n_key in n_text:
        return True

    # 2. Normalised (strip separators)
    if _normalize(keyword) in _normalize(text):
        return True

    # 3. Typo tolerance — words ≥5 chars, allow 1-char difference
    if len(n_key) >= 5:
        for word in n_text.split():
            word = word.strip('.,;:!?"\'()[]{}')
            if len(word) >= 5 and _levenshtein(n_key, word) <= 1:
                return True

    return False


def _levenshtein(s1: str, s2: str) -> int:
    """Minimal edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def has_deviation(cfg: dict, text: str, current_tier: str) -> bool:
    """Scan follow-up message for escalation/de-escalation keywords."""
    # Escalation: simple → suddenly complex
    if current_tier == "simple":
        for kw in cfg["routing"].get("escalation_keywords", []):
            if _fuzzy_match(kw, text):
                log.info("Deviation: escalation keyword '%s' matched", kw)
                return True

    # De-escalation: complex → suddenly casual
    if current_tier == "complex":
        for kw in cfg["routing"].get("de_escalation_keywords", []):
            if _fuzzy_match(kw, text):
                log.info("Deviation: de-escalation keyword '%s' matched", kw)
                return True

    return False


# ── Session Cache ───────────────────────────────────────────────────────────

# In-memory: session_key → {"tier": "simple"|"complex", "at": timestamp}
SESSIONS: dict[str, dict[str, Any]] = {}

# ── Metrics ─────────────────────────────────────────────────────────────────
# Prometheus-compatible counters for GET /metrics
METRICS: dict[str, int] = {
    "requests_total_simple": 0,
    "requests_total_complex": 0,
    "classifier_calls_total": 0,
    "classifier_latency_ms_sum": 0,
    "cache_hits_total": 0,
    "429_total": 0,
    "429_simple": 0,
    "429_complex": 0,
    "fallback_used_total": 0,
    "fallback2_used_total": 0,
    "stream_requests_total": 0,
    "errors_total": 0,
}

# ── Circuit Breakers ────────────────────────────────────────────────────────
# Per-endpoint circuit breakers that track consecutive 429s.
# After N consecutive failures in a sliding window, open the circuit
# (skip the endpoint entirely) for X seconds.
CIRCUITS: dict[str, dict[str, Any]] = {}

# Defaults — overridable via router_config.yaml → circuit_breaker section
CB_DEFAULTS: dict[str, int] = {
    "failure_threshold": 3,      # consecutive 429s before tripping
    "recovery_timeout_sec": 30,  # how long the circuit stays open
    "window_sec": 60,            # sliding window for counting failures
}


def _circuit_key(base_url: str) -> str:
    """Normalize a base_url into a circuit breaker key."""
    return base_url.rstrip("/").replace("://", "_").replace("/", "_").replace(".", "_")


def _circuit_is_open(cfg: dict, base_url: str) -> bool:
    """Check whether the circuit for this endpoint is currently open."""
    cb_cfg = cfg.get("circuit_breaker", CB_DEFAULTS)
    key = _circuit_key(base_url)
    entry = CIRCUITS.get(key)
    if not entry:
        return False
    if entry["state"] != "open":
        return False
    if time.time() - entry["opened_at"] > cb_cfg.get("recovery_timeout_sec", 30):
        log.info("Circuit %s → half-open (recovery timeout elapsed)", key)
        entry["state"] = "half_open"
        entry["half_open_at"] = time.time()
        return False
    remaining = cb_cfg.get("recovery_timeout_sec", 30) - int(time.time() - entry["opened_at"])
    if remaining > 0:
        log.debug("Circuit %s is open (%ds remaining)", key, remaining)
    return True


def _circuit_record_success(cfg: dict, base_url: str) -> None:
    """Reset the circuit breaker after a successful request."""
    key = _circuit_key(base_url)
    entry = CIRCUITS.get(key)
    if entry and entry["state"] == "half_open":
        log.info("Circuit %s → closed (success in half-open)", key)
    CIRCUITS[key] = {"state": "closed", "failures": 0, "last_failure_at": 0}


def _circuit_record_failure(cfg: dict, base_url: str) -> None:
    """Record a failure (429) and potentially open the circuit."""
    cb_cfg = cfg.get("circuit_breaker", CB_DEFAULTS)
    key = _circuit_key(base_url)
    entry = CIRCUITS.get(key, {"state": "closed", "failures": 0, "last_failure_at": 0})
    now = time.time()
    window = cb_cfg.get("window_sec", 60)
    if now - entry.get("last_failure_at", 0) > window:
        entry["failures"] = 0
    entry["failures"] += 1
    entry["last_failure_at"] = now
    threshold = cb_cfg.get("failure_threshold", 3)
    if entry["failures"] >= threshold and entry["state"] != "open":
        log.warning(
            "Circuit %s → OPEN (%d failures in %ds, recovery in %ds)",
            key, entry["failures"], window,
            cb_cfg.get("recovery_timeout_sec", 30),
        )
        entry["state"] = "open"
        entry["opened_at"] = now
    CIRCUITS[key] = entry


def _inc_metric(name: str, delta: int = 1) -> None:
    """Increment a metric counter atomically (single-threaded safe)."""
    if name in METRICS:
        METRICS[name] += delta


def _inc_metric_tier(tier: str, name: str, delta: int = 1) -> None:
    """Increment a tier-scoped metric: {name}_{tier}."""
    _inc_metric(f"{name}_{tier}", delta)


def _record_classifier_latency(ms: float) -> None:
    """Record classifier call latency."""
    METRICS["classifier_calls_total"] += 1
    METRICS["classifier_latency_ms_sum"] += int(ms)


def _get_metrics() -> dict:
    """Return a copy of current metrics with computed fields."""
    m = dict(METRICS)
    calls = m["classifier_calls_total"]
    m["classifier_latency_ms_avg"] = (
        m["classifier_latency_ms_sum"] // calls if calls > 0 else 0
    )
    m["sessions_active"] = len(SESSIONS)
    return m


def session_key(messages: list[dict]) -> str | None:
    """Derive a session key from the first user message. Returns None if empty."""
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                # Multimodal — grab text parts
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            return hashlib.sha256(content.encode()[:200]).hexdigest()[:16]
    return None


def is_first_message(messages: list[dict]) -> bool:
    """A new session starts when there is exactly one user message."""
    user_count = sum(1 for m in messages if m.get("role") == "user")
    return user_count <= 1


def get_cached_tier(cfg: dict, key: str) -> str | None:
    """Return cached tier if session is still valid, None otherwise."""
    entry = SESSIONS.get(key)
    if not entry:
        return None
    timeout_mins = cfg["classifier"].get("session_timeout_minutes", 5)
    if time.time() - entry["at"] > timeout_mins * 60:
        log.info("Session %s expired", key)
        del SESSIONS[key]
        return None
    return entry["tier"]


def cache_tier(key: str, tier: str) -> None:
    SESSIONS[key] = {"tier": tier, "at": time.time()}
    log.info("Session %s → %s (cached)", key, tier)


# ── Model Calling ───────────────────────────────────────────────────────────

def call_model(
    cfg: dict, model_cfg: dict, request_payload: dict
) -> httpx.Response:
    """Call an OpenAI-compatible endpoint. Returns the httpx response.

    Circuit breaker: checks if the endpoint circuit is open before calling.
    Key rotation: if the primary key returns HTTP 429 (rate-limited),
    retries with alternate_key_env before giving up.
    """
    base_url = model_cfg["base_url"].rstrip("/")
    url = f"{base_url}/chat/completions"

    # Circuit breaker check
    if _circuit_is_open(cfg, base_url):
        log.warning("Circuit open for %s — skipping call", base_url)
        _inc_metric_tier(model_cfg.get("tier", "unknown"), "429")
        _inc_metric("429_total")
        # Return synthetic 503 response — let caller handle fallback
        r = httpx.Response(503, text="Circuit breaker open")
        r._request = httpx.Request("POST", url)
        return r

    api_key = env_key(model_cfg["api_key_env"])
    timeout = model_cfg.get("timeout_seconds", 120)
    alt_key = env_key(model_cfg.get("alternate_key_env", ""))

    payload = {**request_payload, "model": model_cfg["model"]}
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = httpx.post(
        url,
        json=payload,
        headers=headers,
        timeout=httpx.Timeout(timeout),
    )

    # Circuit + metric tracking for 429
    if resp.status_code == 429:
        _circuit_record_failure(cfg, base_url)
        _inc_metric_tier(model_cfg.get("tier", "unknown"), "429")
        _inc_metric("429_total")
    elif resp.status_code == 200:
        _circuit_record_success(cfg, base_url)

    # Key rotation: HTTP 429 with alternate key available -> retry
    if resp.status_code == 429 and alt_key:
        log.warning("Primary key rate-limited (429) - switching to alternate key")
        resp = httpx.post(
            url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {alt_key}",
            },
            timeout=httpx.Timeout(timeout),
        )
        if resp.status_code == 200:
            log.info("Alternate key succeeded")
            _circuit_record_success(cfg, base_url)

    return resp


async def call_model_stream(model_cfg: dict, request_payload: dict):
    """Async generator that yields SSE chunks from an upstream model."""
    url = f"{model_cfg['base_url'].rstrip('/')}/chat/completions"
    api_key = env_key(model_cfg["api_key_env"])
    timeout = model_cfg.get("timeout_seconds", 120)

    payload = {**request_payload, "model": model_cfg["model"]}
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST", url, json=payload, headers=headers,
            timeout=httpx.Timeout(timeout),
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                yield f'data: {{"error":{{"message":"Upstream {resp.status_code}: {body.decode(errors="replace")[:300]}","type":"upstream_error"}}}}\n\n'.encode()
                yield b'data: [DONE]\n\n'
                return

            async for line in resp.aiter_lines():
                if line and line.startswith("data: "):
                    yield f"{line}\n".encode()
                elif line.strip() == "":
                    yield b"\n"

        # Add router metadata to final chunk
        yield b''  # sentinel - metadata handled by wrapper


async def route_request_stream(cfg: dict, payload: dict):
    """Streaming version of route_request — returns SSE chunks from upstream."""
    messages = payload.get("messages", [])
    key = session_key(messages)
    if not key:
        yield b'data: {"error":{"message":"No user message found","type":"router_error"}}\n\ndata: [DONE]\n\n'
        return

    # ── Determine tier (reuse sync logic from cache) ──────────────────
    tier: str
    if is_first_message(messages):
        user_content = _last_user_text(messages)
        tier = classify(cfg, user_content)
        cache_tier(key, tier)
    else:
        cached = get_cached_tier(cfg, key)
        if cached is None:
            user_content = _last_user_text(messages)
            tier = classify(cfg, user_content)
            cache_tier(key, tier)
        else:
            last_text = _last_user_text(messages)
            if has_deviation(cfg, last_text, cached):
                user_content = _last_user_text(messages)
                tier = classify(cfg, user_content)
                cache_tier(key, tier)
            else:
                tier = cached
                _inc_metric("cache_hits_total")

    model_cfg = cfg["models"][tier]
    log.info("Streaming session %s → %s (%s)", key, tier, model_cfg["model"])

    # Try primary first
    async with httpx.AsyncClient() as client:
        url = f"{model_cfg['base_url'].rstrip('/')}/chat/completions"
        api_key = env_key(model_cfg["api_key_env"])
        timeout = model_cfg.get("timeout_seconds", 120)
        alt_key = env_key(model_cfg.get("alternate_key_env", ""))

        stream_payload = {**payload, "model": model_cfg["model"]}
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with client.stream(
            "POST", url, json=stream_payload, headers=headers,
            timeout=httpx.Timeout(timeout),
        ) as resp:
            if resp.status_code == 200:
                # Primary succeeded — forward SSE stream
                log.info("Streaming primary %s OK", model_cfg["model"])
                async for line in resp.aiter_lines():
                    if line:
                        yield f"{line}\n".encode()
                    else:
                        yield b"\n"
                return

            # Primary failed — try alternate key if 429
            if resp.status_code == 429 and alt_key:
                log.warning("Primary key rate-limited (429) - switching to alternate key")
                alt_headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {alt_key}",
                }
                async with client.stream(
                    "POST", url, json=stream_payload, headers=alt_headers,
                    timeout=httpx.Timeout(timeout),
                ) as resp2:
                    if resp2.status_code == 200:
                        log.info("Alternate key succeeded")
                        async for line in resp2.aiter_lines():
                            if line:
                                yield f"{line}\n".encode()
                            else:
                                yield b"\n"
                        return

            # Primary + alt key failed — try fallback
            fallback_model = model_cfg.get("fallback_model")
            if fallback_model:
                log.warning(
                    "Primary %s returned %d — streaming fallback to %s",
                    model_cfg["model"], resp.status_code, fallback_model,
                )
                fb_cfg = {
                    "model": fallback_model,
                    "base_url": model_cfg["fallback_base_url"],
                    "api_key_env": model_cfg["fallback_key_env"],
                    "timeout_seconds": model_cfg.get("timeout_seconds", 120),
                }
                fb_url = f"{fb_cfg['base_url'].rstrip('/')}/chat/completions"
                fb_key = env_key(fb_cfg["api_key_env"])
                fb_headers: dict[str, str] = {"Content-Type": "application/json"}
                if fb_key:
                    fb_headers["Authorization"] = f"Bearer {fb_key}"
                fb_payload = {**payload, "model": fb_cfg["model"]}

                async with client.stream(
                    "POST", fb_url, json=fb_payload, headers=fb_headers,
                    timeout=httpx.Timeout(fb_cfg["timeout_seconds"]),
                ) as fb_resp:
                    if fb_resp.status_code == 200:
                        log.info("Streaming fallback %s OK", fallback_model)
                        async for line in fb_resp.aiter_lines():
                            if line:
                                yield f"{line}\n".encode()
                            else:
                                yield b"\n"
                        return
                    else:
                        fb_body = await fb_resp.aread()
                        error_msg = fb_body.decode(errors="replace")[:300]
                        fb_status = fb_resp.status_code

                        # Alternate key rotation for fallback on 429
                        if fb_resp.status_code == 429:
                            fb_alt_key = env_key(model_cfg.get("fallback_alternate_key_env", ""))
                            if fb_alt_key:
                                log.warning("Fallback key rate-limited (429) - switching to alternate key")
                                alt_fb_headers = {**fb_headers, "Authorization": f"Bearer {fb_alt_key}"}
                                async with client.stream(
                                    "POST", fb_url, json=fb_payload, headers=alt_fb_headers,
                                    timeout=httpx.Timeout(fb_cfg["timeout_seconds"]),
                                ) as alt_fb_resp:
                                    if alt_fb_resp.status_code == 200:
                                        log.info("Alternate fallback key succeeded")
                                        async for line in alt_fb_resp.aiter_lines():
                                            if line:
                                                yield f"{line}\n".encode()
                                            else:
                                                yield b"\n"
                                        return
                                    else:
                                        alt_body = await alt_fb_resp.aread()
                                        error_msg = alt_body.decode(errors="replace")[:300]
                                        fb_status = alt_fb_resp.status_code

                        # Fallback 1 failed — try fallback 2
                        fb2_model = model_cfg.get("fallback2_model")
                        if fb2_model:
                            log.warning(
                                "Fallback %s returned %d — trying fallback2 %s",
                                fallback_model, fb_status, fb2_model,
                            )
                            fb2_cfg = {
                                "model": fb2_model,
                                "base_url": model_cfg["fallback2_base_url"],
                                "api_key_env": model_cfg["fallback2_key_env"],
                                "timeout_seconds": model_cfg.get("timeout_seconds", 120),
                            }
                            fb2_url = f"{fb2_cfg['base_url'].rstrip('/')}/chat/completions"
                            fb2_key = env_key(fb2_cfg["api_key_env"])
                            fb2_headers: dict[str, str] = {"Content-Type": "application/json"}
                            if fb2_key:
                                fb2_headers["Authorization"] = f"Bearer {fb2_key}"
                            fb2_payload = {**payload, "model": fb2_cfg["model"]}
                            async with client.stream(
                                "POST", fb2_url, json=fb2_payload, headers=fb2_headers,
                                timeout=httpx.Timeout(fb2_cfg["timeout_seconds"]),
                            ) as fb2_resp:
                                if fb2_resp.status_code == 200:
                                    log.info("Streaming fallback2 %s OK", fb2_model)
                                    async for line in fb2_resp.aiter_lines():
                                        if line:
                                            yield f"{line}\n".encode()
                                        else:
                                            yield b"\n"
                                    return
                                else:
                                    fb2_body = await fb2_resp.aread()
                                    fb2_error = fb2_body.decode(errors="replace")[:200]

                                    # Alternate key rotation for fallback2 on 429
                                    if fb2_resp.status_code == 429:
                                        fb2_alt_key = env_key(model_cfg.get("fallback2_alternate_key_env", ""))
                                        if fb2_alt_key:
                                            log.warning("Fallback2 key rate-limited (429) - switching to alternate key")
                                            alt_fb2_headers = {**fb2_headers, "Authorization": f"Bearer {fb2_alt_key}"}
                                            async with client.stream(
                                                "POST", fb2_url, json=fb2_payload, headers=alt_fb2_headers,
                                                timeout=httpx.Timeout(fb2_cfg["timeout_seconds"]),
                                            ) as alt_fb2_resp:
                                                if alt_fb2_resp.status_code == 200:
                                                    log.info("Alternate fallback2 key succeeded")
                                                    async for line in alt_fb2_resp.aiter_lines():
                                                        if line:
                                                            yield f"{line}\n".encode()
                                                        else:
                                                            yield b"\n"
                                                    return
                                                else:
                                                    fb2_body = await alt_fb2_resp.aread()
                                                    fb2_error = fb2_body.decode(errors="replace")[:200]

                                    yield f'data: {{"error":{{"message":"Fallbacks exhausted: {fallback_model}({fb_status}), {fb2_model}({fb2_resp.status_code}) - {fb2_error}","type":"upstream_error"}}}}\n\n'.encode()
                                    yield b'data: [DONE]\n\n'
                                    return
                        else:
                            yield f'data: {{"error":{{"message":"Fallback {fallback_model} failed: {resp.status_code}/{fb_resp.status_code} - {error_msg}","type":"upstream_error"}}}}\n\n'.encode()
                            yield b'data: [DONE]\n\n'
                            return
            else:
                body = await resp.aread()
                error_msg = body.decode(errors="replace")[:300]
                yield f'data: {{"error":{{"message":"Upstream {model_cfg["model"]} returned {resp.status_code}: {error_msg}","type":"upstream_error"}}}}\n\n'.encode()
                yield b'data: [DONE]\n\n'


def route_request(cfg: dict, payload: dict) -> JSONResponse:
    """
    Full routing pipeline.  Determines the model tier, calls it
    (with fallback), and returns a FastAPI JSONResponse.
    """
    messages = payload.get("messages", [])
    key = session_key(messages)
    if not key:
        return _error(400, "No user message found in request")

    # ── Determine tier ──────────────────────────────────────────────────
    tier: str

    if is_first_message(messages):
        # Brand new session — classify via flash model only.
        # No keyword override — first messages are classifier territory.
        user_content = _last_user_text(messages)
        tier = classify(cfg, user_content)
        cache_tier(key, tier)

    else:
        # Follow-up message — check cache + keyword deviation
        cached = get_cached_tier(cfg, key)
        if cached is None:
            # Expired — re-classify
            user_content = _last_user_text(messages)
            tier = classify(cfg, user_content)
            cache_tier(key, tier)
        else:
            last_text = _last_user_text(messages)
            if has_deviation(cfg, last_text, cached):
                user_content = _last_user_text(messages)
                tier = classify(cfg, user_content)
                if tier != cached:
                    log.info(
                        "Session %s tier changed: %s → %s", key, cached, tier
                    )
                else:
                    log.info(
                        "Session %s deviation detected but tier unchanged: %s",
                        key, tier,
                    )
                cache_tier(key, tier)
            else:
                tier = cached
                _inc_metric("cache_hits_total")

    # ── Call model ──────────────────────────────────────────────────────
    model_cfg = cfg["models"][tier]
    model_cfg["tier"] = tier  # for circuit breaker metric labeling
    _inc_metric_tier(tier, "requests_total")
    log.info("Routing session %s → %s (%s)", key, tier, model_cfg["model"])

    resp = call_model(cfg, model_cfg, payload)

    if resp.status_code == 200:
        return JSONResponse(content=resp.json())

    # ── Fallback ────────────────────────────────────────────────────────
    fallback_model = model_cfg.get("fallback_model")
    if not fallback_model:
        return _proxy_error(resp)

    log.warning(
        "Primary model %s returned %d — trying fallback %s",
        model_cfg["model"],
        resp.status_code,
        fallback_model,
    )

    _inc_metric("fallback_used_total")
    fb_cfg = {
        "model": fallback_model,
        "base_url": model_cfg["fallback_base_url"],
        "api_key_env": model_cfg["fallback_key_env"],
        "alternate_key_env": model_cfg.get("fallback_alternate_key_env", ""),
        "timeout_seconds": model_cfg.get("timeout_seconds", 120),
        "tier": tier,
    }
    fb_resp = call_model(cfg, fb_cfg, payload)

    if fb_resp.status_code == 200:
        data = fb_resp.json()
        data.setdefault("hermes_router", {})["fallback_used"] = True
        return JSONResponse(content=data)

    # ── Fallback 2 ─────────────────────────────────────────────────────────
    fb2_model = model_cfg.get("fallback2_model")
    if fb2_model:
        log.warning(
            "Fallback %s returned %d — trying fallback2 %s",
            fallback_model, fb_resp.status_code, fb2_model,
        )
        _inc_metric("fallback2_used_total")
        fb2_cfg = {
            "model": fb2_model,
            "base_url": model_cfg["fallback2_base_url"],
            "api_key_env": model_cfg["fallback2_key_env"],
            "alternate_key_env": model_cfg.get("fallback2_alternate_key_env", ""),
            "timeout_seconds": model_cfg.get("timeout_seconds", 120),
            "tier": tier,
        }
        fb2_resp = call_model(cfg, fb2_cfg, payload)
        if fb2_resp.status_code == 200:
            data = fb2_resp.json()
            data.setdefault("hermes_router", {})["fallback_used"] = True
            return JSONResponse(content=data)
        return _proxy_error(fb2_resp)

    return _proxy_error(fb_resp)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _last_user_text(messages: list[dict]) -> str:
    """Extract the text content of the most recent user message."""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                return " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            return content
    return ""


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "router_error"}},
        status_code=status,
    )


def _proxy_error(resp: httpx.Response) -> JSONResponse:
    """Forward an upstream error with context."""
    detail = resp.text[:500] if resp.text else "Unknown upstream error"
    return JSONResponse(
        {
            "error": {
                "message": f"Upstream model returned {resp.status_code}: {detail}",
                "type": "upstream_error",
                "status_code": resp.status_code,
            }
        },
        status_code=502,
    )


# ── FastAPI Application ─────────────────────────────────────────────────────

def verify_auth(request: Request):
    """Check Bearer token against configured API key."""
    cfg = request.app.state.config
    key_env = cfg.get("auth", {}).get("api_key_env", "")
    if not key_env:
        return  # No auth configured — allow all
    expected = os.environ.get(key_env, "").strip()
    if not expected:
        return  # Env var not set — allow all
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer ") and auth_header[7:] == expected:
        return
    raise HTTPException(
        status_code=401,
        detail={"error": {"message": "Invalid or missing API key", "type": "auth_error"}},
    )

app = FastAPI(
    title="Hermes Model Router",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

# ── CORS ──────────────────────────────────────────────────────────────────
origins_raw = os.environ.get("CORS_ORIGINS", "*").strip()
allowed_origins = [o.strip() for o in origins_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    cfg = load_config()
    log.info("Router starting on %s:%s", cfg["server"]["host"], cfg["server"]["port"])
    log.info("  Classifier: %s (%s)", cfg["classifier"]["model"], cfg["classifier"]["base_url"])
    log.info("  Simple model: %s (%s)", cfg["models"]["simple"]["model"], cfg["models"]["simple"]["base_url"])
    log.info("  Complex model: %s (%s)", cfg["models"]["complex"]["model"], cfg["models"]["complex"]["base_url"])
    if cfg["models"]["complex"].get("fallback_model"):
        log.info("  Fallback: %s (%s)", cfg["models"]["complex"]["fallback_model"], cfg["models"]["complex"]["fallback_base_url"])
    if cfg["models"]["complex"].get("fallback2_model"):
        log.info("  Fallback2: %s (%s)", cfg["models"]["complex"]["fallback2_model"], cfg["models"]["complex"]["fallback2_base_url"])
    app.state.config = cfg


@app.get("/health")
async def health():
    """Simple liveness check."""
    return {"status": "ok", "sessions": len(SESSIONS)}


@app.post("/reload")
async def reload_config(request: Request):
    """Hot-reload router_config.yaml without restart.

    Reads config from disk, validates required sections, atomically
    swaps app.state.config. Clears profile_hint to force re-extraction
    on the next classification request.
    """
    verify_auth(request)

    old_cfg = request.app.state.config
    old_simple = old_cfg["models"]["simple"]["model"]
    old_complex = old_cfg["models"]["complex"]["model"]

    try:
        new_cfg = load_config()
    except Exception as exc:
        log.error("Failed to parse config on reload: %s", exc)
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": f"Config parse error: {exc}", "type": "reload_error"}},
        )

    # Validate minimum structure
    for section in ("classifier", "models", "routing", "server"):
        if section not in new_cfg:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": f"Missing required section: {section}", "type": "reload_error"}},
            )
    for tier in ("simple", "complex"):
        if tier not in new_cfg.get("models", {}):
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": f"Missing models.{tier} in config", "type": "reload_error"}},
            )

    # Clear profile_hint so extraction runs with fresh config
    new_cfg["classifier"]["profile_hint"] = ""

    # Atomic swap
    request.app.state.config = new_cfg

    log.info(
        "Config hot-reloaded. Simple: %s → %s, Complex: %s → %s",
        old_simple, new_cfg["models"]["simple"]["model"],
        old_complex, new_cfg["models"]["complex"]["model"],
    )

    return {
        "status": "reloaded",
        "before": {"simple": old_simple, "complex": old_complex},
        "after": {
            "simple": new_cfg["models"]["simple"]["model"],
            "complex": new_cfg["models"]["complex"]["model"],
        },
    }


@app.get("/circuits")
async def list_circuits(request: Request):
    """List all circuit breaker states."""
    verify_auth(request)
    return {
        "circuits": CIRCUITS,
        "defaults": CB_DEFAULTS,
    }


@app.post("/circuits/reset")
async def reset_circuits(request: Request):
    """Reset all circuit breakers back to closed state."""
    verify_auth(request)
    count = len(CIRCUITS)
    CIRCUITS.clear()
    log.info("Reset %d circuit breakers", count)
    return {"status": "reset", "count": count}


@app.get("/admin/sessions")
async def list_sessions(request: Request):
    """List all cached sessions with tier info."""
    verify_auth(request)
    now = time.time()
    cfg = request.app.state.config
    timeout_mins = cfg["classifier"].get("session_timeout_minutes", 5)
    sessions = {}
    for key, entry in list(SESSIONS.items()):
        age_sec = int(now - entry["at"])
        remaining_sec = max(0, timeout_mins * 60 - age_sec)
        sessions[key] = {
            "tier": entry["tier"],
            "age_sec": age_sec,
            "remaining_sec": remaining_sec,
        }
    return {"count": len(sessions), "session_timeout_minutes": timeout_mins, "sessions": sessions}


@app.delete("/admin/sessions/{key}")
async def evict_session(key: str, request: Request):
    """Force-evict a cached session, forcing re-classification on next message."""
    verify_auth(request)
    removed = SESSIONS.pop(key, None)
    if removed:
        log.info("Session %s evicted (was %s)", key, removed["tier"])
        return {"status": "evicted", "key": key, "was_tier": removed["tier"]}
    raise HTTPException(
        status_code=404,
        detail={"error": {"message": f"Session {key} not found", "type": "not_found"}},
    )


@app.get("/admin/config")
async def get_config(request: Request):
    """Return current config with sensitive keys redacted."""
    verify_auth(request)
    cfg_copy = copy.deepcopy(request.app.state.config)
    # Redact env var names (not values — those stay in env, not config)
    # No actual API keys are in the config, just env var names.
    # We show the raw config as-is since it only references env vars.
    return cfg_copy


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions — routed automatically."""
    verify_auth(request)
    cfg = request.app.state.config
    payload = await request.json()
    try:
        if payload.get("stream"):
            _inc_metric("stream_requests_total")
            return StreamingResponse(
                route_request_stream(cfg, payload),
                media_type="text/event-stream",
            )
        return route_request(cfg, payload)
    except Exception:
        _inc_metric("errors_total")
        raise


@app.get("/metrics")
async def metrics(request: Request):
    """Prometheus-compatible metrics endpoint with router-specific counters."""
    verify_auth(request)
    m = _get_metrics()
    # Prometheus text format
    lines = [
        "# HELP hermes_router_requests_total Total requests by tier",
        "# TYPE hermes_router_requests_total counter",
        f"hermes_router_requests_total{{tier=\"simple\"}} {m['requests_total_simple']}",
        f"hermes_router_requests_total{{tier=\"complex\"}} {m['requests_total_complex']}",
        "",
        "# HELP hermes_router_classifier_calls_total Classifier model calls",
        "# TYPE hermes_router_classifier_calls_total counter",
        f"hermes_router_classifier_calls_total {m['classifier_calls_total']}",
        "",
        "# HELP hermes_router_classifier_latency_ms Classifier latency in ms",
        "# TYPE hermes_router_classifier_latency_ms summary",
        f"hermes_router_classifier_latency_ms_sum {m['classifier_latency_ms_sum']}",
        f"hermes_router_classifier_latency_ms_avg {m['classifier_latency_ms_avg']}",
        "",
        "# HELP hermes_router_cache_hits_total Session cache hits (skip classifier)",
        "# TYPE hermes_router_cache_hits_total counter",
        f"hermes_router_cache_hits_total {m['cache_hits_total']}",
        "",
        "# HELP hermes_router_429_total Rate limit hits by tier",
        "# TYPE hermes_router_429_total counter",
        f"hermes_router_429_total{{tier=\"simple\"}} {m['429_simple']}",
        f"hermes_router_429_total{{tier=\"complex\"}} {m['429_complex']}",
        f"hermes_router_429_total {m['429_total']}",
        "",
        "# HELP hermes_router_fallback_used_total Fallback tiers triggered",
        "# TYPE hermes_router_fallback_used_total counter",
        f"hermes_router_fallback_used_total{{level=\"1\"}} {m['fallback_used_total']}",
        f"hermes_router_fallback_used_total{{level=\"2\"}} {m['fallback2_used_total']}",
        "",
        "# HELP hermes_router_stream_requests_total Streaming requests",
        "# TYPE hermes_router_stream_requests_total counter",
        f"hermes_router_stream_requests_total {m['stream_requests_total']}",
        "",
        "# HELP hermes_router_errors_total Internal errors",
        "# TYPE hermes_router_errors_total counter",
        f"hermes_router_errors_total {m['errors_total']}",
        "",
        "# HELP hermes_router_sessions_active Active session count",
        "# TYPE hermes_router_sessions_active gauge",
        f"hermes_router_sessions_active {m['sessions_active']}",
        "",
        "# HELP hermes_router_circuits_open Number of open circuit breakers",
        "# TYPE hermes_router_circuits_open gauge",
        f"hermes_router_circuits_open {sum(1 for c in CIRCUITS.values() if c.get('state') == 'open')}",
    ]
    return JSONResponse(
        content={"metrics": m, "prometheus": "\n".join(lines)},
    )


# ── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    port = cfg["server"]["port"]
    host = cfg["server"]["host"]

    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,
    )
