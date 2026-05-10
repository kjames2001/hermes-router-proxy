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
from fastapi.responses import JSONResponse, StreamingResponse

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [router] %(levelname)s %(message)s",
)
log = logging.getLogger("hermes-router")

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
    reads SOUL.md and caches a 2–3 sentence summary back to config.
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
    """Read SOUL.md, pass to flash model, return a 2–3 sentence summary."""
    soul_path = Path(cfg["persona"]["soul_path"]).expanduser()
    if not soul_path.exists():
        log.warning("SOUL.md not found at %s — skipping profile extraction", soul_path)
        return "No profile available."

    max_chars = cfg["persona"].get("max_context_chars", 800)
    raw = soul_path.read_text()[:max_chars]

    prompt = (
        "Summarize this AI agent's user identity, environment, and key tools "
        "in 2–3 concise sentences. Keep only what helps classify tasks as 'simple' "
        f"or 'complex'.\n\n{raw}"
    )

    log.info("Extracting profile hint from SOUL.md (%d chars) → flash model", len(raw))
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
    prompt = build_classification_prompt(cfg, user_message)
    result = _call_classifier_raw(cfg, prompt, max_tokens=32)

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
    model_cfg: dict, request_payload: dict
) -> httpx.Response:
    """Call an OpenAI-compatible endpoint. Returns the httpx response.

    Key rotation: if the primary key returns HTTP 429 (rate-limited),
    retries with alternate_key_env before giving up.
    """
    url = f"{model_cfg['base_url'].rstrip('/')}/chat/completions"
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

    # ── Call model ──────────────────────────────────────────────────────
    model_cfg = cfg["models"][tier]
    log.info("Routing session %s → %s (%s)", key, tier, model_cfg["model"])

    resp = call_model(model_cfg, payload)

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

    fb_cfg = {
        "model": fallback_model,
        "base_url": model_cfg["fallback_base_url"],
        "api_key_env": model_cfg["fallback_key_env"],
        "timeout_seconds": model_cfg.get("timeout_seconds", 120),
    }
    fb_resp = call_model(fb_cfg, payload)

    if fb_resp.status_code == 200:
        data = fb_resp.json()
        data.setdefault("hermes_router", {})["fallback_used"] = True
        return JSONResponse(content=data)

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


@app.on_event("startup")
def _startup() -> None:
    cfg = load_config()
    log.info("Router starting on %s:%s", cfg["server"]["host"], cfg["server"]["port"])
    log.info("  Classifier: %s (%s)", cfg["classifier"]["model"], cfg["classifier"]["base_url"])
    log.info("  Simple model: %s (%s)", cfg["models"]["simple"]["model"], cfg["models"]["simple"]["base_url"])
    log.info("  Complex model: %s (%s)", cfg["models"]["complex"]["model"], cfg["models"]["complex"]["base_url"])
    if cfg["models"]["complex"].get("fallback_model"):
        log.info("  Fallback: %s (%s)", cfg["models"]["complex"]["fallback_model"], cfg["models"]["complex"]["fallback_base_url"])
    app.state.config = cfg


@app.get("/health")
async def health():
    """Simple liveness check."""
    return {"status": "ok", "sessions": len(SESSIONS)}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions — routed automatically."""
    verify_auth(request)
    cfg = request.app.state.config
    payload = await request.json()
    if payload.get("stream"):
        return StreamingResponse(
            route_request_stream(cfg, payload),
            media_type="text/event-stream",
        )
    return route_request(cfg, payload)


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
