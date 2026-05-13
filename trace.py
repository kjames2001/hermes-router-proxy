"""
Structured JSONL trace logging for the Hermes Router-Proxy classifier.

Every routing decision — classification, cache hit, deviation detection,
fallback, circuit breaker state change — is emitted as a single JSON line
to a rotating trace log file.  This gives full auditability and makes it
trivial to grep, jq, or pipe into analysis tools.

Configuration via environment variables:
    TRACE_LOG_DIR   — directory for trace files (default: ./traces)
    TRACE_LOG_MAX_BYTES — max bytes per file before rotation (default: 10 MB)
    TRACE_LOG_BACKUPS   — number of rotated files to keep (default: 5)
    TRACE_LOG_ENABLED   — set to "false" or "0" to disable (default: enabled)

Author: James Huang + Jarvis (Hermes Agent)
License: MIT
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Config ──────────────────────────────────────────────────────────────────

_TRACE_DIR = Path(os.environ.get("TRACE_LOG_DIR", "./traces"))
_TRACE_MAX_BYTES = int(os.environ.get("TRACE_LOG_MAX_BYTES", 10 * 1024 * 1024))  # 10 MB
_TRACE_BACKUPS = int(os.environ.get("TRACE_LOG_BACKUPS", 5))
_TRACE_ENABLED = os.environ.get("TRACE_LOG_ENABLED", "true").strip().lower() not in (
    "false", "0", "no", "off",
)

# Thread-safe write lock — only one writer at a time.
_write_lock = threading.Lock()


# ── Public API ──────────────────────────────────────────────────────────────

def trace_event(event: str, **fields: Any) -> None:
    """
    Emit a structured trace event.

    Args:
        event: Event type (e.g. "classify", "cache_hit", "deviation", "route").
        **fields: Additional key-value pairs merged into the trace record.

    Each event automatically gets:
        - ts: ISO-8601 timestamp with milliseconds
        - event: the event type
        - trace_id: unique UUID for correlating events
    """
    if not _TRACE_ENABLED:
        return

    record: dict[str, Any] = {
        "ts": _now_iso(),
        "event": event,
        **fields,
    }

    line = json.dumps(record, ensure_ascii=False, default=str, separators=(",", ":"))
    _write_line(line)


def trace_classify(
    session_key: str,
    user_message: str,
    classifier_result: str,
    classifier_raw: str,
    latency_ms: float,
    tier: str,
    model: str,
    is_first: bool,
) -> None:
    """Emit a classification trace event."""
    # Truncate user message to avoid bloating the trace log
    msg_preview = user_message[:120] + ("..." if len(user_message) > 120 else "")
    trace_event(
        "classify",
        session_key=session_key,
        user_message_preview=msg_preview,
        classifier_result=classifier_result,
        classifier_raw=classifier_raw[:200],
        latency_ms=round(latency_ms, 1),
        tier=tier,
        model=model,
        is_first=is_first,
    )


def trace_cache_hit(
    session_key: str,
    tier: str,
    model: str,
    age_sec: float,
) -> None:
    """Emit a cache-hit trace event."""
    trace_event(
        "cache_hit",
        session_key=session_key,
        tier=tier,
        model=model,
        age_sec=round(age_sec, 1),
    )


def trace_deviation(
    session_key: str,
    keyword: str,
    direction: str,
    previous_tier: str,
    new_tier: str,
    model: str,
) -> None:
    """Emit a keyword deviation trace event."""
    trace_event(
        "deviation",
        session_key=session_key,
        deviation_keyword=keyword,
        deviation_direction=direction,
        previous_tier=previous_tier,
        new_tier=new_tier,
        model=model,
    )


def trace_route(
    session_key: str,
    tier: str,
    model: str,
    upstream_status: int,
    latency_ms: float | None = None,
    fallback_level: int | None = None,
    fallback_model: str | None = None,
    stream: bool = False,
) -> None:
    """Emit a route/forward trace event."""
    fields: dict[str, Any] = {
        "session_key": session_key,
        "tier": tier,
        "model": model,
        "upstream_status": upstream_status,
        "stream": stream,
    }
    if latency_ms is not None:
        fields["latency_ms"] = round(latency_ms, 1)
    if fallback_level is not None:
        fields["fallback_level"] = fallback_level
    if fallback_model is not None:
        fields["fallback_model"] = fallback_model
    trace_event("route", **fields)


def trace_circuit(
    base_url: str,
    old_state: str,
    new_state: str,
    failures: int | None = None,
) -> None:
    """Emit a circuit breaker state change trace event."""
    fields: dict[str, Any] = {
        "base_url": base_url,
        "old_state": old_state,
        "new_state": new_state,
    }
    if failures is not None:
        fields["failures"] = failures
    trace_event("circuit", **fields)


def trace_key_rotation(
    base_url: str,
    tier: str,
    reason: str,
) -> None:
    """Emit a key rotation trace event."""
    trace_event(
        "key_rotation",
        base_url=base_url,
        tier=tier,
        reason=reason,
    )


def trace_stream_error(
    session_key: str,
    model: str,
    error: str,
    failure_count: int,
    max_failures: int,
) -> None:
    """Emit a streaming error trace event."""
    trace_event(
        "stream_error",
        session_key=session_key,
        model=model,
        error=error[:200],
        failure_count=failure_count,
        max_failures=max_failures,
    )


# ── Internal ────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    """Return current UTC time in ISO-8601 with milliseconds."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _trace_file_path() -> Path:
    """Return the current trace file path (date-stamped)."""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    return _TRACE_DIR / f"router-trace-{date_str}.jsonl"


def _write_line(line: str) -> None:
    """Append a line to the trace log file with rotation."""
    with _write_lock:
        path = _trace_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        # Rotate if needed
        if path.exists() and path.stat().st_size >= _TRACE_MAX_BYTES:
            _rotate(path)

        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _rotate(path: Path) -> None:
    """Rotate trace log files: trace-DATE.jsonl → trace-DATE.jsonl.1, etc."""
    for i in range(_TRACE_BACKUPS - 1, 0, -1):
        older = Path(f"{path}.{i}")
        newer = Path(f"{path}.{i - 1}") if i > 1 else path
        if older.exists():
            older.unlink()
        if newer.exists():
            newer.rename(older)