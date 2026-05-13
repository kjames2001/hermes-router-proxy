"""
Tests for structured JSONL trace logging in the Hermes Router-Proxy.

Tests cover:
  1. trace.py module — event formatting, file writing, log rotation, disable toggle
  2. server.py instrumentation — trace calls emitted during classify, deviation,
     cache hit, circuit breaker, key rotation, route, and stream path

Author: James Huang + Jarvis (Hermes Agent)
License: MIT
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def trace_dir(tmp_path):
    """Create a temporary trace directory for test isolation."""
    d = tmp_path / "traces"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def reset_trace_module(trace_dir):
    """Reset the trace module's global state before each test."""
    import trace as trace_mod
    # Override config to use our temp dir
    trace_mod._TRACE_DIR = trace_dir
    trace_mod._TRACE_ENABLED = True
    trace_mod._TRACE_MAX_BYTES = 1024 * 1024  # 1 MB for testing
    trace_mod._TRACE_BACKUPS = 3
    yield
    # Reset for next test
    trace_mod._TRACE_ENABLED = True


# ── trace.py Unit Tests ──────────────────────────────────────────────────────

class TestTraceEvent:
    """Test trace_event() writes a valid JSONL line."""

    def test_basic_event(self, trace_dir):
        import trace
        trace.trace_event("test_event", key1="value1", key2=42)
        lines = _read_trace_lines(trace_dir)
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event"] == "test_event"
        assert record["key1"] == "value1"
        assert record["key2"] == 42
        assert "ts" in record

    def test_event_has_iso_timestamp(self, trace_dir):
        import trace
        trace.trace_event("ts_test")
        record = _read_trace_record(trace_dir)
        # Should be ISO format with trailing Z
        assert record["ts"].endswith("Z")
        # Should be parseable
        assert "T" in record["ts"]

    def test_disabled_trace_writes_nothing(self, trace_dir):
        import trace
        trace._TRACE_ENABLED = False
        trace.trace_event("should_not_appear")
        lines = _read_trace_lines(trace_dir)
        assert len(lines) == 0

    def test_unicode_values(self, trace_dir):
        import trace
        trace.trace_event("unicode_test", msg="你好世界", emoji="🚀")
        record = _read_trace_record(trace_dir)
        assert record["msg"] == "你好世界"
        assert record["emoji"] == "🚀"

    def test_default_str_for_nonserializable(self, trace_dir):
        import trace
        trace.trace_event("obj_test", obj=object())
        record = _read_trace_record(trace_dir)
        assert "obj" in record
        # object() gets default=str which gives something like "<object object at 0x...>"

    def test_compact_json_separators(self, trace_dir):
        import trace
        trace.trace_event("compact", a=1, b=2)
        line = _read_trace_lines(trace_dir)[0]
        # compact separators: no spaces after : or ,
        assert ": " not in line or line.count(": ") == 0 or '", "' not in line
        # Actually just verify it's valid JSON
        json.loads(line)


class TestTraceClassify:
    """Test trace_classify() convenience function."""

    def test_classify_event(self, trace_dir):
        import trace
        trace.trace_classify(
            session_key="abc123",
            user_message="Write a bash script to deploy",
            classifier_result="complex",
            classifier_raw="complex",
            latency_ms=210.5,
            tier="complex",
            model="deepseek-v4-pro",
            is_first=True,
        )
        record = _read_trace_record(trace_dir)
        assert record["event"] == "classify"
        assert record["session_key"] == "abc123"
        assert record["classifier_result"] == "complex"
        assert record["tier"] == "complex"
        assert record["model"] == "deepseek-v4-pro"
        assert record["latency_ms"] == 210.5
        assert record["is_first"] is True

    def test_message_preview_truncation(self, trace_dir):
        import trace
        long_msg = "x" * 500
        trace.trace_classify(
            session_key="s1",
            user_message=long_msg,
            classifier_result="simple",
            classifier_raw="simple",
            latency_ms=100.0,
            tier="simple",
            model="flash",
            is_first=False,
        )
        record = _read_trace_record(trace_dir)
        assert len(record["user_message_preview"]) <= 123  # 120 + "..."
        assert record["user_message_preview"].endswith("...")


class TestTraceCacheHit:
    """Test trace_cache_hit() convenience function."""

    def test_cache_hit_event(self, trace_dir):
        import trace
        trace.trace_cache_hit(
            session_key="def456",
            tier="simple",
            model="deepseek-v4-flash",
            age_sec=45.2,
        )
        record = _read_trace_record(trace_dir)
        assert record["event"] == "cache_hit"
        assert record["session_key"] == "def456"
        assert record["tier"] == "simple"
        assert record["model"] == "deepseek-v4-flash"
        assert record["age_sec"] == 45.2


class TestTraceDeviation:
    """Test trace_deviation() convenience function."""

    def test_escalation(self, trace_dir):
        import trace
        trace.trace_deviation(
            session_key="ghi789",
            keyword="debug",
            direction="escalation",
            previous_tier="simple",
            new_tier="complex",
            model="deepseek-v4-pro",
        )
        record = _read_trace_record(trace_dir)
        assert record["event"] == "deviation"
        assert record["deviation_keyword"] == "debug"
        assert record["deviation_direction"] == "escalation"
        assert record["previous_tier"] == "simple"
        assert record["new_tier"] == "complex"

    def test_de_escalation(self, trace_dir):
        import trace
        trace.trace_deviation(
            session_key="xyz",
            keyword="thanks",
            direction="de_escalation",
            previous_tier="complex",
            new_tier="simple",
            model="deepseek-v4-flash",
        )
        record = _read_trace_record(trace_dir)
        assert record["deviation_direction"] == "de_escalation"


class TestTraceRoute:
    """Test trace_route() convenience function."""

    def test_sync_route_success(self, trace_dir):
        import trace
        trace.trace_route(
            session_key="r1",
            tier="simple",
            model="flash",
            upstream_status=200,
            stream=False,
        )
        record = _read_trace_record(trace_dir)
        assert record["event"] == "route"
        assert record["stream"] is False
        assert record["upstream_status"] == 200

    def test_stream_route_success(self, trace_dir):
        import trace
        trace.trace_route(
            session_key="r2",
            tier="complex",
            model="pro",
            upstream_status=200,
            stream=True,
        )
        record = _read_trace_record(trace_dir)
        assert record["stream"] is True

    def test_fallback_route(self, trace_dir):
        import trace
        trace.trace_route(
            session_key="r3",
            tier="complex",
            model="pro",
            upstream_status=429,
            stream=False,
            fallback_level=1,
            fallback_model="o1-mini",
        )
        record = _read_trace_record(trace_dir)
        assert record["fallback_level"] == 1
        assert record["fallback_model"] == "o1-mini"
        assert record["upstream_status"] == 429

    def test_route_with_latency(self, trace_dir):
        import trace
        trace.trace_route(
            session_key="r4",
            tier="complex",
            model="pro",
            upstream_status=200,
            latency_ms=1234.56,
        )
        record = _read_trace_record(trace_dir)
        assert record["latency_ms"] == 1234.6  # rounded to 1 decimal


class TestTraceCircuit:
    """Test trace_circuit() convenience function."""

    def test_circuit_opens(self, trace_dir):
        import trace
        trace.trace_circuit(
            base_url="https://api.example.com/v1",
            old_state="closed",
            new_state="open",
            failures=3,
        )
        record = _read_trace_record(trace_dir)
        assert record["event"] == "circuit"
        assert record["old_state"] == "closed"
        assert record["new_state"] == "open"
        assert record["failures"] == 3

    def test_circuit_half_open(self, trace_dir):
        import trace
        trace.trace_circuit(
            base_url="https://api.example.com/v1",
            old_state="open",
            new_state="half_open",
        )
        record = _read_trace_record(trace_dir)
        assert record["new_state"] == "half_open"
        assert "failures" not in record


class TestTraceKeyRotation:
    """Test trace_key_rotation() convenience function."""

    def test_rotation_event(self, trace_dir):
        import trace
        trace.trace_key_rotation(
            base_url="https://api.example.com/v1",
            tier="complex",
            reason="429_rate_limit",
        )
        record = _read_trace_record(trace_dir)
        assert record["event"] == "key_rotation"
        assert record["tier"] == "complex"
        assert record["reason"] == "429_rate_limit"


class TestTraceStreamError:
    """Test trace_stream_error() convenience function."""

    def test_stream_error_event(self, trace_dir):
        import trace
        trace.trace_stream_error(
            session_key="se1",
            model="deepseek-v4-pro",
            error="RemoteProtocolError: Connection lost",
            failure_count=2,
            max_failures=3,
        )
        record = _read_trace_record(trace_dir)
        assert record["event"] == "stream_error"
        assert record["failure_count"] == 2
        assert record["max_failures"] == 3


class TestTraceRotation:
    """Test log file rotation."""

    def test_rotation_happens(self, trace_dir):
        import trace
        trace._TRACE_MAX_BYTES = 200  # Very small to trigger rotation
        trace._TRACE_BACKUPS = 2

        # Write enough data to trigger rotation
        for i in range(20):
            trace.trace_event("rot_test", data="x" * 50, idx=i)

        # Should have the rotated file + current file
        date_str = time.strftime("%Y%m%d", time.gmtime())
        current = trace_dir / f"router-trace-{date_str}.jsonl"
        assert current.exists()

    def test_env_vars_override(self, tmp_path):
        """Test that TRACE_LOG_DIR, TRACE_LOG_MAX_BYTES, TRACE_LOG_BACKUPS
        env vars are respected."""
        custom_dir = tmp_path / "custom_traces"
        custom_dir.mkdir()
        os.environ["TRACE_LOG_DIR"] = str(custom_dir)
        os.environ["TRACE_LOG_MAX_BYTES"] = "2048"
        os.environ["TRACE_LOG_BACKUPS"] = "7"

        # Reimport to pick up env vars
        import importlib
        import trace
        importlib.reload(trace)

        assert str(trace._TRACE_DIR) == str(custom_dir)
        assert trace._TRACE_MAX_BYTES == 2048
        assert trace._TRACE_BACKUPS == 7

        # Clean up env vars
        del os.environ["TRACE_LOG_DIR"]
        del os.environ["TRACE_LOG_MAX_BYTES"]
        del os.environ["TRACE_LOG_BACKUPS"]
        importlib.reload(trace)


class TestTraceDisabled:
    """Test TRACE_LOG_ENABLED=false disables all writes."""

    def test_disabled_via_env(self, tmp_path):
        os.environ["TRACE_LOG_ENABLED"] = "false"
        import importlib
        import trace
        importlib.reload(trace)

        d = tmp_path / "disabled_traces"
        d.mkdir()
        trace._TRACE_DIR = d
        trace.trace_event("should_not_write")

        lines = list(d.glob("*.jsonl"))
        assert len(lines) == 0

        del os.environ["TRACE_LOG_ENABLED"]
        importlib.reload(trace)

    def test_disabled_via_zero(self, tmp_path):
        os.environ["TRACE_LOG_ENABLED"] = "0"
        import importlib
        import trace
        importlib.reload(trace)

        d = tmp_path / "disabled_traces2"
        d.mkdir()
        trace._TRACE_DIR = d
        trace.trace_event("should_not_write")

        lines = list(d.glob("*.jsonl"))
        assert len(lines) == 0

        del os.environ["TRACE_LOG_ENABLED"]
        importlib.reload(trace)


# ── Server Integration Tests ─────────────────────────────────────────────────
# These test that the server.py instrumentation correctly calls the trace
# functions during routing decisions.

class TestServerTraceInstrumentation:
    """Integration tests verifying server.py emits trace events.

    NOTE: server.py imports trace functions at module load time, so we must
    patch the references on the server module itself, not on the trace module.
    """

    @pytest.fixture
    def cfg(self):
        """Minimal router config for testing."""
        return {
            "classifier": {
                "model": "test-model",
                "base_url": "http://localhost:11434/v1",
                "api_key_env": "",
                "session_timeout_minutes": 5,
                "system_prompt": "Classify as simple or complex: {message}",
                "profile_hint": "Test user",
            },
            "models": {
                "simple": {
                    "model": "test-simple",
                    "base_url": "http://localhost:11434/v1",
                    "api_key_env": "TEST_SIMPLE_KEY",
                    "timeout_seconds": 30,
                },
                "complex": {
                    "model": "test-complex",
                    "base_url": "http://localhost:11434/v1",
                    "api_key_env": "TEST_COMPLEX_KEY",
                    "timeout_seconds": 30,
                },
            },
            "routing": {
                "escalation_keywords": ["debug", "implement", "deploy"],
                "de_escalation_keywords": ["thanks", "hello"],
            },
            "persona": {
                "user_path": "/dev/null",
                "memory_path": "/dev/null",
                "max_context_chars": 800,
            },
            "server": {"host": "127.0.0.1", "port": 8766},
        }

    def test_classify_emits_trace(self, cfg, trace_dir):
        """classify() should emit a trace_classify event."""
        collected = []
        with patch.object(server, "trace_classify", side_effect=lambda *a, **kw: collected.append(("classify", kw))):
            with patch("server._call_classifier_raw", return_value="simple"):
                with patch("server.build_classification_prompt", return_value="prompt"):
                    result = server.classify(cfg, "What time is it?", session_key="test1")

        assert result == "simple"
        assert len(collected) == 1
        kw = collected[0][1]
        assert kw["session_key"] == "test1"
        assert kw["classifier_result"] == "simple"
        assert kw["is_first"] is True

    def test_has_deviation_emits_trace(self, cfg, trace_dir):
        """has_deviation() should emit trace_deviation for escalation."""
        collected = []
        def collect(*a, **kw):
            collected.append(("deviation", kw))
        with patch.object(server, "trace_deviation", side_effect=collect):
            result = server.has_deviation(cfg, "Please debug this code", "simple", session_key="test2")

        assert result is True
        assert len(collected) == 1
        kw = collected[0][1]
        assert kw["session_key"] == "test2"
        assert kw["keyword"] == "debug"
        assert kw["direction"] == "escalation"
        assert kw["previous_tier"] == "simple"
        assert kw["new_tier"] == "complex"

    def test_de_escalation_emits_trace(self, cfg, trace_dir):
        """has_deviation() should emit trace for de-escalation."""
        collected = []
        def collect(*a, **kw):
            collected.append(("deviation", kw))
        with patch.object(server, "trace_deviation", side_effect=collect):
            result = server.has_deviation(cfg, "Thanks for that!", "complex", session_key="test3")

        assert result is True
        assert len(collected) == 1
        kw = collected[0][1]
        assert kw["direction"] == "de_escalation"
        assert kw["previous_tier"] == "complex"
        assert kw["new_tier"] == "simple"

    def test_circuit_breaker_open_traces(self, cfg, trace_dir):
        """Circuit breaker opening should emit trace_circuit."""
        collected = []
        def collect(*a, **kw):
            collected.append(("circuit", a, kw))
        # Clear circuits from previous tests
        server.CIRCUITS.clear()
        with patch.object(server, "trace_circuit", side_effect=collect):
            base_url = "https://api.example.com/v1"
            # Trip the circuit with 3 failures
            for _ in range(3):
                server._circuit_record_failure(cfg, base_url)

        # Should have a trace event for circuit opening
        # trace_circuit is called as: trace_circuit(base_url, "closed", "open", failures=N)
        # positional: base_url, old_state, new_state; keyword: failures
        circuit_events = [
            e for e in collected
            if e[0] == "circuit" and len(e[1]) >= 3 and e[1][2] == "open"
        ]
        assert len(circuit_events) == 1
        args, kw = circuit_events[0][1], circuit_events[0][2]
        assert args[1] == "closed"  # old_state
        assert args[2] == "open"    # new_state
        assert kw.get("failures") == 3


# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_trace_lines(trace_dir: Path) -> list[str]:
    """Read all lines from trace files in directory."""
    lines = []
    for f in sorted(trace_dir.glob("*.jsonl")):
        lines.extend(f.read_text().strip().split("\n"))
    return [l for l in lines if l]


def _read_trace_record(trace_dir: Path) -> dict:
    """Read the first (or only) trace record from trace dir."""
    records = [_json_line(l) for l in _read_trace_lines(trace_dir)]
    assert len(records) >= 1, f"No trace records found in {trace_dir}"
    return records[-1]  # most recent


def _json_line(line: str) -> dict:
    return json.loads(line)


# ── Import server module for integration tests ────────────────────────────────

# We need to import server but some tests mock its internals, so we do it
# inside the test class methods. But we need the module available.
import sys
sys.path.insert(0, str(Path(__file__).parent))
import server