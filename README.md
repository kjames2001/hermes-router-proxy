# Hermes Router-Proxy

An intelligent LLM routing proxy that classifies prompts as "simple" or "complex" using a cheap local model, then transparently routes them to the right cloud model. **OpenAI API-compatible** — drop it in front of any OpenAI-compatible client.

**The idea:** Not every query needs your most expensive model. Simple questions (chat, trivia, definitions) go to a fast/cheap model. Hard questions (code, debug, deploy) go to a smart/expensive model. A small classifier handles routing with ~200ms overhead.

**Tested results** with `qwen2.5:3b` on Ollama: **9/10 accuracy**, **~200ms classification overhead**.

---

## Features

- **Smart classification** — Uses a local flash model (e.g., `qwen2.5:3b` via Ollama) to classify each new prompt as `simple` or `complex`
- **TRACER surrogate classifier** — Offline-learned surrogate model (TF-IDF or MiniLM-L6-v2 embeddings) that handles 90%+ of classifications locally with ~1ms latency and zero LLM cost. Trained on production trace data with teacher agreement gating. Weekly auto-refit via cron
- **Sentence embeddings** — When `sentence-transformers` is installed, `all-MiniLM-L6-v2` (384-dim) embeddings provide richer semantic representations for the surrogate. Auto-selected with `--prefer-embeddings` flag
- **`/classifier/report` endpoint** — Live dashboard showing surrogate coverage, confidence distribution, per-model routing stats, hourly drift timeline, and fallback usage. Supports `?format=html` for dark-themed visualization
- **Session awareness** — Classifies only the first message in a conversation. Follow-ups reuse the cached tier with sub-millisecond keyword deviation detection
- **Keyword overrides** — "implement", "debug", "deploy" always route complex. "thanks", "lol", "how are you" always route simple. These bypass the classifier entirely
- **Fuzzy keyword matching** — Typo-tolerant keyword detection using normalized substring matching and Levenshtein distance (≤1 edit)
- **OpenAI-compatible endpoint** — Drop-in replacement at `POST /v1/chat/completions`. Works with any OpenAI SDK, Hermes Agent, or compatible client
- **Full multimodal support** — Handles image+text payloads (array content messages)
- **Health endpoint** — `GET /health` for liveness checks and session count monitoring

### Resilience

- **Multi-tier fallback cascade** — Primary → fallback1 → fallback2, each with independent base URL and API key
- **API key rotation** — Automatic retry with alternate keys on HTTP 429 at every tier (primary, fallback1, fallback2)
- **503 retry with exponential backoff** — Sync path retries HTTP 503 up to 3x (2s/4s/8s) before escalating
- **Transport retry** — Retryable errors (ConnectError, RemoteProtocolError, ReadTimeout, ConnectTimeout) retry up to 3x with 1s backoff
- **Circuit breakers** — Per-endpoint breaker trips after N consecutive 429s/5xx, recovers automatically after a configurable timeout (half-open state)

### Streaming Resilience (Transparent Proxy)

The router-proxy acts as a **transparent SSE proxy** — no internal retry loops on the streaming path:

- **Clean cut on failure** — When the inner stream breaks, the outer SSE is immediately closed with `data: [DONE]`. No backoff, no silence.
- **Delegated retry** — The Hermes gateway's own `HERMES_STREAM_RETRIES` mechanism detects the clean close and reconnects with a fresh request.
- **3-strike fallback** — A per-session failure counter tracks consecutive breaks. Only after **3 consecutive failures** does the next gateway retry skip the primary model and go directly to fallback. This means the router tries hard to reconnect to the primary (like a direct connection would), only downgrading when the endpoint is genuinely unhealthy.
- **Auto-recovery** — A successful stream immediately resets the counter to zero.

### Observability

- **Prometheus-compatible metrics** — `GET /metrics` with counters for requests by tier, classifier calls/latency, cache hits, 429s by tier, fallback usage, streaming requests, errors, active sessions, open circuits
- **JSON structured logging** — `LOG_FORMAT=json` env var enables one-line JSON logs for both router and uvicorn
- **Structured JSONL trace logging** — Every routing decision (classify, cache hit, deviation, fallback, circuit breaker, key rotation) is emitted as a JSON line to a date-stamped trace file. Enables full audit trail and post-hoc analysis of classifier behavior. See [Trace Logging](#trace-logging) below.
- **Admin API** — `POST /reload` (hot-reload config without restart), `GET/DELETE /admin/sessions` (inspect/evict cached sessions), `GET /admin/config` (inspect live config), `GET /circuits` + `POST /circuits/reset` (circuit breaker management)

### Security & Compatibility

- **CORS** — Configurable via `CORS_ORIGINS` env var (defaults to `*`)
- **API key auth** — `ROUTER_PROXY_API_KEY` env var protects admin/metrics/circuit endpoints; `/health` always open, `/v1/chat/completions` optional
- **Profile-aware classification** — Reads both `USER.md` and `MEMORY.md` (your agent persona + durable memory) and injects a 2–3 sentence summary into the classifier prompt

---

## Architecture

```
User → Hermes Agent → router-proxy (localhost:8766)
                           │
                    ┌──────┴──────┐
                    │  classifier  │
                    │  (2-stage)   │
                    │              │
                    │ surrogate    │ ← ~1ms, zero-cost
                    │ ↓ (low conf) │
                    │ qwen2.5:3b   │ ← ~200ms fallback
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              ↓                         ↓
        simple model              complex model
   (e.g. deepseek-v4-flash)    (e.g. deepseek-v4-pro)
                                   │
              ┌────────────────────┼────────────────────┐
              ↓                    ↓                    ↓
        fallback1             fallback2            alt keys
   (e.g. glm-5.1)         (e.g. big-pickle)     (key rotation)
```

---

## How It Works

### Routing Pipeline

1. **New session** — First message is sent to the flash classifier model. It reads the prompt (with a profile hint from `USER.md` + `MEMORY.md`) and returns `simple` or `complex`. The tier is cached with a configurable timeout.

2. **Follow-up messages** — The cached tier is reused instantly. The router scans for escalation keywords (e.g., "fix", "error", "why doesn't this work") or de-escalation keywords (e.g., "thanks", "ok"). If a keyword is detected, it re-classifies and potentially switches tiers mid-conversation.

3. **Model call** — The request payload (with full message history) is forwarded to the appropriate model endpoint. The `model` field in the payload is overwritten to match the routed model.

4. **Fallback cascade** — If the primary model fails, the request retries against fallback1. If that also fails, fallback2 is tried. If all fail, a 502 error is returned with upstream context.

5. **Key rotation at every tier** — If any tier returns HTTP 429, the alternate key for that tier is tried automatically before falling through.

6. **Circuit breakers** — After N consecutive 429s from an endpoint within a sliding window, the circuit opens and that endpoint is skipped entirely for X seconds (configurable recovery timeout). After recovery, a single success in half-open state closes the circuit.

### Session Lifecycle

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│ "What time  │     │ "Also, can  │     │ "Actually,   │
│  is it?"    │────▶│  you write  │────▶│  write a      │
│ → simple    │     │  me a bash  │     │  deployment   │
│  (cached)   │     │  script?"   │     │  script?"     │
└─────────────┘     │ → keyword   │     │ → keyword    │
                    │   deviation │     │   deviation  │
                    │   detected  │     │   detected   │
                    │ → reclass   │     │ → reclass    │
                    │ → complex   │     │ → complex    │
                    └─────────────┘     └──────────────┘
```

Sessions expire after a configurable timeout (default: 5 minutes), triggering re-classification on the next message.

---

## Setup

```bash
# Clone
git clone https://github.com/kjames2001/hermes-router-proxy.git
cd hermes-router-proxy

# Copy and edit config
cp router_config.example.yaml router_config.yaml
# Edit router_config.yaml with your models and API keys

# Install dependencies
pip install httpx uvicorn fastapi pyyaml

# Run
python server.py
```

### Systemd Service

```bash
sudo cp router-config/systemd/hermes-router.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now hermes-router
```

---

## Configuration

See `router_config.example.yaml` for the full annotated template. Key sections:

### Server

```yaml
server:
  host: "127.0.0.1"
  port: 8766
```

### Classifier

```yaml
classifier:
  model: "qwen2.5:3b"
  base_url: "http://localhost:11434/v1"
  api_key_env: ""              # Ollama doesn't need auth
  session_timeout_minutes: 5   # Cache expiry
  system_prompt: "..."          # Classification prompt template
  profile_hint: ""              # Auto-filled from USER.md + MEMORY.md on first run
```

### Models (with fallback cascade + key rotation)

```yaml
models:
  simple:
    model: "deepseek-v4-flash"
    base_url: "https://api.example.com/v1"
    api_key_env: "PROVIDER_API_KEY"
    alternate_key_env: ""          # Optional — key rotation on 429
    fallback_model: "gpt-4o-mini"  # Optional fallback1
    fallback_base_url: "https://api.openai.com/v1"
    fallback_key_env: "OPENAI_API_KEY"
    fallback_alternate_key_env: "" # Optional — key rotation on fallback1 429
    fallback2_model: "claude-haiku" # Optional fallback2 (last resort)
    fallback2_base_url: "https://api.anthropic.com/v1"
    fallback2_key_env: "ANTHROPIC_API_KEY"
    fallback2_alternate_key_env: "" # Optional — key rotation on fallback2 429
    timeout_seconds: 120

  complex:
    model: "deepseek-v4-pro"
    # ... same structure as simple
```

### Circuit Breaker

```yaml
circuit_breaker:
  failure_threshold: 3       # Consecutive 429s before tripping
  recovery_timeout_sec: 30   # How long circuit stays open
  window_sec: 60             # Sliding window for counting failures
```

### Routing Keywords

```yaml
routing:
  escalation_keywords:
    - implement - debug - deploy - configure - fix
    - error - script - code - automation - server
  de_escalation_keywords:
    - thanks - thank - lol - ok - cool - nice
    - good morning - how are you - hello - hi
```

### Persona (reads both USER.md + MEMORY.md)

```yaml
persona:
  user_path: "~/.hermes/USER.md"      # Who the user is
  memory_path: "~/.hermes/MEMORY.md"  # Durable facts & rules
  max_context_chars: 800              # Max chars read for hint extraction
```

### Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `ROUTER_PROXY_API_KEY` | Auth for admin/metrics/circuit endpoints | none (open) |
| `LOG_FORMAT` | Set to `json` for structured JSON logging | plain text |
| `CORS_ORIGINS` | Comma-separated allowed origins | `*` |
| `TRACE_LOG_DIR` | Directory for JSONL trace files | `./traces` |
| `TRACE_LOG_MAX_BYTES` | Max bytes per trace file before rotation | `10485760` (10 MB) |
| `TRACE_LOG_BACKUPS` | Number of rotated trace files to keep | `5` |
| `TRACE_LOG_ENABLED` | Set to `false` or `0` to disable trace logging | `true` |

---

## API Endpoints

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/v1/chat/completions` | POST | Optional | OpenAI-compatible chat completions (routed) |
| `/health` | GET | None | Liveness check + active session count |
| `/metrics` | GET | Required | Prometheus-compatible metrics |
| `/reload` | POST | Required | Hot-reload `router_config.yaml` without restart |
| `/admin/sessions` | GET | Required | List all cached sessions with tier + age |
| `/admin/sessions/{key}` | DELETE | Required | Force-evict a cached session |
| `/admin/config` | GET | Required | Show live config (env var names only, no secrets) |
| `/circuits` | GET | Required | List all circuit breaker states |
| `/circuits/reset` | POST | Required | Reset all circuit breakers to closed |
| `/classifier/report` | GET | Required | Surrogate coverage, confidence distribution, routing stats, drift timeline. `?format=html` for dark-themed dashboard |

The `/v1/chat/completions` endpoint accepts the standard OpenAI chat completions request body (`model`, `messages`, `temperature`, `stream`, etc.). The `model` field is overwritten by the router based on classification.

### Authentication

By default, the router-proxy is **open** (no auth required for `/v1/chat/completions` or `/health`). To protect admin, metrics, and circuit endpoints:

```bash
# Generate a key
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Set it
echo 'ROUTER_PROXY_API_KEY=your-secret-key-here' >> ~/.hermes/.env
```

Clients must send the key as a Bearer token:
```
Authorization: Bearer your-secret-key-here
```

If the env var is unset or empty, admin endpoints **reject all requests** (not open). The `/health` and `/v1/chat/completions` endpoints remain available.

---

## Point Hermes At It

In `~/.hermes/config.yaml`:

```yaml
model:
  default: deepseek-v4-pro          # Primary model — router proxies to this or fallback
  provider: auto-router             # Routes through the proxy

# ── Custom provider block ───────────────────────────────────────────────────
custom_providers:
- name: auto-router
  base_url: http://localhost:8766/v1
  model: deepseek-v4-pro            # Must match one of the configured tier models

# ── Fallback chain (gateway-level, fires if router-proxy is unreachable) ────
fallback_providers:
- provider: nous
  model: qwen/qwen3.6-plus
- provider: opencode-zen
  model: big-pickle
```

Then restart the Hermes gateway.

---

## Docker Deployment

The router-proxy can run as a Docker container on any host with Docker Engine.

### Build & Run

```bash
# Build the image
docker build -t hermes-router:latest .

# Run (mount your config + env)
docker run -d --name hermes-router --restart unless-stopped \
  -p 8766:8766 \
  -v /path/to/router_config.yaml:/app/router_config.yaml:ro \
  --env-file /root/.hermes/.env \
  hermes-router:latest
```

### Docker Compose

```yaml
# docker-compose.yaml
services:
  router-proxy:
    image: hermes-router:latest
    build: .
    container_name: hermes-router
    restart: unless-stopped
    ports:
      - "8766:8766"
    env_file:
      - /root/.hermes/.env
    volumes:
      - ./router_config.yaml:/app/router_config.yaml:ro
```

```bash
docker compose up -d
```

> **Note:** The default image comes with `router_config.example.yaml` as the config. Mount your real `router_config.yaml` to override. The env file needs the API keys and optionally `ROUTER_PROXY_API_KEY` for auth, `CORS_ORIGINS` for CORS, and `LOG_FORMAT` for JSON logging.

---

## Classifier Performance

### LLM Classifier

Tested with `qwen2.5:3b` on Ollama:
- **Accuracy:** 9/10 correct classifications
- **Overhead:** ~200ms average per classification
- **Misses:** Ambiguous queries ("What time is it?") — easily patched with keyword rules

### TRACER Surrogate Classifier

Inspired by [TRACER (Trace-Based Adaptive Cost-Efficient Routing for LLM Classification)](https://github.com/adrida/tracer) — an offline-learned surrogate model that handles the majority of classifications locally, eliminating LLM classifier calls for most requests.

**How it works:**
1. Production trace data (`classify` + `cache_hit` events) is collected continuously
2. A weekly cron job (`refit_surrogate.sh`) refits TF-IDF and sentence-embedding classifiers against the LLM teacher labels
3. The best candidate (by cross-validated accuracy) is saved as `.router/surrogate/pipeline.joblib`
4. At inference time, the surrogate classifies requests with ~1ms latency and zero LLM cost
5. A calibrated acceptor gate rejects low-confidence predictions, falling back to the LLM classifier

**Current surrogate:** `embeddings_lr` (all-MiniLM-L6-v2 + LogisticRegression)
- CV accuracy: 1.0000 (11 traces — will improve with more data)
- Coverage: 100% of traffic handled locally
- Teacher agreement: 100%

**Key difference from TRACER:** Our router-proxy implements TRACER's trace-based adaptive routing concept but with a different architecture — we use a dual-tier (simple/complex) classifier with keyword deviation detection, session caching, and a streaming 3-tier fallback cascade with circuit breakers and key rotation. TRACER focuses on multi-class routing with a k-NN + linear probe surrogate; we use TF-IDF/embeddings + LR with teacher agreement gating.

**Fitting the surrogate:**

```bash
# Default (best CV accuracy wins)
python fit_surrogate.py

# Prefer sentence embeddings when within 5% of TF-IDF accuracy
python fit_surrogate.py --prefer-embeddings

# Custom teacher agreement target
python fit_surrogate.py --target 0.98
```

**Weekly auto-refit (cron):**

A cron job runs `refit_surrogate.sh` weekly — reads traces, refits the surrogate, restarts the proxy, and runs a smoke test.

---

## Monitoring

### Plain text (default)

```
2026-05-10 13:33:00,123 [router] INFO Routing session a1b2c3 → simple (deepseek-v4-flash)
2026-05-10 13:33:00,456 [router] INFO Deviation: escalation keyword 'debug' matched
2026-05-10 13:33:00,789 [router] INFO Session a1b2c3 tier changed: simple → complex
2026-05-10 13:33:01,012 [router] WARNING Primary key rate-limited (429) - switching to alternate key
```

### JSON (`LOG_FORMAT=json`)

```json
{"timestamp":"2026-05-10T13:33:00","level":"INFO","logger":"hermes-router","module":"server","line":123,"message":"Routing session a1b2c3 → simple (deepseek-v4-flash)"}
```

### Metrics (`GET /metrics`)

```
hermes_router_requests_total{tier="simple"} 15234
hermes_router_requests_total{tier="complex"} 3892
hermes_router_classifier_calls_total 4201
hermes_router_classifier_latency_ms_avg 187
hermes_router_cache_hits_total 14826
hermes_router_429_total{tier="complex"} 3
hermes_router_fallback_used_total{level="1"} 12
hermes_router_fallback_used_total{level="2"} 2
hermes_router_circuits_open 1
hermes_router_sessions_active 8
```

### Trace Logging (JSONL)

Every routing decision is emitted as a structured JSON line to a rotating trace log. This provides a full audit trail for debugging classifier accuracy, analyzing routing patterns, and post-hoc investigation.

**Configuration (environment variables):**

| Variable | Purpose | Default |
|---|---|---|
| `TRACE_LOG_DIR` | Directory for trace files | `./traces` |
| `TRACE_LOG_MAX_BYTES` | Max bytes per file before rotation | `10485760` (10 MB) |
| `TRACE_LOG_BACKUPS` | Number of rotated files to keep | `5` |
| `TRACE_LOG_ENABLED` | Set to `false` or `0` to disable | `true` |

**Trace file format:**

Files are named `router-trace-YYYYMMDD.jsonl` (one per day) and each line is a JSON object:

```json
{"ts":"2026-05-13T09:15:23.456Z","event":"classify","session_key":"a1b2c3","classifier_result":"complex","latency_ms":210.5,"tier":"complex","model":"deepseek-v4-pro","is_first":true}
{"ts":"2026-05-13T09:15:24.100Z","event":"cache_hit","session_key":"a1b2c3","tier":"simple","model":"deepseek-v4-flash","age_sec":45.2}
{"ts":"2026-05-13T09:15:25.200Z","event":"deviation","session_key":"d4e5f6","keyword":"debug","direction":"escalation","previous_tier":"simple","new_tier":"complex","model":"deepseek-v4-pro"}
{"ts":"2026-05-13T09:15:26.300Z","event":"route","session_key":"a1b2c3","tier":"complex","model":"deepseek-v4-pro","upstream_status":200,"stream":true}
{"ts":"2026-05-13T09:15:27.400Z","event":"circuit","base_url":"https://api.example.com/v1","old_state":"closed","new_state":"open","failures":3}
{"ts":"2026-05-13T09:15:28.500Z","event":"key_rotation","base_url":"https://api.example.com/v1","tier":"complex","reason":"429_rate_limit"}
{"ts":"2026-05-13T09:15:29.600Z","event":"stream_error","session_key":"a1b2c3","model":"deepseek-v4-pro","error":"RemoteProtocolError: Connection lost","failure_count":2,"max_failures":3}
```

**Event types:**

| Event | When emitted | Key fields |
|---|---|---|
| `classify` | Flash model classifies a prompt | `session_key`, `classifier_result`, `latency_ms`, `tier`, `model`, `is_first` |
| `cache_hit` | Reused cached tier for follow-up | `session_key`, `tier`, `model`, `age_sec` |
| `deviation` | Keyword match changes tier mid-session | `keyword`, `direction` (escalation/de_escalation), `previous_tier`, `new_tier` |
| `route` | Request forwarded to upstream model | `session_key`, `tier`, `model`, `upstream_status`, `stream`, `fallback_level`, `fallback_model` |
| `circuit` | Circuit breaker state change | `base_url`, `old_state`, `new_state`, `failures` |
| `key_rotation` | API key rotated on 429 | `base_url`, `tier`, `reason` |
| `stream_error` | Streaming transport failure | `session_key`, `model`, `error`, `failure_count`, `max_failures` |

**Querying traces:**

```bash
# All classification decisions for a session
cat traces/router-trace-20260513.jsonl | jq 'select(.event=="classify")' | less

# Sessions that escalated from simple to complex
cat traces/router-trace-*.jsonl | jq 'select(.event=="deviation" and .direction=="escalation")'

# Average classifier latency (ms)
cat traces/router-trace-*.jsonl | jq 'select(.event=="classify") .latency_ms' | awk '{sum+=$1; count++} END{print sum/count}'

# Circuit breaker trips
cat traces/router-trace-*.jsonl | jq 'select(.event=="circuit" and .new_state=="open")'
```

---

## License

MIT

## Author

James Huang + Jarvis (Hermes Agent)
