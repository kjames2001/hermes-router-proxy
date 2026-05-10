# Hermes Router-Proxy

An intelligent LLM routing proxy that classifies prompts as "simple" or "complex" using a cheap local model, then transparently routes them to the right cloud model. **OpenAI API-compatible** — drop it in front of any OpenAI-compatible client.

**The idea:** Not every query needs your most expensive model. Simple questions (chat, trivia, definitions) go to a fast/cheap model. Hard questions (code, debug, deploy) go to a smart/expensive model. A small classifier handles routing with ~200ms overhead.

**Tested results** with `qwen2.5:3b` on Ollama: **9/10 accuracy**, **~200ms classification overhead**.

---

## Features

- **Smart classification** — Uses a local flash model (e.g., `qwen2.5:3b` via Ollama) to classify each new prompt as `simple` or `complex`
- **Session awareness** — Classifies only the first message in a conversation. Follow-ups reuse the cached tier with sub-millisecond keyword deviation detection
- **Keyword overrides** — "implement", "debug", "deploy" always route complex. "thanks", "lol", "how are you" always route simple. These bypass the classifier entirely
- **Fuzzy keyword matching** — Typo-tolerant keyword detection using normalized substring matching and Levenshtein distance (≤1 edit)
- **OpenAI-compatible endpoint** — Drop-in replacement at `POST /v1/chat/completions`. Works with any OpenAI SDK, Hermes Agent, or compatible client
- **Full multimodal support** — Handles image+text payloads (array content messages)
- **Health endpoint** — `GET /health` for liveness checks and session count monitoring

### Resilience

- **Multi-tier fallback cascade** — Primary → fallback1 → fallback2, each with independent base URL and API key
- **API key rotation** — Automatic retry with alternate keys on HTTP 429 at every tier (primary, fallback1, fallback2)
- **Circuit breakers** — Per-endpoint breaker trips after N consecutive 429s, recovers automatically after a configurable timeout (half-open state)

### Observability

- **Prometheus-compatible metrics** — `GET /metrics` with counters for requests by tier, classifier calls/latency, cache hits, 429s by tier, fallback usage, streaming requests, errors, active sessions, open circuits
- **JSON structured logging** — `LOG_FORMAT=json` env var enables one-line JSON logs for both router and uvicorn
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
                           ▼
                    classifier (qwen2.5:3b)
                           │
              ┌────────────┴────────────┐
              ↓                         ↓
        simple model              complex model
   (e.g. deepseek-v4-flash)    (e.g. deepseek-v4-pro)
                                   │
              ┌────────────────────┼────────────────────┐
              ↓                    ↓                    ↓
        fallback1             fallback2            alt keys
   (e.g. o1-mini)         (e.g. big-pickle)     (key rotation)
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
  default: deepseek-v4-pro      # Must match a configured model name
  provider: ollama-cloud
  base_url: http://localhost:8766/v1   # ← point here instead of direct API
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

Tested with `qwen2.5:3b` on Ollama:
- **Accuracy:** 9/10 correct classifications
- **Overhead:** ~200ms average per classification
- **Misses:** Ambiguous queries ("What time is it?") — easily patched with keyword rules

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

---

## License

MIT

## Author

James Huang + Jarvis (Hermes Agent)
