# Hermes Router-Proxy

An intelligent LLM routing proxy that classifies prompts as "simple" or "complex" using a cheap local model, then transparently routes them to the right cloud model. **OpenAI API-compatible** — drop it in front of any OpenAI-compatible client.

**The idea:** Not every query needs your most expensive model. Simple questions (chat, trivia, definitions) go to a fast/cheap model. Hard questions (code, debug, deploy) go to a smart/expensive model. A small classifier handles routing with ~200ms overhead.

**Tested results** with `qwen2.5:3b` on Ollama: **9/10 accuracy**, **~200ms classification overhead**.

---

## Features

- **Smart classification** — Uses a local flash model (e.g., `qwen2.5:3b` via Ollama) to classify each new prompt as `simple` or `complex`
- **Session awareness** — Classifies only the first message in a conversation. Follow-ups reuse the cached tier with sub-millisecond keyword deviation detection
- **Keyword overrides** — "implement", "debug", "deploy" always route complex. "thanks", "lol", "how are you" always route simple. These bypass the classifier entirely
- **Automatic fallback** — If the primary model returns an error (401, 429, etc.), automatically retries with a configured fallback model
- **API key rotation** — If the primary API key gets rate-limited (429), automatically retries with an alternate key
- **Profile-aware classification** — Reads your `SOUL.md` agent persona file and injects a 2–3 sentence summary into the classifier prompt, improving routing accuracy for your specific use case
- **Fuzzy keyword matching** — Typo-tolerant keyword detection using normalized substring matching and Levenshtein distance (≤1 edit)
- **OpenAI-compatible endpoint** — Drop-in replacement at `POST /v1/chat/completions`. Works with any OpenAI SDK, Hermes Agent, or compatible client
- **Full multimodal support** — Handles image+text payloads (array content messages)
- **Health endpoint** — `GET /health` for liveness checks and session count monitoring

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
                              fallback model
                           (e.g. claude-sonnet-4)
```

---

## How It Works

### Routing Pipeline

1. **New session** — First message is sent to the flash classifier model. It reads the prompt (with an optional agent profile hint from `SOUL.md`) and returns `simple` or `complex`. The tier is cached with a configurable timeout.

2. **Follow-up messages** — The cached tier is reused instantly. The router scans for escalation keywords (e.g., "fix", "error", "why doesn't this work") or de-escalation keywords (e.g., "thanks", "ok"). If a keyword is detected, it re-classifies and potentially switches tiers mid-conversation.

3. **Model call** — The request payload (with full message history) is forwarded to the appropriate model endpoint. The `model` field in the payload is overwritten to match the routed model.

4. **Fallback cascade** — If the primary model fails, the request retries against the configured fallback. If that also fails, a 502 error is returned with upstream context.

5. **Key rotation** — If the primary API key returns HTTP 429, the alternate key is tried automatically before falling back.

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
  profile_hint: ""              # Auto-filled from SOUL.md on first run
```

### Models
```yaml
models:
  simple:
    model: "deepseek-v4-flash"
    base_url: "https://api.example.com/v1"
    api_key_env: "PROVIDER_API_KEY"
    alternate_key_env: ""       # Optional key rotation

  complex:
    model: "deepseek-v4-pro"
    base_url: "https://api.example.com/v1"
    api_key_env: "PROVIDER_API_KEY"
    alternate_key_env: ""
    fallback_model: "claude-sonnet-4"        # Optional fallback
    fallback_base_url: "https://api.anthropic.com/v1"
    fallback_key_env: "ANTHROPIC_API_KEY"
    timeout_seconds: 120
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

### Persona
```yaml
persona:
  soulPath: "~/.hermes/SOUL.md"   # Path to your agent persona
  max_context_chars: 800          # Max chars to read for hint extraction
```

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

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions (routed) |
| `/health` | GET | Liveness check + active session count |

The `/v1/chat/completions` endpoint accepts the standard OpenAI chat completions request body (`model`, `messages`, `temperature`, `stream`, etc.). The `model` field is overwritten by the router based on classification.

### Authentication

By default, the router-proxy is **open** (no auth). To enable API key authentication:

1. Set environment variable:
   ```bash
   echo 'ROUTER_PROXY_API_KEY=your-secret-key-here' >> ~/.hermes/.env
   ```
   Generate a key with: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`

2. In `router_config.yaml`, uncomment or add:
   ```yaml
   auth:
     api_key_env: ROUTER_PROXY_API_KEY
   ```

3. Restart the service:
   ```bash
   systemctl --user restart hermes-router
   ```

Clients must send the key as a Bearer token in the `Authorization` header:
```
Authorization: Bearer your-secret-key-here
```
If the env var is unset or empty, auth is **skipped** (open mode). The `/health` endpoint always remains open.

---

## Classifier Performance

Tested with `qwen2.5:3b` on Ollama:
- **Accuracy:** 9/10 correct classifications
- **Overhead:** ~200ms average per classification
- **Misses:** Ambiguous queries ("What time is it?") — easily patched with keyword rules

---

## Monitoring

Router logs follow this format:
```
2025-01-15 03:33:00,123 [router] INFO Routing session a1b2c3d4e5f6 → simple (deepseek-v4-flash)
2025-01-15 03:33:00,456 [router] INFO Deviation: escalation keyword 'debug' matched
2025-01-15 03:33:00,789 [router] INFO Session a1b2c3d4e5f6 tier changed: simple → complex
2025-01-15 03:33:01,012 [router] WARNING Primary key rate-limited (429) - switching to alternate key
```

---

## License

MIT

## Author

James Huang + Jarvis (Hermes Agent)
