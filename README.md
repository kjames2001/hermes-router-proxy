# Hermes Router-Proxy

An intelligent LLM router that classifies prompts as "simple" or "complex" using a cheap local model (e.g., `qwen2.5:3b` on Ollama), then transparently routes them to the right cloud model.

**The idea:** Not every query needs GPT-4. Simple questions (chat, trivia, definitions) go to a fast/cheap model. Hard questions (implement, debug, deploy) go to a smart/expensive model. A small classifier model handles routing with ~200ms overhead.

## Architecture

```
User → Hermes Agent → router-proxy (localhost:8766) → classifier (qwen2.5:3b)
                                                        │
                                          ┌─────────────┴──────────┐
                                          ↓                        ↓
                                    simple model            complex model
                                  (e.g. deepseek-v4-flash)  (e.g. deepseek-v4-pro)
```

## How It Works

1. **Sessions** — After first classification, subsequent messages in the same session are cached. No re-classification needed.
2. **Keyword overrides** — "implement", "debug", "deploy", etc. always → complex. "thanks", "lol", "how are you" always → simple. These take precedence over the classifier.
3. **Fallback cascade** — If complex model returns 401, tries fallback model. If everything fails, returns 502.

## Setup

```bash
# Install
git clone https://github.com/kjames2001/hermes-router-proxy.git
cd hermes-router-proxy

# Copy and edit config
cp router_config.example.yaml router_config.yaml
# Edit router_config.yaml with your models and API keys

# Run
pip install httpx uvicorn fastapi pyyaml
python server.py
```

## Config

See `router_config.example.yaml` for full annotated configuration.

## Point Hermes At It

In `~/.hermes/config.yaml`:

```yaml
model:
  default: deepseek-v4-pro    # Must match one of your configured models
  provider: ollama-cloud
  base_url: http://localhost:8766/v1   # ← point here instead of direct API
```

Then restart the Hermes gateway.

## Classifier Performance

Tested with `qwen2.5:3b` on Ollama:
- **Accuracy:** 9/10 correct
- **Overhead:** ~200ms average per classification
- **Misses:** Ambiguous queries ("What time is it?") — easily patched with keyword rules
