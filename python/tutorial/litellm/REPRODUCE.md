# Reproduce: Compresr × LiteLLM — query-aware compression that cuts tokens *and* improves accuracy

A step-by-step guide to reproduce the whole demo on your own machine. Three
parts, increasing in setup cost. Part 1 takes ~2 minutes.

> This copy ships inside the SDK at `python/tutorial/litellm/`. From here, the
> editable install below is `pip install -e "../../[litellm]"` (the SDK's
> `python/` dir); the runnable `demo/` scripts sit next to this file.

## What you'll see

1. **Compression + metadata** — a bulky `web_search` tool output shrinks ~47%
   while the user's question is kept verbatim.
2. **Accuracy holds, tokens drop** — on a distractor-heavy QA set, a real
   `gpt-5-mini` answers just as accurately on the compressed context as on the
   full noisy context (100% = 100% in our run) while using **~42% fewer prompt
   tokens**. On a weaker model (e.g. the older `gpt-4o-mini`) the full noisy
   context actually *loses* accuracy, so compression there *improves* answers —
   query-aware compression never hurts and often helps. (See Part 2.)
3. **The real LiteLLM proxy** compressing in the live request path (the
   `x-litellm-applied-guardrails: compresr` header + the compressed prompt the
   upstream model actually receives).

## Prerequisites

- Python 3.9+ (3.11+ recommended).
- A **Compresr API key** (`cmp_…`) → `export COMPRESR_API_KEY=cmp_...`
- *(Parts 2 & the real-answer proxy run)* an **OpenAI key** →
  `export OPENAI_API_KEY=sk-...`
- This tutorial bundle (`python/tutorial/litellm/`) and access to the
  **Compresr SDK** source (`Compresr-SDK-Private`). Once the SDK is published
  with the integration, `pip install 'compresr[litellm]'` replaces the
  editable install below.

> Set keys via env vars (never hardcode). A `.env` next to the bundle with
> `COMPRESR_API_KEY=…` / `OPENAI_API_KEY=…` also works — the demo scripts load it.

---

## Setup (venv + install)

```bash
python -m venv .venv && source .venv/bin/activate

# Compresr SDK + the LiteLLM integration (pulls in litellm via the extra):
pip install -e "/path/to/Compresr-SDK-Private/python[litellm]"
#   ── once published, just:  pip install 'compresr[litellm]'

pip install openai            # for Parts 2 and the real-answer proxy run

export COMPRESR_API_KEY=cmp_...
export OPENAI_API_KEY=sk-...   # optional (Part 2 + real-answer run)
```

Sanity check the package imports:

```bash
python -c "from compresr.integrations.litellm import CompresrGuardrail; print('ok', CompresrGuardrail)"
```

---

## Part 1 — Compression + metadata (no proxy, fastest)

```bash
cd demo
python show_compression.py websearch_request.json
```

**Expected:** the `role: tool` web_search output goes **3396 → ~1811 chars
(~47%)**, ~368 tokens saved, and the user question + system prompt are
preserved verbatim. You'll see the `compresr_stats` metadata and the compressed
text (with `[N tokens dropped]` markers where low-relevance spans were removed).

---

## Part 2 — Accuracy: compression *improves* answers (real model)

```bash
python accuracy_eval.py        # needs OPENAI_API_KEY
```

**Expected** (illustrative — exact accuracy varies by model/run; token reduction
is stable around ~42%):

```
ACCURACY   full=10/10 (100%)   compressed=10/10 (100%)
AVG PROMPT TOKENS   full≈2901   compressed≈1669   (~42% fewer)
```

With `gpt-5-mini` both conditions answer every question correctly, so the
takeaway is "same accuracy at ~42% fewer tokens." Swap in a weaker model and the
full noisy context starts dropping items the compressed one still gets right.

It builds a 24-company knowledge base with confusingly-similar names
(e.g. *Veridian Labs* vs *Veridian Analytics*) and asks `gpt-5-mini` a specific
company's metric. The full noisy context misleads the model on one item;
query-aware compression isolates the right entity, so the compressed run is
correct **and** cheaper. (`gpt-5-mini` is a reasoning model, so the run is not
forced to `temperature=0`; minor run-to-run variation is expected.)

---

## Part 3 — The real LiteLLM proxy (compression in the request path)

The integration is a built-in LiteLLM **guardrail** named `compresr`. Until the
upstream LiteLLM PR is merged, install the in-repo discovery shim from
`python/litellm_shim/` — one command. After it merges, run the installer with
`--uninstall` and skip step 2 entirely.

### 1. Install the SDK with the litellm extra

```bash
pip install -e "/path/to/Compresr-SDK-Private/python[litellm]"
#   ── this pulls in litellm[proxy] (fastapi + uvicorn + …) as well
```

### 2. Install the discovery shim (pre-merge only)

```bash
python /path/to/Compresr-SDK-Private/python/litellm_shim/install_shim.py
```

This copies a 20-line `__init__.py` into your active litellm's
`proxy/guardrails/guardrail_hooks/compresr/` — that's the only file the proxy
needs to find the guardrail. See `python/litellm_shim/README.md` for what it
does and how to uninstall.

> Skipped on purpose: the admin-UI config model
> (`litellm/types/proxy/guardrails/guardrail_hooks/compresr.py`) and the 3
> additive edits to `litellm/types/guardrails.py` listed in earlier drafts —
> those are **admin-UI-only**. The runtime loader keys by string, so the
> guardrail works via YAML without them.

> The dotted-path custom-guardrail loader (`guardrail: some.module.Class`) does
> **not** work here — LiteLLM resolves that as a local `.py` file next to the
> config, not an installed package. Use the built-in `guardrail: compresr`
> path (the shim), which is also what ships in the PR.

### 3. Run it — no provider key needed (echo upstream)

```bash
SDK=/path/to/Compresr-SDK-Private/python

# Terminal A — a fake OpenAI-compatible upstream that records what it receives:
python "$SDK/tutorial/litellm/demo/echo_upstream.py" 8199 /tmp/forwarded_request.json

# Terminal B — the proxy with the compresr guardrail on:
litellm --config "$SDK/tutorial/litellm/demo/proxy_config.yaml" --port 4000

# Terminal C — send a web_search agent request:
curl -sD - http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-1234" \
  -H "Content-Type: application/json" \
  --data @"$SDK/tutorial/litellm/demo/websearch_request.json" | grep -i applied-guardrails
```

**Expected:** `x-litellm-applied-guardrails: compresr` on the 200 response, and
`forwarded_request.json` (what the proxy forwarded "upstream") shows the tool
output compressed ~47% with the user question intact.

### 4. (Optional) real grounded answer through OpenAI

```bash
litellm --config "$SDK/tutorial/litellm/demo/proxy_config_openai.yaml" --port 4000   # needs OPENAI_API_KEY
curl -s http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-1234" -H "Content-Type: application/json" \
  --data @"$SDK/tutorial/litellm/demo/websearch_request.json" | python -m json.tool
```

`gpt-5-mini` answers correctly off the **compressed** prompt (~513 prompt
tokens instead of the full dump).

---

## Configure it (production)

In your own proxy `config.yaml`, the guardrail block:

```yaml
guardrails:
  - guardrail_name: "compresr"
    litellm_params:
      guardrail: compresr
      mode: pre_call
      api_key: os.environ/COMPRESR_API_KEY
      api_base: http://compresr-onprem:8000   # on-prem; omit for cloud default
      default_on: true
```

Full tuning knobs (ratio, what to compress, fail-open vs fail-closed,
per-request overrides) and where the on-prem `api_base` goes are documented in
the main SDK README (the **LiteLLM proxy** section) and in
`python/litellm_shim/README.md` for the deployment/install matrix.

---

## Troubleshooting

- `ImportError: compresr LiteLLM integration requires litellm` → install with
  the extra: `pip install 'compresr[litellm]'` (or `pip install litellm`).
- Proxy starts but `x-litellm-applied-guardrails` is missing → the guardrail
  didn't run; check `default_on: true` and that `COMPRESR_API_KEY` is set.
- No compression happened (no `compresr_stats`) but the call succeeded → Compresr
  was unreachable and the guardrail **failed open** (by design). Check logs; set
  `fail_closed: true` to make it hard-fail instead. The applied-guardrails header
  reads `compresr:fail_open` in this state.
- `500 "Compresr authentication error" / "Invalid or expired API key"` → the
  guardrail ran but your `COMPRESR_API_KEY` was rejected. Auth and validation
  errors are surfaced as hard errors **regardless of `fail_closed`** (only
  network/unreachable errors fail open). Refresh the key in your env / `.env`.
- Messages shorter than `min_chars_to_compress` (default 500) are skipped on
  purpose.
