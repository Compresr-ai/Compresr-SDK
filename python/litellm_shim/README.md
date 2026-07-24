# LiteLLM Discovery Shim

A bridge that makes the Compresr guardrail discoverable by the LiteLLM proxy
**until the upstream LiteLLM PR is merged**. After that lands, none of this is
needed.

## Three ways to wire it up

Pick one — they all achieve the same thing.

### 1. `compresr-litellm` CLI wrapper (no site-packages writes)

```bash
pip install 'compresr[litellm]'
compresr-litellm --config /path/to/proxy_config.yaml --port 4000
```

Drop-in for the stock `litellm` CLI, same flags. Registers the guardrail
in-process before starting the proxy.

### 2. `install-compresr-shim` console script (matches a built-in install)

```bash
pip install 'compresr[litellm]'
install-compresr-shim                 # install
install-compresr-shim --uninstall     # remove
```

Copies a tiny discovery file into your active litellm install so plain
`litellm --config ...` finds the guardrail. Equivalent to:

```bash
python python/litellm_shim/install_shim.py
```

(`install_shim.py` here just delegates to the packaged installer.)

### 3. `COMPRESR_AUTO_INSTALL_SHIM=1` env var

```bash
export COMPRESR_AUTO_INSTALL_SHIM=1
python -c "import compresr.integrations.litellm"  # installer runs on import
```

Best for Docker images and ephemeral containers: set the env, import once at
container-start time, then run the proxy normally.

## Proxy config

```yaml
guardrails:
  - guardrail_name: "compresr"
    litellm_params:
      guardrail: compresr
      mode: pre_call
      api_key: os.environ/COMPRESR_API_KEY
      default_on: true
```

## What's intentionally NOT shipped

- **Admin-UI config model** (`litellm/types/proxy/guardrails/guardrail_hooks/compresr.py`)
  and the 3 enum edits to `litellm/types/guardrails.py` — both admin-UI-only.
  The runtime loader keys by string, so the guardrail works via YAML without them.

If you need the admin UI, follow Part 3 Step 2 in `python/tutorial/litellm/REPRODUCE.md`.
