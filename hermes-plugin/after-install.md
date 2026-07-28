# Compresr for Hermes — next steps

1. **Install the SDK** in the same Python environment that runs Hermes:

   ```bash
   pip install compresr
   ```

2. **Enable the plugin** (plugins are opt-in):

   ```bash
   hermes plugins enable compresr
   ```

3. **Pick what to turn on** in `~/.hermes/config.yaml`:

   ```yaml
   # Context compaction via Compresr instead of an auxiliary-LLM summary:
   context:
     engine: compresr

   # Per-turn compression of large tool outputs (originals stay recoverable):
   compresr:
     tool_output_enabled: true
   ```

Both features are fail-open: on any API error Hermes falls back to its
built-in behavior and never loses context. Check live stats any time with the
`/compresr` slash command.

Config reference and tuning knobs (model, ratios, thresholds):
https://github.com/Compresr-ai/Compresr-SDK/tree/main/hermes-plugin#configuration
