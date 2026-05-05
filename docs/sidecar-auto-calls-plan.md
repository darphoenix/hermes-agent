# Sidecar Auto Calls And Main Telegram Sharing

## Goal

Keep the main local wrapper on `127.0.0.1:1236` as the interactive model shared by Hermes CLI/TUI and Telegram gateway, while routing automatic/background helper calls to a second loaded wrapper on `127.0.0.1:1237`.

This gives fast switching between the two real user surfaces without letting background review, `/background`, or gateway maintenance calls interrupt the foreground decode stream. Worst case, both loaded copies contend for compute and the foreground slows down; best case, only the helper lane waits.

## Runtime Layout

- Main wrapper: `1236`, stateful Responses enabled, large response store and 128 GiB paged prefix cache.
- Sidecar wrapper: `1237`, separate response store and separate paged prefix cache directory.
- Telegram gateway: uses the same main runtime as CLI/TUI for normal user turns.
- Auto/background calls: use `background_runtime` when enabled for their task class.

The two wrappers must not write to the same response-store file or paged-prefix-cache directory. Shared cache directories would create confusing cache ownership and possible corruption when both processes rotate state.

## Config Surface

`~/.hermes/config.yaml` gets:

```yaml
background_runtime:
  enabled: true
  provider: custom
  model: /Users/karolis/.lmstudio/models/dealignai/Qwen3.6-27B-JANG_4M-CRACK
  base_url: http://127.0.0.1:1237/v1
  api_key: mlx-key
  api_mode: codex_responses
  responses_stateful: false
  responses_stateful_for:
    background_review: true
  fallback_to_main: false
  use_for:
    background_review: true
    cli_background: true
    gateway_background: true
    gateway_hygiene: true
    auxiliary: false
```

`auxiliary: false` is deliberate for the first pass. Compression/title/search helper calls can be moved later, but starting with explicit background work avoids surprising failures if the sidecar is not running.

## Routing Rules

- Normal CLI/TUI turns: unchanged, use the main runtime.
- Normal Telegram turns: unchanged, use the same main runtime.
- End-of-turn background memory/skill review: sidecar when configured, with a sidecar-local stateful Responses chain. Foreground response IDs are scrubbed from inherited history before the review runs.
- CLI `/background`: sidecar when configured.
- Telegram/gateway `/background`: sidecar when configured.
- Gateway hygiene compression: sidecar when configured, stateless Responses.
- Auxiliary tasks: supported by the router, disabled by current config.

If a task is enabled for the sidecar and the sidecar runtime is incomplete, Hermes fails that side task instead of silently running it on the main model.

## Fast Session Switching Improvements

The main-wrapper stateful cache remains the source of speed for active user sessions. The next useful improvements are:

- Keep gateway and CLI/TUI on one main wrapper unless they are used concurrently.
- Keep `responses_stateful` enabled only for foreground sessions that benefit from stored state.
- Use separate response-store files for main and sidecar so each wrapper can rotate cache entries independently.
- Increase main `--response-store-size` when switching among many active Telegram/CLI sessions.
- Keep sidecar `responses_stateful` off by default; enable it per helper task only after that path has its own sidecar-local chain hygiene.

## Operational Commands

Main:

```bash
/Users/karolis/mlx-openai-wrapper/start_mlx_wrapper.sh
```

Sidecar:

```bash
/Users/karolis/mlx-openai-wrapper/start_mlx_wrapper_sidecar.sh
```

The sidecar launcher reuses the same model by default, but uses port `1237` and a sidecar cache profile.
