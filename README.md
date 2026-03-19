# thorllm

vLLM manager for **NVIDIA Jetson Thor** (aarch64 / SM 11.0a / CUDA 13.0).

Installs vLLM from a pre-compiled `cu130` wheel (falls back to source build), sets up a
systemd service, and provides a CLI + TUI for day-to-day management.

---

## Quick start

```bash
# On a fresh Jetson Thor with JetPack 7.x:

# 1. System prerequisites (once)
sudo apt-get install -y cuda-toolkit-13-0 python3-dev python3.12-dev \
    jq curl git wget whiptail gettext-base

# 2. Install thorllm
curl -fsSL https://raw.githubusercontent.com/ms1design/thorllm/main/install.sh | bash
# → clones repo, symlinks CLI to ~/.local/bin/thorllm, launches TUI wizard

# 3. Manage
thorllm start
thorllm logs -f
thorllm status
```

---

## CLI reference

```
thorllm <command> [options]

Install:
  setup                  Interactive TUI wizard (whiptail)
  install                Non-interactive install (uses saved config)
  install --update       Update vLLM to the latest release

Service:
  start                  sudo systemctl start vllm
  stop                   sudo systemctl stop vllm
  restart                sudo systemctl restart vllm
  status                 systemctl status vllm
  logs [-f]              journalctl -u vllm [-f]

Models:
  model list             List all model configs (active one is highlighted)
  model add <org/name>   Create a new model config from template
  model switch <org/name> Switch the active model (edits vllm.env + optionally restarts)
  model show <org/name>  Print the model YAML
  model edit <org/name>  Open the model YAML in $EDITOR

Info:
  version                Show installed vLLM and dependency versions
  config                 Show current configuration
```

---

## File layout

After installation:

```
~/.local/
  bin/thorllm              ← CLI symlink
  share/thorllm/           ← repo clone (source of truth for templates)

~/thorllm/                 ← BUILD_PATH (default)
  thorllm.conf             ← persisted config
  vllm.env                   ← systemd EnvironmentFile (chmod 600)
  vllm-serve.sh              ← systemd ExecStart launcher
  activate_vllm.sh           ← source this for manual shell use
  .vllm/                     ← Python venv
  models/
    <org>/<model>.yaml       ← per-model vLLM config
    example/gpt-oss-120b.yaml

~/.cache/vllm/               ← CACHE_ROOT (default)
  huggingface/               ← model weights
  flashinfer/
  triton/
  kernels/
  ...
```

---

## Switching models

```bash
# Create config for a new model
thorllm model add Qwen/Qwen3-VL-32B-Instruct

# Edit the YAML (set gpu_memory_utilization, max_model_len, etc.)
thorllm model edit Qwen/Qwen3-VL-32B-Instruct

# Switch and restart
thorllm model switch Qwen/Qwen3-VL-32B-Instruct
```

---

## Setting HF_TOKEN

Never store your token in a world-readable file. Use a systemd override:

```bash
sudo systemctl edit vllm
# add:
# [Service]
# Environment=HF_TOKEN=hf_xxxxxxxxxxxx
```

---

## Per-model YAML reference

```yaml
model: openai/gpt-oss-120b
port: 8000
served_model_name:
  - openai/gpt-oss-120b
  - gpt-oss-120b

tensor_parallel_size: 1
gpu_memory_utilization: 0.90
max_model_len: 32768

attention_backend: FLASHINFER
enable_chunked_prefill: true
enable_prefix_caching: true

# quantization: fp8
# kv_cache_dtype: fp8

# reasoning_parser: qwen3
# speculative_config:
#   method: qwen3_next_mtp
#   num_speculative_tokens: 2
```

Hardcoded launcher args (set in `templates/vllm-serve.sh`, applied to every model):

| Arg | Value |
|---|---|
| `--max-num-seqs` | `4` |
| `--cpu-offload-gb` | `0` |
| `--swap-space` | `0` |
| `--host` | `0.0.0.0` |
| `--disable-fastapi-docs` | on |
| `--enable-force-include-usage` | on |
| `--download-dir` | `${CACHE_ROOT}/huggingface` |

---

## First-start note

The first `thorllm start` after install will appear frozen for **5–15 minutes** while
Triton and FlashInfer JIT-compile CUDA kernels for `sm_110a`. The compiled artifacts are
cached in `${CACHE_ROOT}/triton` and `${CACHE_ROOT}/flashinfer`, so every subsequent start
is fast. Make sure `CACHE_ROOT` is on persistent storage.

---

## Environment variables

All variables have sane defaults and can be overridden before running any command:

| Variable | Default | Description |
|---|---|---|
| `BUILD_PATH` | `~/thorllm` | Installation root |
| `CACHE_ROOT` | `~/.cache/vllm` | All cache directories |
| `SERVE_MODEL` | `openai/gpt-oss-120b` | Active model |
| `CUDA_HOME` | `/usr/local/cuda` | CUDA installation |
| `SERVICE_USER` | `$(whoami)` | systemd service user |
| `VLLM_VERSION` | *(latest)* | Pin a specific version |
| `HF_TOKEN` | *(empty)* | HuggingFace access token |

---

## Development

```bash
git clone https://github.com/ms1design/thorllm
cd thorllm
bash -n bin/thorllm lib/*.sh tui/*.sh   # syntax check
shellcheck -S warning bin/thorllm lib/*.sh tui/*.sh
```

---

## License

MIT
