# Contributing to thorllm

## Dev setup

```bash
git clone https://github.com/ms1design/thorllm
cd thorllm

# Install locally against your local clone (no GitHub clone)
make install

# Lint everything
make lint
```

## Coding conventions

- All scripts use `#!/bin/bash` and `set -euo pipefail`
- Every script must pass `bash -n` and `shellcheck -S warning`
- Run `make lint` before every commit — CI enforces it
- Logging: use `info`, `success`, `warn`, `die` from `lib/common.sh` — never raw `echo` for user-facing output
- No hardcoded paths outside `lib/config.sh` — always use variables from config
- Template variables must be listed explicitly in the `render_template` call — never pass empty string (substitutes everything including `${PATH}`)

## Adding a new CLI command

1. Add a `case` entry in `bin/thorllm`
2. Implement the logic in the appropriate `lib/` file
3. Update `usage()` in `bin/thorllm`
4. Add a TUI page in `tui/wizard.sh` if the setting warrants interactive configuration
5. Add a smoke-test step to `.github/workflows/ci.yml`

## Adding a new template variable

1. Define the variable (with default) in `lib/config.sh`
2. Add it to `config_export()` in `lib/config.sh`
3. Add `${VAR}` to the relevant template file
4. Add `${VAR}` to the `vars` list in the `render_template` call in `lib/service.sh`
5. Add it to `thorllm.conf` generation in `config_save()`

## Template files

Templates in `templates/` are rendered with `envsubst` at install time. Rules:

- Use `${VAR}` syntax — envsubst only substitutes `${VAR}`, not `$VAR`
- Runtime variables (expanded when the service runs, not at install time) should use `${VAR}` in the *rendered* output — envsubst won't touch vars not listed in the substitution list
- Template shell scripts (`.sh`) should **not** be marked executable — they are rendered then `chmod +x` is applied to the output

## Updating the thorllm monolith (install_vllm_thor.sh → repo)

The `install_vllm_thor.sh` monolith is the predecessor to this repo. When backporting fixes:

1. Find the change in the monolith
2. Identify which `lib/` file owns that concern
3. Apply the fix there
4. Run `make lint`

## Release checklist

- [ ] `make lint` passes
- [ ] `VLLM_VERSION`, `TORCH_VERSION`, `FLASHINFER_VERSION`, `NUMBA_VERSION` verified against current vLLM release notes
- [ ] `README.md` version table updated
- [ ] Git tag: `git tag v0.x.y && git push --tags`
