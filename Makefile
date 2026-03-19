# thorllm — developer Makefile
# Usage: make <target>

SHELL      := /bin/bash
SHELL_FILES := $(shell find . -name "*.sh" ! -path "./.git/*" | sort)

.PHONY: help lint syntax shellcheck test clean install update

help: ## Show this help
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ── Code quality ──────────────────────────────────────────────────────────────

lint: syntax shellcheck ## Run all linters

syntax: ## bash -n syntax check on all .sh files
	@echo "── bash syntax check ──────────────────────────────────"
	@errors=0; \
	for f in $(SHELL_FILES); do \
	    result=$$(bash -n "$$f" 2>&1); \
	    if [[ -n "$$result" ]]; then \
	        echo "FAIL: $$f"; echo "$$result"; errors=$$((errors+1)); \
	    else \
	        echo "OK:   $$f"; \
	    fi; \
	done; \
	[[ $$errors -eq 0 ]] || exit 1

shellcheck: ## shellcheck -S warning on all .sh files
	@echo "── shellcheck ─────────────────────────────────────────"
	@shellcheck -S warning -s bash $(SHELL_FILES) && echo "All clean."

check-templates: ## Verify all expected template files exist
	@echo "── template files ─────────────────────────────────────"
	@missing=0; \
	for f in \
	    templates/vllm.env \
	    templates/vllm.service \
	    templates/vllm-serve.sh \
	    templates/activate.sh \
	    templates/models/example.yaml; do \
	    if [[ ! -f "$$f" ]]; then echo "MISSING: $$f"; missing=1; \
	    else echo "OK:      $$f"; fi; \
	done; \
	[[ $$missing -eq 0 ]] || exit 1

# ── Local usage ───────────────────────────────────────────────────────────────

install: ## Install thorllm to ~/.local (uses local repo, not GitHub)
	@REPO_URL="$$PWD" AUTO_SETUP=0 bash install.sh

install-wizard: ## Install and launch TUI wizard
	@REPO_URL="$$PWD" bash install.sh

update: ## Update vLLM to latest version
	@thorllm install --update

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean: ## Remove generated runtime files from repo root (safe to run)
	@rm -f vllm.env vllm-serve.sh activate_vllm.sh thorllm.conf *.rendered
	@echo "Cleaned generated files."
