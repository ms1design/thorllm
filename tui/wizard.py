#!/usr/bin/env python3
"""
thorllm TUI wizard — Textual-based interactive setup wizard.

Architecture:
  wizard.py   The actual TUI.  Receives config as --defaults JSON,
              writes result to --output JSON.  Never touches stdout during
              TUI mode (so Textual can own the terminal fully).

  wizard.sh   Bash bridge that serialises shell vars → JSON, calls this
              file, reads result JSON back into shell vars.  No UI here.

Textual patterns used:
  - Button(variant="primary"|"error"|"default"|"success") for built-in styling
  - on_mount: focus the primary action button so ENTER works naturally
  - @on(Button.Pressed, "#id") per-button handlers — clean and explicit
  - Screen.pop_screen() for Prev navigation
  - @work(thread=True) for background network calls (HF search, versions)
  - app.call_from_thread() to update UI from worker threads
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Input,
    Label,
    LoadingIndicator,
    OptionList,
    Static,
)
from textual.widgets.option_list import Option

# ─── Logo ─────────────────────────────────────────────────────────────────────

LOGO = """\

  ▗ ▌     ▜ ▜    
  ▜▘▛▌▛▌▛▘▐ ▐ ▛▛▌
  ▐▖▌▌▙▌▌ ▐▖▐▖▌▌▌"""

# ─── CSS ──────────────────────────────────────────────────────────────────────
# Button colours come from Textual's built-in variant system:
#   variant="primary"  → accent blue  (Next)
#   variant="success"  → green        (Install / Select)
#   variant="error"    → red          (Cancel — universally understood as "stop")
#   variant="default"  → neutral      (Prev / secondary)
# We never recolour buttons via custom CSS to avoid invisible-text bugs.

APP_CSS = """
Screen {
    background: #0e1a00;
    color: #c8e88a;
}

/* ── Logo ── */
#logo {
    color: #76b900;
    text-style: bold;
    text-align: center;
    padding: 1 0 0 0;
    width: 100%;
}

/* ── Step bar ── */
#step-bar {
    dock: top;
    height: 1;
    background: #162000;
    color: #5a7a20;
    padding: 0 2;
}
.step-title {
    width: 1fr;
    text-style: bold;
    color: #a0d832;
}
.step-num {
    color: #5a7a20;
    text-align: right;
    width: auto;
}

/* ── Content ── */
#content {
    padding: 1 3;
    overflow-y: auto;
    height: 1fr;
}
.section-title {
    color: #a0d832;
    text-style: bold;
    margin-bottom: 1;
    border-bottom: solid #2a3d00;
    padding-bottom: 1;
    width: 100%;
}
.field-label { color: #c8e88a; margin-top: 1; }
.hint        { color: #5a7a20; margin-bottom: 1; }
.warn        { color: #ffbd2e; text-style: italic; margin: 1 0; }

/* ── Summary ── */
.summary-row { height: auto; padding: 0; }
.summary-key { width: 24; color: #5a7a20; padding: 0 1; }
.summary-val { color: #76b900; text-style: bold; width: 1fr; }

/* ── Input ── */
Input {
    background: #162000;
    border: solid #2a3d00;
    color: #e8f5cc;
    margin-bottom: 1;
}
Input:focus { border: solid #76b900; background: #1a2700; }

/* ── Option list ── */
OptionList {
    background: #162000;
    border: solid #2a3d00;
    height: auto;
    max-height: 10;
    margin-bottom: 1;
}
OptionList:focus { border: solid #76b900; }
OptionList > .option-list--option-highlighted {
    background: #2d4700;
    color: #a0d832;
    text-style: bold;
}
OptionList > .option-list--separator { color: #2a3d00; }

/* ── Spinner ── */
LoadingIndicator {
    height: 1;
    background: transparent;
    color: #76b900;
}

/* ── Navigation bar ── */
#nav {
    dock: bottom;
    height: 6;
    background: #0a1200;
    border-top: solid #2a3d00;
    padding: 1 2;
}
.nav-left  { width: 1fr; height: 3; align: left middle; }
.nav-right { width: auto; height: 3; align: right middle; }

/* Explicit button colours — Textual's built-in variant palette does not
   contrast against this dark green theme, so we override every state here. */
#nav Button {
    margin: 0 1;
    min-width: 14;
    background: #162000;
    border: tall #2a3d00;
    color: #c8e88a;
}
#nav Button:hover {
    background: #1e3200;
    border: tall #3a5a00;
    color: #a0d832;
}
#nav Button:focus {
    background: #1e3200;
    border: tall #4a7200;
    color: #a0d832;
    text-style: bold;
}

/* primary  → Next → */
#nav Button.-primary {
    background: #1a3800;
    border: tall #76b900;
    color: #b8e060;
    text-style: bold;
}
#nav Button.-primary:hover { background: #253f00; color: #c8e88a; }
#nav Button.-primary:focus { background: #2d4700; border: tall #a0d832; color: #d4f080; }

/* success  → ✓ Install / ✓ Select */
#nav Button.-success {
    background: #0d2800;
    border: tall #5ab900;
    color: #80d840;
    text-style: bold;
}
#nav Button.-success:hover { background: #183800; color: #a8e878; }
#nav Button.-success:focus { background: #1f4000; border: tall #76b900; color: #c8e88a; }

/* error    → ✕ Cancel / ✕ Exit */
#nav Button.-error {
    background: #280800;
    border: tall #803000;
    color: #d06040;
    text-style: bold;
}
#nav Button.-error:hover { background: #3a1000; color: #ffbd2e; }
#nav Button.-error:focus { background: #3a1000; border: tall #a04020; color: #ffbd2e; }

/* default  → ← Prev / secondary actions */
#nav Button.-default {
    background: #111a00;
    border: tall #2a3d00;
    color: #8ab840;
}
#nav Button.-default:hover { background: #1a2800; color: #a0d832; }
#nav Button.-default:focus { background: #1e3200; border: tall #3a5a00; color: #b8e060; }
"""

# ─── Network helpers ──────────────────────────────────────────────────────────


def _get(url: str, timeout: int = 6) -> Any:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "thorllm/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def fetch_vllm_versions() -> tuple[str, list[str]]:
    data = _get("https://api.github.com/repos/vllm-project/vllm/releases?per_page=20")
    if data and isinstance(data, list):
        stable = [
            r["tag_name"].lstrip("v")
            for r in data
            if not r.get("prerelease") and not r.get("draft") and r.get("tag_name")
        ]
        if stable:
            return stable[0], stable[:8]

    tags = _get("https://api.github.com/repos/vllm-project/vllm/tags?per_page=20")
    if tags and isinstance(tags, list):
        versions = [
            t["name"].lstrip("v")
            for t in tags
            if t.get("name", "").startswith("v")
            and all(x not in t["name"] for x in ("rc", "dev", "a", "b", "alpha", "beta"))
        ]
        if versions:
            return versions[0], versions[:8]

    pypi = _get("https://pypi.org/pypi/vllm/json")
    if pypi:
        try:
            from packaging.version import Version

            all_v = pypi.get("releases", {}).keys()
            stable = sorted(
                [v for v in all_v if all(x not in v for x in ("rc", "dev", "a", "b", "post"))],
                key=Version,
                reverse=True,
            )
            if stable:
                return stable[0], stable[:8]
        except Exception:
            pass
    return "unknown", []


def fetch_torch_versions() -> tuple[str, list[str]]:
    data = _get("https://pypi.org/pypi/torch/json")
    if not data:
        return "2.10.0", ["2.10.0", "2.9.1", "2.8.0"]
    try:
        from packaging.version import Version

        all_v = data.get("releases", {}).keys()
        stable = sorted(
            [v for v in all_v if all(x not in v for x in ("rc", "dev", "a", "b", "post"))],
            key=Version,
            reverse=True,
        )
        return (stable[0] if stable else "2.10.0"), stable[:8]
    except Exception:
        releases = sorted(data.get("releases", {}).keys(), reverse=True)
        stable = [
            v for v in releases
            if all(x not in v for x in ("rc", "dev", "a", "b", "post"))
        ][:8]
        return (stable[0] if stable else "2.10.0"), stable


def hf_search(query: str, limit: int = 8) -> list[str]:
    if not query.strip():
        return []
    url = (
        f"https://huggingface.co/api/models"
        f"?search={urllib.parse.quote(query)}&limit={limit}"
        f"&sort=likes&direction=-1&full=false"
    )
    data = _get(url, timeout=5)
    if not data:
        return []
    return [m.get("id", "") for m in data if m.get("id")]


# ─── Base screen ──────────────────────────────────────────────────────────────


class WizardScreen(Screen):
    """Common base: logo, step-bar header, and nav button bar.

    Navigation pattern:
      • Buttons are the primary UI — each has a clear label and variant colour.
      • on_mount focuses the primary button (or first input) so ENTER always
        does the expected thing without any custom key binding magic.
      • The Escape key pops the screen (goes back) as a universal shortcut.
      • 'q' quits / cancels everywhere.
    """

    STEP: str = ""
    STEP_NUM: str = ""

    # Subclass flags
    _show_prev: bool = True
    _show_install: bool = False  # True → green "✓ Install" instead of blue "Next →"

    BINDINGS = [
        Binding("escape", "go_prev", "Back", show=False),
        Binding("q", "cancel", "Cancel", show=False),
    ]

    # ── Shared layout pieces ──────────────────────────────────────────────────

    def compose_header(self) -> ComposeResult:
        yield Static(LOGO, id="logo")
        with Horizontal(id="step-bar"):
            yield Static(self.STEP, classes="step-title")
            yield Static(self.STEP_NUM, classes="step-num")

    def compose_nav(self) -> ComposeResult:
        """Bottom nav bar: [✕ Cancel] (spacer) [← Prev] [Next → / ✓ Install]"""
        with Horizontal(id="nav"):
            with Horizontal(classes="nav-left"):
                yield Button("✕  Cancel", id="btn-cancel", variant="error")
            with Horizontal(classes="nav-right"):
                if self._show_prev:
                    yield Button("←  Prev", id="btn-prev", variant="default")
                if self._show_install:
                    yield Button("✓  Install", id="btn-install", variant="success")
                else:
                    yield Button("Next  →", id="btn-next", variant="primary")

    # ── Button handlers ───────────────────────────────────────────────────────

    @on(Button.Pressed, "#btn-cancel")
    def _press_cancel(self) -> None:
        self.app.action_exit_cancel()

    @on(Button.Pressed, "#btn-prev")
    def _press_prev(self) -> None:
        self.action_go_prev()

    @on(Button.Pressed, "#btn-next")
    def _press_next(self) -> None:
        self.action_go_next()

    @on(Button.Pressed, "#btn-install")
    def _press_install(self) -> None:
        self.action_go_install()

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_cancel(self) -> None:
        self.app.action_exit_cancel()

    def action_go_prev(self) -> None:
        self.app.pop_screen()

    def action_go_next(self) -> None:
        """Override in subclasses to save data before advancing."""
        self.app.action_next()

    def action_go_install(self) -> None:
        """Override in subclasses for the final install action."""
        self.app.action_install()


# ─── Screen 1 — Welcome ───────────────────────────────────────────────────────


class WelcomeScreen(WizardScreen):
    STEP = "Welcome"
    STEP_NUM = "1 / 7"
    _show_prev = False

    def __init__(self, version: str, build_path: str) -> None:
        super().__init__()
        self._version = version
        self._build_path = build_path

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        with Vertical(id="content"):
            yield Static(
                "thorllm — vLLM manager for NVIDIA Jetson Thor",
                classes="section-title",
            )
            yield Static(
                "This wizard will guide you through installation setup.\n\n"
                "You will configure:\n"
                "  • Installation and cache paths\n"
                "  • vLLM version  (fetched from GitHub)\n"
                "  • PyTorch version  (fetched from PyPI)\n"
                "  • Default model to serve\n"
                "  • HuggingFace token  (optional, for gated models)\n"
                "  • systemd service user\n\n"
                f"Config will be saved to:  {self._build_path}/thorllm.conf",
            )
        yield from self.compose_nav()

    def on_mount(self) -> None:
        # No inputs — focus Next directly so ENTER advances the wizard
        self.query_one("#btn-next", Button).focus()


# ─── Screen 2 — Paths ─────────────────────────────────────────────────────────


class PathsScreen(WizardScreen):
    STEP = "Installation Paths"
    STEP_NUM = "2 / 7"

    def __init__(self, build_path: str, cache_root: str) -> None:
        super().__init__()
        self._bp = build_path
        self._cr = cache_root

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        with Vertical(id="content"):
            yield Static("Installation directory  (BUILD_PATH)", classes="field-label")
            yield Static("vLLM files, venv, model configs.", classes="hint")
            yield Input(value=self._bp, id="inp-build", placeholder="~/thorllm")
            yield Static("Cache root  (CACHE_ROOT)", classes="field-label")
            yield Static(
                "Triton / FlashInfer / HuggingFace caches. Use fastest NVMe storage.",
                classes="hint",
            )
            yield Input(value=self._cr, id="inp-cache", placeholder="~/.cache/vllm")
            yield Static("Tab between fields  •  click Next when done", classes="hint")
        yield from self.compose_nav()

    def on_mount(self) -> None:
        self.query_one("#inp-build", Input).focus()

    def action_go_next(self) -> None:
        bp = self.query_one("#inp-build", Input).value.strip() or self._bp
        cr = self.query_one("#inp-cache", Input).value.strip() or self._cr
        self.app.config["BUILD_PATH"] = bp
        self.app.config["CACHE_ROOT"] = cr
        self.app.action_next()


# ─── Screens 3 & 4 — Version pickers ─────────────────────────────────────────


class VersionScreen(WizardScreen):

    def __init__(
        self,
        step: str,
        step_num: str,
        config_key: str,
        latest: str,
        versions: list[str],
    ) -> None:
        super().__init__()
        self.STEP = step
        self.STEP_NUM = step_num
        self._config_key = config_key
        self._latest = latest
        self._versions = versions
        self._selected = "latest"

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        with Vertical(id="content"):
            yield Static(f"Latest available: {self._latest}", classes="hint")
            ol = OptionList(id="ol")
            ol.add_option(Option("latest  — always install newest stable", id="latest"))
            ol.add_option(None)
            for v in self._versions:
                ol.add_option(Option(f"  v{v}", id=v))
            yield ol
            yield Static("Or enter a specific version:", classes="field-label")
            yield Input(placeholder="e.g. 0.17.1", id="inp-custom")
            yield Static(
                "↑↓ to browse  •  Enter in the list to select & continue  "
                "•  or fill the field and click Next",
                classes="hint",
            )
        yield from self.compose_nav()

    def on_mount(self) -> None:
        ol = self.query_one("#ol", OptionList)
        ol.focus()
        ol.highlighted = 0

    @on(OptionList.OptionHighlighted, "#ol")
    def _highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_id:
            self._selected = event.option_id
            self.query_one("#inp-custom", Input).value = ""

    @on(OptionList.OptionSelected, "#ol")
    def _list_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id:
            self._selected = event.option_id
            self.query_one("#inp-custom", Input).value = ""
            self.action_go_next()

    def action_go_next(self) -> None:
        custom = self.query_one("#inp-custom", Input).value.strip()
        chosen = custom if custom else self._selected
        self.app.config[self._config_key] = "" if chosen == "latest" else chosen
        self.app.action_next()


# ─── Screen 5 — Model ─────────────────────────────────────────────────────────


class ModelScreen(WizardScreen):
    STEP = "Default Model"
    STEP_NUM = "5 / 7"

    def __init__(self, serve_model: str) -> None:
        super().__init__()
        self._model = serve_model

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        with Vertical(id="content"):
            yield Static("Model to serve  (SERVE_MODEL)", classes="field-label")
            yield Static("Format: org/model-name  e.g. openai/gpt-oss-120b", classes="hint")
            yield Input(value=self._model, id="inp-model", placeholder="openai/gpt-oss-120b")
            yield Static(
                "Live HuggingFace search (type below to search public models):",
                classes="field-label",
            )
            yield Input(placeholder="Search models…", id="inp-search")
            yield LoadingIndicator(id="spinner")
            yield OptionList(id="search-results")
            yield Static(
                "Select a result above to copy it into the model field",
                classes="hint",
            )
        yield from self.compose_nav()

    def on_mount(self) -> None:
        self.query_one("#spinner").display = False
        self.query_one("#inp-model", Input).focus()

    @on(Input.Changed, "#inp-search")
    def _search_changed(self, event: Input.Changed) -> None:
        self._do_search(event.value)

    @work(exclusive=True, thread=True)
    def _do_search(self, query: str) -> None:
        time.sleep(0.35)
        self.app.call_from_thread(self._set_spinner, True)
        results = hf_search(query) if query.strip() else []
        self.app.call_from_thread(self._update_results, results)

    def _set_spinner(self, visible: bool) -> None:
        self.query_one("#spinner").display = visible

    def _update_results(self, results: list[str]) -> None:
        self.query_one("#spinner").display = False
        ol = self.query_one("#search-results", OptionList)
        ol.clear_options()
        for r in results:
            ol.add_option(Option(r, id=r))

    @on(OptionList.OptionSelected, "#search-results")
    def _pick_result(self, event: OptionList.OptionSelected) -> None:
        if event.option_id:
            self.query_one("#inp-model", Input).value = event.option_id
            self.query_one("#inp-model", Input).focus()

    def action_go_next(self) -> None:
        val = self.query_one("#inp-model", Input).value.strip() or self._model
        self.app.config["SERVE_MODEL"] = val
        self.app.action_next()


# ─── Screen 6 — HF Token ─────────────────────────────────────────────────────


class HFTokenScreen(WizardScreen):
    STEP = "HuggingFace Token"
    STEP_NUM = "6 / 7"

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        with Vertical(id="content"):
            yield Static("HF_TOKEN  (optional)", classes="section-title")
            yield Static(
                "Required only for gated models (Llama, Gemma, Mistral-restricted…).\n"
                "Leave empty to skip — you can add it later in thorllm.conf.",
                classes="hint",
            )
            yield Input(password=True, id="inp-token", placeholder="hf_…")
        yield from self.compose_nav()

    def on_mount(self) -> None:
        self.query_one("#inp-token", Input).focus()

    def action_go_next(self) -> None:
        self.app.config["HF_TOKEN"] = self.query_one("#inp-token", Input).value.strip()
        self.app.action_next()


# ─── Screen 7 — Service user ──────────────────────────────────────────────────


class ServiceUserScreen(WizardScreen):
    STEP = "Service User"
    STEP_NUM = "7 / 7"
    _show_install = True   # "Next →" becomes "Next →" leading to SummaryScreen

    def __init__(self, service_user: str) -> None:
        super().__init__()
        self._user = service_user

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        with Vertical(id="content"):
            yield Static("User to run the vLLM systemd service", classes="field-label")
            yield Input(value=self._user, id="inp-user", placeholder="username")
            yield Static("  Why sudo is needed:", classes="warn")
            yield Static(
                "  • Install systemd unit:  /etc/systemd/system/vllm.service\n"
                "  • Add sudoers rule:  /etc/sudoers.d/vllm-drop-caches\n"
                "  • Run systemctl daemon-reload\n\n"
                "  Password is only used for these steps and is never stored.",
                classes="hint",
            )
        yield from self.compose_nav()

    def on_mount(self) -> None:
        self.query_one("#inp-user", Input).focus()

    def _commit(self) -> None:
        val = self.query_one("#inp-user", Input).value.strip() or self._user
        self.app.config["SERVICE_USER"] = val

    def action_go_next(self) -> None:
        self._commit()
        self.app.action_summary()

    def action_go_install(self) -> None:
        self._commit()
        self.app.action_summary()


# ─── Summary / confirm screen ─────────────────────────────────────────────────


class SummaryScreen(WizardScreen):
    STEP = "Review & Install"
    STEP_NUM = "✓"
    _show_install = True

    def compose(self) -> ComposeResult:
        cfg = self.app.config
        rows = [
            ("BUILD_PATH",    cfg.get("BUILD_PATH", "")),
            ("CACHE_ROOT",    cfg.get("CACHE_ROOT", "")),
            ("VLLM_VERSION",  cfg.get("VLLM_VERSION") or "latest"),
            ("TORCH_VERSION", cfg.get("TORCH_VERSION") or "latest"),
            ("SERVE_MODEL",   cfg.get("SERVE_MODEL", "")),
            ("SERVICE_USER",  cfg.get("SERVICE_USER", "")),
            ("HF_TOKEN",      "(set)" if cfg.get("HF_TOKEN") else "not set"),
        ]
        yield from self.compose_header()
        with Vertical(id="content"):
            yield Static(
                "Review your configuration before installation starts.",
                classes="hint",
            )
            for key, val in rows:
                with Horizontal(classes="summary-row"):
                    yield Static(key, classes="summary-key")
                    yield Static(val, classes="summary-val")
            yield Static("\nPress Enter or click  ✓ Install  to begin.", classes="hint")
        yield from self.compose_nav()

    def on_mount(self) -> None:
        # Focus Install so ENTER triggers it immediately
        self.query_one("#btn-install", Button).focus()

    def action_go_prev(self) -> None:
        self.app.pop_screen()

    def action_go_install(self) -> None:
        self.app.exit(self.app.config)


# ─── Model select screen ──────────────────────────────────────────────────────


class ModelSelectScreen(WizardScreen):
    """Pick the active model — reuses WizardScreen's layout."""

    STEP = "Select Active Model"
    _show_prev = False

    def compose_nav(self) -> ComposeResult:  # type: ignore[override]
        with Horizontal(id="nav"):
            with Horizontal(classes="nav-left"):
                yield Button("✕  Cancel", id="btn-cancel", variant="error")
            with Horizontal(classes="nav-right"):
                yield Button("✓  Select", id="btn-select", variant="success")

    def __init__(self, models: list[str], active: str, version: str) -> None:
        super().__init__()
        self._models = models
        self._active = active
        self._version = version
        self.STEP_NUM = f"{len(models)} model(s)"

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        with Vertical(id="content"):
            yield Static("Active Model", classes="section-title")
            yield Static(
                "↑↓ navigate  •  Enter in list to select  •  or click  ✓ Select",
                classes="hint",
            )
            ol = OptionList(id="ol")
            for m in self._models:
                prefix = "● " if m == self._active else "  "
                ol.add_option(Option(f"{prefix}{m}", id=m))
            yield ol
        yield from self.compose_nav()

    def on_mount(self) -> None:
        ol = self.query_one("#ol", OptionList)
        ol.focus()
        if self._active in self._models:
            ol.highlighted = self._models.index(self._active)
        elif self._models:
            ol.highlighted = 0

    def _commit(self) -> None:
        ol = self.query_one("#ol", OptionList)
        if ol.highlighted is not None and 0 <= ol.highlighted < len(self._models):
            self.app.exit({"selected": self._models[ol.highlighted]})

    @on(Button.Pressed, "#btn-select")
    def _press_select(self) -> None:
        self._commit()

    @on(OptionList.OptionSelected, "#ol")
    def _list_selected(self, event: OptionList.OptionSelected) -> None:
        self.app.exit({"selected": event.option_id})


# ─── Model management screen ──────────────────────────────────────────────────


class ModelManagementScreen(WizardScreen):
    """Full model manager: list, add, switch, show, edit — reuses WizardScreen."""

    STEP = "Model Management"
    _show_prev = False

    def compose_nav(self) -> ComposeResult:  # type: ignore[override]
        with Horizontal(id="nav"):
            with Horizontal(classes="nav-left"):
                yield Button("✕  Exit", id="btn-exit", variant="error")
            with Horizontal(classes="nav-right"):
                yield Button("＋ Add", id="btn-add", variant="default")
                yield Button("⇄ Switch", id="btn-switch", variant="primary")
                yield Button("⊞ Show", id="btn-show", variant="default")
                yield Button("✎ Edit", id="btn-edit", variant="default")

    def __init__(self, models: list[str], active: str, version: str) -> None:
        super().__init__()
        self._models = models
        self._active = active
        self._version = version
        self.STEP_NUM = f"{len(models)} model(s)"

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        with Vertical(id="content"):
            yield Static("Model configurations", classes="section-title")
            yield Static(
                "↑↓ navigate  •  select a model  •  then use the action buttons below",
                classes="hint",
            )
            ol = OptionList(id="ol")
            if self._models:
                for m in self._models:
                    prefix = "● " if m == self._active else "  "
                    ol.add_option(Option(f"{prefix}{m}", id=m))
            else:
                ol.add_option(
                    Option("  (no models yet — use  ＋ Add  below)", id="__none__")
                )
            yield ol
            yield Static("Add a new model (org/name):", classes="field-label")
            yield Input(
                placeholder="e.g. Qwen/Qwen3-VL-32B-Instruct", id="inp-add"
            )
        yield from self.compose_nav()

    def on_mount(self) -> None:
        ol = self.query_one("#ol", OptionList)
        ol.focus()
        if self._active in self._models:
            ol.highlighted = self._models.index(self._active)
        elif self._models:
            ol.highlighted = 0

    def _selected_model(self) -> str | None:
        ol = self.query_one("#ol", OptionList)
        if ol.highlighted is not None and 0 <= ol.highlighted < len(self._models):
            return self._models[ol.highlighted]
        return None

    @on(Button.Pressed, "#btn-exit")
    def _exit(self) -> None:
        self.app.exit(None)

    @on(Button.Pressed, "#btn-add")
    def _add(self) -> None:
        val = self.query_one("#inp-add", Input).value.strip()
        if val:
            self.app.exit({"action": "add", "model": val})
        else:
            self.query_one("#inp-add", Input).focus()

    @on(Button.Pressed, "#btn-switch")
    def _switch(self) -> None:
        m = self._selected_model()
        if m:
            self.app.exit({"action": "switch", "model": m})

    @on(Button.Pressed, "#btn-show")
    def _show_model(self) -> None:
        m = self._selected_model()
        if m:
            self.app.exit({"action": "show", "model": m})

    @on(Button.Pressed, "#btn-edit")
    def _edit(self) -> None:
        m = self._selected_model()
        if m:
            self.app.exit({"action": "edit", "model": m})


# ─── Apps ─────────────────────────────────────────────────────────────────────


class WizardApp(App[dict | None]):
    CSS = APP_CSS

    def __init__(
        self,
        defaults: dict,
        version: str,
        vllm_latest: str,
        vllm_versions: list[str],
        torch_latest: str,
        torch_versions: list[str],
    ) -> None:
        super().__init__()
        self.config: dict = dict(defaults)
        self._version = version
        d = defaults
        bp = d.get("BUILD_PATH", os.path.expanduser("~/thorllm"))
        cr = d.get("CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
        su = d.get("SERVICE_USER", os.environ.get("USER", "ubuntu"))

        self._screens: list[WizardScreen] = [
            WelcomeScreen(version, bp),
            PathsScreen(bp, cr),
            VersionScreen("vLLM Version",   "3 / 7", "VLLM_VERSION",  vllm_latest,  vllm_versions),
            VersionScreen("PyTorch Version", "4 / 7", "TORCH_VERSION", torch_latest, torch_versions),
            ModelScreen(d.get("SERVE_MODEL", "openai/gpt-oss-120b")),
            HFTokenScreen(),
            ServiceUserScreen(su),
        ]
        self._idx = 0

    def on_mount(self) -> None:
        self.push_screen(self._screens[0])

    def action_exit_cancel(self) -> None:
        self.exit(None)

    def action_next(self) -> None:
        if self._idx < len(self._screens) - 1:
            self._idx += 1
            self.push_screen(self._screens[self._idx])

    def action_summary(self) -> None:
        self.push_screen(SummaryScreen())

    def action_install(self) -> None:
        self.exit(self.config)


class ModelSelectApp(App[dict | None]):
    CSS = APP_CSS

    def __init__(self, models: list[str], active: str, version: str) -> None:
        super().__init__()
        self._models = models
        self._active = active
        self._version = version

    def on_mount(self) -> None:
        self.push_screen(ModelSelectScreen(self._models, self._active, self._version))

    def action_exit_cancel(self) -> None:
        self.exit(None)


class ModelManagementApp(App[dict | None]):
    CSS = APP_CSS

    def __init__(self, models: list[str], active: str, version: str) -> None:
        super().__init__()
        self._models = models
        self._active = active
        self._version = version

    def on_mount(self) -> None:
        self.push_screen(
            ModelManagementScreen(self._models, self._active, self._version)
        )

    def action_exit_cancel(self) -> None:
        self.exit(None)


# ─── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="thorllm TUI wizard")
    parser.add_argument(
        "--mode",
        choices=["wizard", "model-select", "model-manage"],
        default="wizard",
    )
    parser.add_argument("--defaults", default="{}")
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--models", default="[]")
    parser.add_argument("--active", default="")
    parser.add_argument(
        "--output",
        default="",
        help="Write result JSON to this file (keeps stdout free for Textual)",
    )
    args = parser.parse_args()

    defaults = json.loads(args.defaults)
    version = args.version

    if args.mode == "wizard":
        print("Fetching vLLM releases…", flush=True)
        vllm_latest, vllm_versions = fetch_vllm_versions()
        print(f"  latest vLLM: {vllm_latest}", flush=True)
        print("Fetching PyTorch releases…", flush=True)
        torch_latest, torch_versions = fetch_torch_versions()
        print(f"  latest PyTorch: {torch_latest}", flush=True)
        time.sleep(0.5)

        result = WizardApp(
            defaults=defaults,
            version=version,
            vllm_latest=vllm_latest,
            vllm_versions=vllm_versions,
            torch_latest=torch_latest,
            torch_versions=torch_versions,
        ).run()

    elif args.mode == "model-select":
        models = json.loads(args.models)
        result = ModelSelectApp(models, args.active, version).run()

    else:  # model-manage
        models = json.loads(args.models)
        result = ModelManagementApp(models, args.active, version).run()

    if result is None:
        sys.exit(1)

    payload = json.dumps(result)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(payload)
    else:
        print(payload)
    sys.exit(0)


if __name__ == "__main__":
    main()
