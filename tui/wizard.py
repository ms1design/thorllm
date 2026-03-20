#!/usr/bin/env python3
"""
thorllm TUI wizard — Textual-based interactive setup wizard.

Design principles (per Textual docs):
  - BINDINGS on every screen → Footer renders keybindings automatically
  - on_mount focuses the first interactive widget
  - OptionList handles arrow keys + Enter natively (OptionSelected fires on Enter)
  - @work(thread=True, exclusive=True) for background tasks (HF search, version fetch)
  - app.call_from_thread() to update UI from threads
  - Result dict written to --output file (never stdout) so terminal stays free

Note: This is the TUI implementation. Use tui/wizard.sh to launch it with fallback support.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Input,
    Label,
    LoadingIndicator,
    OptionList,
    Static,
)
from textual.widgets.option_list import Option

# ─── palette ──────────────────────────────────────────────────────────────────

LOGO = """\
  ▗ ▌     ▜ ▜    
  ▜▘▛▌▛▌▛▘▐ ▐ ▛▛▌
  ▐▖▌▌▙▌▌ ▐▖▐▖▌▌▌"""

APP_CSS = """
/* ── base ── */
Screen {
    background: #0e1a00;
    color: #c8e88a;
    layers: below above;
}

/* ── logo ── */
#logo {
    color: #76b900;
    text-style: bold;
    text-align: center;
    padding: 1 0 0 0;
    width: 100%;
}

/* ── header bar ── */
#step-bar {
    dock: top;
    height: 1;
    background: #162000;
    color: #5a7a20;
    padding: 0 2;
}
#step-bar .step-title {
    width: 1fr;
    text-style: bold;
    color: #a0d832;
}
#step-bar .step-num {
    color: #5a7a20;
    text-align: right;
    width: auto;
}

/* ── content ── */
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
.field-label {
    color: #c8e88a;
    margin-top: 1;
}
.hint {
    color: #5a7a20;
    margin-bottom: 1;
}
.warn {
    color: #ffbd2e;
    text-style: italic;
    margin: 1 0;
}
.summary-row {
    height: auto;
    padding: 0;
}
.summary-key {
    width: 24;
    color: #5a7a20;
    padding: 0 1;
}
.summary-val {
    color: #76b900;
    text-style: bold;
    width: 1fr;
}

/* ── inputs ── */
Input {
    background: #162000;
    border: solid #2a3d00;
    color: #e8f5cc;
    margin-bottom: 1;
}
Input:focus {
    border: solid #76b900;
    background: #1a2700;
}

/* ── option list ── */
OptionList {
    background: #162000;
    border: solid #2a3d00;
    height: auto;
    max-height: 10;
    margin-bottom: 1;
}
OptionList:focus {
    border: solid #76b900;
}
OptionList > .option-list--option-highlighted {
    background: #2d4700;
    color: #a0d832;
    text-style: bold;
}
OptionList > .option-list--separator {
    color: #2a3d00;
}

/* ── loading ── */
LoadingIndicator {
    height: 1;
    background: transparent;
    color: #76b900;
}

/* ── footer ── */
Footer {
    background: #0e1a00;
    color: #5a7a20;
    border-top: solid #2a3d00;
}
Footer > .footer--key {
    background: #2d4700;
    color: #a0d832;
}
Footer > .footer--description {
    color: #5a7a20;
}
"""

# ─── helpers ──────────────────────────────────────────────────────────────────


def _get(url: str, timeout: int = 6) -> Any:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "thorllm/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def fetch_vllm_versions() -> tuple[str, list[str]]:
    """GitHub Releases API (stable only), PyPI fallback, then git tags fallback."""
    # Strategy 1: GitHub releases (filters prerelease/draft)
    data = _get("https://api.github.com/repos/vllm-project/vllm/releases?per_page=20")
    if data and isinstance(data, list):
        stable = [
            r["tag_name"].lstrip("v")
            for r in data
            if not r.get("prerelease") and not r.get("draft") and r.get("tag_name")
        ]
        if stable:
            return stable[0], stable[:8]

    # Strategy 2: GitHub tags (catches versions published as tags, not releases)
    tags = _get("https://api.github.com/repos/vllm-project/vllm/tags?per_page=20")
    if tags and isinstance(tags, list):
        versions = [
            t["name"].lstrip("v")
            for t in tags
            if t.get("name", "").startswith("v")
            and all(
                x not in t["name"] for x in ("rc", "dev", "a", "b", "alpha", "beta")
            )
        ]
        if versions:
            return versions[0], versions[:8]

    # Strategy 3: PyPI
    pypi = _get("https://pypi.org/pypi/vllm/json")
    if pypi:
        from packaging.version import Version

        all_v = pypi.get("releases", {}).keys()
        stable = sorted(
            [
                v
                for v in all_v
                if all(x not in v for x in ("rc", "dev", "a", "b", "post"))
            ],
            key=Version,
            reverse=True,
        )
        if stable:
            return stable[0], stable[:8]

    return "unknown", []


def fetch_torch_versions() -> tuple[str, list[str]]:
    """PyPI JSON API with proper semver sorting."""
    data = _get("https://pypi.org/pypi/torch/json")
    if not data:
        return "2.10.0", ["2.10.0", "2.9.1", "2.8.0"]
    try:
        from packaging.version import Version

        all_v = data.get("releases", {}).keys()
        stable = sorted(
            [
                v
                for v in all_v
                if all(x not in v for x in ("rc", "dev", "a", "b", "post"))
            ],
            key=Version,
            reverse=True,
        )
        return (stable[0] if stable else "2.10.0"), stable[:8]
    except Exception:
        releases = sorted(data.get("releases", {}).keys(), reverse=True)
        stable = [
            v
            for v in releases
            if all(x not in v for x in ("rc", "dev", "a", "b", "post"))
        ][:8]
        return (stable[0] if stable else "2.10.0"), stable


def hf_search(query: str, limit: int = 8) -> list[str]:
    """Search HuggingFace public model hub — no token required."""
    if not query.strip():
        return []
    url = (
        f"https://huggingface.co/api/models"
        f"?search={urllib.parse.quote(query)}&limit={limit}&sort=likes&direction=-1&full=false"
    )
    data = _get(url, timeout=5)
    if not data:
        return []
    return [m.get("id", "") for m in data if m.get("id")]


# urllib.parse is needed — import it
import urllib.parse


# ─── base screen ──────────────────────────────────────────────────────────────


class WizardScreen(Screen):
    """Base class: logo, step bar, content area, footer."""

    STEP = ""
    STEP_NUM = ""

    def compose_header(self) -> ComposeResult:
        yield Static(LOGO, id="logo")
        with Horizontal(id="step-bar"):
            yield Static(self.STEP, classes="step-title")
            yield Static(self.STEP_NUM, classes="step-num")

    def compose_footer(self) -> ComposeResult:
        yield Footer()

    def action_prev(self) -> None:
        if hasattr(self.app, "action_prev"):
            self.app.action_prev()

    def action_next(self) -> None:
        if hasattr(self.app, "action_next"):
            self.app.action_next()


# ─── screen 1: welcome ────────────────────────────────────────────────────────


class WelcomeScreen(WizardScreen):
    STEP = "Welcome"
    STEP_NUM = "1 / 7"
    BINDINGS = [
        Binding("q", "app.exit_cancel", "Cancel", show=True),
        Binding("enter,n", "next", "Next →", show=True),
    ]

    def __init__(self, version: str, build_path: str) -> None:
        super().__init__()
        self._version = version
        self._build_path = build_path

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        with Vertical(id="content"):
            yield Static(
                "thorllm — vLLM manager for NVIDIA Jetson Thor", classes="section-title"
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
                f"Config saved to: {self._build_path}/thorllm.conf"
            )
        yield from self.compose_footer()

    def on_mount(self) -> None:
        self.focus()

    def action_next(self) -> None:
        self.app.action_next()


# ─── screen 2: paths ──────────────────────────────────────────────────────────


class PathsScreen(WizardScreen):
    STEP = "Installation Paths"
    STEP_NUM = "2 / 7"
    BINDINGS = [
        Binding("q", "app.exit_cancel", "Cancel", show=True),
        Binding("enter,n", "next", "Next →", show=True),
        Binding("escape,b", "prev", "Prev ←", show=True),
    ]

    def __init__(self, build_path: str, cache_root: str) -> None:
        super().__init__()
        self._bp = build_path
        self._cr = cache_root

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        with Vertical(id="content"):
            yield Static("Installation directory  (BUILD_PATH)", classes="field-label")
            yield Static("All vLLM files, venv, and configs.", classes="hint")
            yield Input(value=self._bp, id="inp-build", placeholder="~/thorllm")
            yield Static("Cache root  (CACHE_ROOT)", classes="field-label")
            yield Static(
                "Triton / FlashInfer / HuggingFace caches. Use fastest storage (NVMe).",
                classes="hint",
            )
            yield Input(value=self._cr, id="inp-cache", placeholder="~/.cache/vllm")
            yield Static(
                "Tab  focus next field    Ctrl+N  continue    Esc  back", classes="hint"
            )
        yield from self.compose_footer()

    def on_mount(self) -> None:
        self.query_one("#inp-build", Input).focus()

    def action_next(self) -> None:
        bp = self.query_one("#inp-build", Input).value.strip() or self._bp
        cr = self.query_one("#inp-cache", Input).value.strip() or self._cr
        self.app.config["BUILD_PATH"] = bp
        self.app.config["CACHE_ROOT"] = cr
        self.app.action_next()


# ─── screen 3 & 4: version picker ─────────────────────────────────────────────


class VersionScreen(WizardScreen):
    BINDINGS = [
        Binding("q", "app.exit_cancel", "Cancel", show=True),
        Binding("enter,n", "next", "Next →", show=True),
        Binding("escape,b", "prev", "Prev ←", show=True),
    ]

    def __init__(
        self,
        step: str,
        step_num: str,
        config_key: str,
        latest: str,
        versions: list[str],
        next_idx: int,
    ) -> None:
        super().__init__()
        self.STEP = step
        self.STEP_NUM = step_num
        self._config_key = config_key
        self._latest = latest
        self._versions = versions
        self._next_idx = next_idx
        self._selected = "latest"

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        with Vertical(id="content"):
            yield Static(f"Latest detected: {self._latest}", classes="hint")
            ol = OptionList(id="ol")
            ol.add_option(Option("latest  — always install newest stable", id="latest"))
            ol.add_option(None)  # separator
            for v in self._versions:
                ol.add_option(Option(f"  v{v}", id=v))
            yield ol
            yield Static("Or enter a custom version:", classes="field-label")
            yield Input(placeholder="e.g. 0.17.1", id="inp-custom")
            yield Static(
                "↑↓ navigate    Enter select    Ctrl+N continue    Esc back",
                classes="hint",
            )
        yield from self.compose_footer()

    def on_mount(self) -> None:
        ol = self.query_one("#ol", OptionList)
        ol.focus()
        ol.highlighted = 0  # pre-select "latest" so Enter works immediately

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id:
            self._selected = event.option_id
            self.query_one("#inp-custom", Input).value = ""
            # Enter on a list item automatically advances
            self.action_next()

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        if event.option_id:
            self._selected = event.option_id
            self.query_one("#inp-custom", Input).value = ""

    def action_next(self) -> None:
        custom = self.query_one("#inp-custom", Input).value.strip()
        chosen = custom if custom else self._selected
        self.app.config[self._config_key] = "" if chosen == "latest" else chosen
        self.app.action_next()


# ─── screen 5: model (with live HF search) ────────────────────────────────────


class ModelScreen(WizardScreen):
    STEP = "Default Model"
    STEP_NUM = "5 / 7"
    BINDINGS = [
        Binding("q", "app.exit_cancel", "Cancel", show=True),
        Binding("enter,n", "next", "Next →", show=True),
        Binding("escape,b", "prev", "Prev ←", show=True),
    ]

    def __init__(self, serve_model: str) -> None:
        super().__init__()
        self._model = serve_model

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        with Vertical(id="content"):
            yield Static("Model to serve  (SERVE_MODEL)", classes="field-label")
            yield Static(
                "Format: org/model-name    e.g. openai/gpt-oss-120b", classes="hint"
            )
            yield Input(
                value=self._model, id="inp-model", placeholder="openai/gpt-oss-120b"
            )
            yield Static(
                "Live HuggingFace search  (type to search public models):",
                classes="field-label",
            )
            yield Input(placeholder="Search models…", id="inp-search")
            yield LoadingIndicator(id="spinner")
            yield OptionList(id="search-results")
            yield Static(
                "Type in search box to find models • Select to use • No token needed for public search",
                classes="hint",
            )
            yield Static("Ctrl+N  continue    Esc  back", classes="hint")
        yield from self.compose_footer()

    def on_mount(self) -> None:
        self.query_one("#spinner").display = False
        self.query_one("#inp-model", Input).focus()

    @on(Input.Changed, "#inp-search")
    def on_search_changed(self, event: Input.Changed) -> None:
        self._search_hf(event.value)

    @work(exclusive=True, thread=True)
    def _search_hf(self, query: str) -> None:
        time.sleep(0.35)  # debounce
        self.app.call_from_thread(self._set_loading, True)
        results = hf_search(query) if query.strip() else []
        self.app.call_from_thread(self._update_results, results)

    def _set_loading(self, loading: bool) -> None:
        self.query_one("#spinner").display = loading

    def _update_results(self, results: list[str]) -> None:
        self.query_one("#spinner").display = False
        ol = self.query_one("#search-results", OptionList)
        ol.clear_options()
        for r in results:
            ol.add_option(Option(r, id=r))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id:
            self.query_one("#inp-model", Input).value = event.option_id
            self.query_one("#inp-model", Input).focus()

    def action_next(self) -> None:
        val = self.query_one("#inp-model", Input).value.strip() or self._model
        self.app.config["SERVE_MODEL"] = val
        self.app.action_next()


# ─── screen 6: HF token ───────────────────────────────────────────────────────


class HFTokenScreen(WizardScreen):
    STEP = "HuggingFace Token"
    STEP_NUM = "6 / 7"
    BINDINGS = [
        Binding("q", "app.exit_cancel", "Cancel", show=True),
        Binding("enter,n", "next", "Next →", show=True),
        Binding("escape,b", "prev", "Prev ←", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield from self.compose_header()
        with Vertical(id="content"):
            yield Static("HF_TOKEN  (optional)", classes="section-title")
            yield Static(
                "Required only for gated models (Llama, Gemma, Mistral-restricted, etc).\n"
                "Leave empty to skip — add later in thorllm.conf",
                classes="hint",
            )
            yield Input(password=True, id="inp-token", placeholder="hf_…")
            yield Static("Ctrl+N  continue    Esc  back", classes="hint")
        yield from self.compose_footer()

    def on_mount(self) -> None:
        self.query_one("#inp-token", Input).focus()

    def action_next(self) -> None:
        self.app.config["HF_TOKEN"] = self.query_one("#inp-token", Input).value.strip()
        self.app.action_next()


# ─── screen 7: service user ───────────────────────────────────────────────────


class ServiceUserScreen(WizardScreen):
    STEP = "Service User"
    STEP_NUM = "7 / 7"
    BINDINGS = [
        Binding("q", "app.exit_cancel", "Cancel", show=True),
        Binding("enter,i", "summary", "Install ✓", show=True),
        Binding("escape,b", "prev", "Prev ←", show=True),
    ]

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
                "  • Add sudoers cache-drop rule:  /etc/sudoers.d/vllm-drop-caches\n"
                "  • Run systemctl daemon-reload\n\n"
                "  Password is used only for these operations and never stored.",
                classes="hint",
            )
            yield Static("Ctrl+N  review summary    Esc  back", classes="hint")
        yield from self.compose_footer()

    def on_mount(self) -> None:
        self.query_one("#inp-user", Input).focus()

    def action_summary(self) -> None:
        val = self.query_one("#inp-user", Input).value.strip() or self._user
        self.app.config["SERVICE_USER"] = val
        self.app.action_summary()


# ─── summary screen ───────────────────────────────────────────────────────────


class SummaryScreen(WizardScreen):
    STEP = "Review & Install"
    STEP_NUM = ""
    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back", show=True),
        Binding("q", "app.exit_cancel", "Cancel", show=True),
        Binding("enter,i", "install", "Install ✓", show=True),
    ]

    def compose(self) -> ComposeResult:
        cfg = self.app.config
        rows = [
            ("BUILD_PATH", cfg.get("BUILD_PATH", "")),
            ("CACHE_ROOT", cfg.get("CACHE_ROOT", "")),
            ("VLLM_VERSION", cfg.get("VLLM_VERSION") or "latest"),
            ("TORCH_VERSION", cfg.get("TORCH_VERSION") or "latest"),
            ("SERVE_MODEL", cfg.get("SERVE_MODEL", "")),
            ("SERVICE_USER", cfg.get("SERVICE_USER", "")),
            ("HF_TOKEN", "(set)" if cfg.get("HF_TOKEN") else "not set"),
        ]
        yield from self.compose_header()
        with Vertical(id="content"):
            yield Static(
                "Confirm your configuration before installation starts.", classes="hint"
            )
            for key, val in rows:
                with Horizontal(classes="summary-row"):
                    yield Static(key, classes="summary-key")
                    yield Static(val, classes="summary-val")
            yield Static(
                "\nPress Enter or I to install, Esc to go back.", classes="hint"
            )
        yield from self.compose_footer()

    def on_mount(self) -> None:
        self.focus()

    def action_install(self) -> None:
        self.app.exit(self.app.config)


# ─── model select screen ──────────────────────────────────────────────────────


class ModelSelectScreen(Screen):
    BINDINGS = [
        Binding("escape,q", "app.exit_cancel", "Cancel", show=True),
    ]

    def __init__(self, models: list[str], active: str, version: str) -> None:
        super().__init__()
        self._models = models
        self._active = active
        self._version = version

    def compose(self) -> ComposeResult:
        yield Static(LOGO, id="logo")
        with Vertical(id="content"):
            yield Static("Active Model", classes="section-title")
            yield Static("↑↓ navigate    Enter select    Esc cancel", classes="hint")
            ol = OptionList(id="ol")
            for m in self._models:
                prefix = "● " if m == self._active else "  "
                ol.add_option(Option(f"{prefix}{m}", id=m))
            yield ol
        yield Footer()

    def on_mount(self) -> None:
        ol = self.query_one("#ol", OptionList)
        ol.focus()
        if self._active in self._models:
            ol.highlighted = self._models.index(self._active)
        elif self._models:
            ol.highlighted = 0

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.app.exit({"selected": event.option_id})


# ─── apps ─────────────────────────────────────────────────────────────────────


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
        # Build wizard screen list once
        d = defaults
        bp = d.get("BUILD_PATH", os.path.expanduser("~/thorllm"))
        cr = d.get("CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
        su = d.get("SERVICE_USER", os.environ.get("USER", "ubuntu"))
        self._ws = [
            WelcomeScreen(version, bp),  # 0
            PathsScreen(bp, cr),  # 1
            VersionScreen(
                "vLLM Version",
                "3 / 7",  # 2
                "VLLM_VERSION",
                vllm_latest,
                vllm_versions,
                next_idx=3,
            ),
            VersionScreen(
                "PyTorch Version",
                "4 / 7",  # 3
                "TORCH_VERSION",
                torch_latest,
                torch_versions,
                next_idx=4,
            ),
            ModelScreen(d.get("SERVE_MODEL", "openai/gpt-oss-120b")),  # 4
            HFTokenScreen(),  # 5
            ServiceUserScreen(su),  # 6
            # SummaryScreen is created on-demand (it reads config at compose time)
        ]

    def on_mount(self) -> None:
        self.push_screen(self._ws[0])

    def action_exit_cancel(self) -> None:
        self.exit(None)

    def action_next(self) -> None:
        current_screen = self.screen
        if isinstance(current_screen, WizardScreen):
            screen_idx = None
            for i, s in enumerate(self._ws):
                if s is current_screen:
                    screen_idx = i
                    break
            if screen_idx is not None and screen_idx < len(self._ws) - 1:
                self.push_screen(self._ws[screen_idx + 1])

    def action_prev(self) -> None:
        current_screen = self.screen
        if isinstance(current_screen, WizardScreen):
            screen_idx = None
            for i, s in enumerate(self._ws):
                if s is current_screen:
                    screen_idx = i
                    break
            if screen_idx is not None and screen_idx > 0:
                self.pop_screen()
                self.push_screen(self._ws[screen_idx - 1])

    def action_summary(self) -> None:
        self.push_screen(SummaryScreen())


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


# ─── entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="thorllm TUI wizard")
    parser.add_argument("--mode", choices=["wizard", "model-select"], default="wizard")
    parser.add_argument("--defaults", default="{}")
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--models", default="[]")
    parser.add_argument("--active", default="")
    parser.add_argument(
        "--output",
        default="",
        help="Write result JSON to file (keeps stdout free for TUI)",
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
        # Small pause so the user sees the fetched versions before TUI takes over
        time.sleep(0.5)

        result = WizardApp(
            defaults=defaults,
            version=version,
            vllm_latest=vllm_latest,
            vllm_versions=vllm_versions,
            torch_latest=torch_latest,
            torch_versions=torch_versions,
        ).run()
    else:
        models = json.loads(args.models)
        result = ModelSelectApp(models, args.active, version).run()

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
