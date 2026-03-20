#!/usr/bin/env python3
"""
thorllm — Textual-based interactive setup wizard
Outputs selected config as JSON to stdout on completion.
"""

from __future__ import annotations

import asyncio
import json
import sys
import os
import urllib.request
import urllib.error
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center, Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import Screen, ModalScreen
from textual.widgets import (
    Button, Footer, Header, Input, Label, ListItem,
    ListView, OptionList, Select, Static, RadioButton, RadioSet,
)
from textual.widgets.option_list import Option
from textual.reactive import reactive
from textual.css.query import NoMatches

# ── NVIDIA Green palette ──────────────────────────────────────────────────────
CSS = """
$nvidia-green: #76b900;
$nvidia-dark:  #1a2700;
$nvidia-mid:   #2d4700;
$nvidia-light: #a0d832;
$surface:      #0e1a00;
$panel:        #162000;
$border:       #2a3d00;
$text:         #c8e88a;
$muted:        #5a7a20;
$white:        #e8f5cc;
$error:        #ff5f57;
$warning:      #ffbd2e;

Screen {
    background: $surface;
    color: $text;
}

/* ── Logo ── */
#logo-box {
    width: 100%;
    height: auto;
    content-align: center middle;
    padding: 1 0 0 0;
}
.logo {
    color: $nvidia-green;
    text-style: bold;
    text-align: center;
}

/* ── Footer strip ── */
#app-footer {
    dock: bottom;
    width: 100%;
    height: 1;
    background: $nvidia-dark;
    color: $muted;
    content-align: center middle;
    text-align: center;
}

/* ── Page containers ── */
.page {
    width: 100%;
    height: 1fr;
    padding: 1 4;
    overflow-y: auto;
}
.page-title {
    color: $nvidia-green;
    text-style: bold;
    padding-bottom: 1;
    border-bottom: solid $border;
    margin-bottom: 1;
    width: 100%;
}
.section-title {
    color: $nvidia-light;
    text-style: bold;
    margin-top: 1;
}
.hint {
    color: $muted;
    margin-bottom: 1;
}
.field-label {
    color: $text;
    margin-top: 1;
    margin-bottom: 0;
}
.value-display {
    color: $nvidia-green;
    text-style: bold;
}

/* ── Input fields ── */
Input {
    background: $panel;
    border: solid $border;
    color: $white;
    margin-bottom: 1;
}
Input:focus {
    border: solid $nvidia-green;
}

/* ── Buttons ── */
Button {
    background: $nvidia-mid;
    color: $nvidia-light;
    border: solid $border;
    margin: 0 1;
}
Button:hover, Button:focus {
    background: $nvidia-green;
    color: $nvidia-dark;
    border: solid $nvidia-light;
}
Button.-primary {
    background: $nvidia-green;
    color: $nvidia-dark;
    text-style: bold;
    border: solid $nvidia-light;
}
Button.-primary:hover {
    background: $nvidia-light;
}
Button.-destructive {
    background: $nvidia-dark;
    color: $error;
    border: solid $error;
}

/* ── Button rows ── */
.btn-row {
    height: auto;
    align: center middle;
    margin-top: 1;
    padding-top: 1;
    border-top: solid $border;
}

/* ── Version list / OptionList ── */
OptionList {
    background: $panel;
    border: solid $border;
    height: auto;
    max-height: 12;
    scrollbar-color: $nvidia-green;
}
OptionList:focus {
    border: solid $nvidia-green;
}
OptionList > .option-list--option-highlighted {
    background: $nvidia-mid;
    color: $nvidia-light;
    text-style: bold;
}
OptionList > .option-list--option-selected {
    color: $nvidia-green;
    text-style: bold;
}

/* ── Summary table ── */
.summary-row {
    height: auto;
    padding: 0 0;
}
.summary-key {
    width: 28;
    color: $muted;
}
.summary-val {
    color: $nvidia-green;
    text-style: bold;
}

/* ── Welcome screen ── */
.welcome-box {
    border: solid $border;
    background: $panel;
    padding: 1 3;
    margin: 1 0;
}
.welcome-text {
    color: $text;
}

/* ── Password field ── */
.password-hint {
    color: $warning;
    text-style: italic;
    margin-top: 0;
    margin-bottom: 1;
}

/* ── Divider ── */
.divider {
    color: $border;
    width: 100%;
    margin: 1 0;
}
"""

LOGO = """\
▗ ▌     ▜ ▜    
▜▘▛▌▛▌▛▘▐ ▐ ▛▛▌
▐▖▌▌▙▌▌ ▐▖▐▖▌▌▌"""

FOOTER_TEXT = "thorllm · v{version} · made by ms1design"

# ── Small utilities ────────────────────────────────────────────────────────────

def _fetch_json(url: str, timeout: int = 6) -> Any:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "thorllm/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def fetch_vllm_versions() -> tuple[str, list[str]]:
    """Return (latest, [recent...]) vLLM versions."""
    latest_data = _fetch_json(
        "https://api.github.com/repos/vllm-project/vllm/releases/latest"
    )
    latest = (latest_data or {}).get("tag_name", "").lstrip("v") or "unknown"

    recent_data = _fetch_json(
        "https://api.github.com/repos/vllm-project/vllm/releases?per_page=6"
    ) or []
    versions = [r.get("tag_name", "").lstrip("v") for r in recent_data if r.get("tag_name")]
    return latest, versions[:5]


def fetch_torch_versions() -> tuple[str, list[str]]:
    """Return (latest, [recent...]) PyTorch versions available on cu130 index."""
    # PyPI simple index for torch
    data = _fetch_json("https://pypi.org/pypi/torch/json")
    if not data:
        return "2.10.0", ["2.10.0", "2.9.1", "2.8.0"]
    releases = sorted(data.get("releases", {}).keys(), reverse=True)
    # Filter to stable (no rc/dev/a/b) and recent
    stable = [v for v in releases if all(x not in v for x in ("rc", "dev", "a", "b", "post"))][:5]
    latest = stable[0] if stable else "2.10.0"
    return latest, stable


# ── Reusable widgets ───────────────────────────────────────────────────────────

class LogoWidget(Static):
    def __init__(self) -> None:
        super().__init__(LOGO, classes="logo")


class AppFooterBar(Static):
    def __init__(self, version: str) -> None:
        super().__init__(FOOTER_TEXT.format(version=version), id="app-footer")


class PageTitle(Static):
    def __init__(self, text: str) -> None:
        super().__init__(text, classes="page-title")


# ── Screens ───────────────────────────────────────────────────────────────────

class WelcomeScreen(Screen):
    BINDINGS = [Binding("enter", "next", "Continue"), Binding("q", "quit", "Quit")]

    def __init__(self, version: str, build_path: str) -> None:
        super().__init__()
        self.version = version
        self.build_path = build_path

    def compose(self) -> ComposeResult:
        yield LogoWidget()
        with ScrollableContainer(classes="page"):
            yield PageTitle("Welcome to thorllm setup")
            with Container(classes="welcome-box"):
                yield Static(
                    "thorllm — vLLM manager for NVIDIA Jetson Thor\n\n"
                    "This wizard will configure your installation.\n\n"
                    "You will be asked about:\n"
                    "  • Installation and cache directories\n"
                    "  • vLLM version to install\n"
                    "  • PyTorch version to install\n"
                    "  • Default model to serve\n"
                    "  • HuggingFace token (optional)\n"
                    "  • systemd service user\n\n"
                    f"Settings are saved to {self.build_path}/thorllm.conf\n"
                    "and can be changed by re-running: thorllm setup",
                    classes="welcome-text",
                )
            with Horizontal(classes="btn-row"):
                yield Button("Quit", variant="default", id="btn-quit", classes="-destructive")
                yield Button("Continue →", variant="primary", id="btn-next", classes="-primary")

    @on(Button.Pressed, "#btn-next")
    def action_next(self) -> None:
        self.app.push_screen("paths")

    @on(Button.Pressed, "#btn-quit")
    def action_quit(self) -> None:
        self.app.exit(None)


class PathsScreen(Screen):
    def __init__(self, build_path: str, cache_root: str) -> None:
        super().__init__()
        self._build_path = build_path
        self._cache_root = cache_root

    def compose(self) -> ComposeResult:
        yield LogoWidget()
        with ScrollableContainer(classes="page"):
            yield PageTitle("Installation Paths")
            yield Static("Installation directory  (BUILD_PATH)", classes="field-label")
            yield Static(
                "All vllm files, venv, and model configs will be stored here.",
                classes="hint",
            )
            yield Input(value=self._build_path, id="inp-build", placeholder="~/thorllm")
            yield Static("Cache root directory  (CACHE_ROOT)", classes="field-label")
            yield Static(
                "Triton / FlashInfer / HuggingFace caches. Use fastest storage (NVMe if available).",
                classes="hint",
            )
            yield Input(value=self._cache_root, id="inp-cache", placeholder="~/.cache/vllm")
            with Horizontal(classes="btn-row"):
                yield Button("← Back", id="btn-back")
                yield Button("Next →", variant="primary", id="btn-next", classes="-primary")

    @on(Button.Pressed, "#btn-next")
    def go_next(self) -> None:
        build = self.query_one("#inp-build", Input).value.strip() or self._build_path
        cache = self.query_one("#inp-cache", Input).value.strip() or self._cache_root
        self.app.config["BUILD_PATH"] = build
        self.app.config["CACHE_ROOT"] = cache
        self.app.push_screen("vllm_version")

    @on(Button.Pressed, "#btn-back")
    def go_back(self) -> None:
        self.app.pop_screen()


class VersionScreen(Screen):
    """Generic version-picker screen for vLLM or PyTorch."""

    _fetching: reactive[bool] = reactive(True)
    _latest: reactive[str] = reactive("…")
    _versions: list[str] = []

    def __init__(
        self,
        title: str,
        description: str,
        default_version: str,
        fetch_fn,
        config_key: str,
        next_screen: str,
    ) -> None:
        super().__init__()
        self._title = title
        self._description = description
        self._default_version = default_version
        self._fetch_fn = fetch_fn
        self._config_key = config_key
        self._next_screen = next_screen
        self._selected: str = ""

    def compose(self) -> ComposeResult:
        yield LogoWidget()
        with ScrollableContainer(classes="page"):
            yield PageTitle(self._title)
            yield Static(self._description, classes="hint")
            yield Static("Fetching available versions…", id="fetch-status", classes="hint")
            yield OptionList(id="version-list")
            yield Input(
                placeholder="Or type a custom version (e.g. 2.10.0)",
                id="inp-custom",
            )
            with Horizontal(classes="btn-row"):
                yield Button("← Back", id="btn-back")
                yield Button("Next →", variant="primary", id="btn-next", classes="-primary")

    def on_mount(self) -> None:
        self._load_versions()

    @work(thread=True)
    def _load_versions(self) -> None:
        latest, versions = self._fetch_fn()
        self.call_from_thread(self._populate, latest, versions)

    def _populate(self, latest: str, versions: list[str]) -> None:
        self._latest = latest
        self._versions = versions
        try:
            status = self.query_one("#fetch-status", Static)
            status.update(f"Latest detected: [bold green]{latest}[/bold green]")
            ol = self.query_one("#version-list", OptionList)
            ol.clear_options()
            ol.add_option(Option("latest  (always install latest stable)", id="latest"))
            ol.add_option(None)  # separator
            for v in versions:
                ol.add_option(Option(f"v{v}", id=v))
            # pre-select default
            target = self._default_version or "latest"
            for idx, v in enumerate(["latest"] + versions):
                if v == target:
                    ol.highlighted = idx
                    break
        except NoMatches:
            pass

    @on(OptionList.OptionHighlighted, "#version-list")
    def _on_highlight(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_id:
            self._selected = event.option_id
            self.query_one("#inp-custom", Input).value = ""

    @on(Input.Changed, "#inp-custom")
    def _on_custom_input(self, event: Input.Changed) -> None:
        if event.value.strip():
            self._selected = event.value.strip()

    @on(Button.Pressed, "#btn-next")
    def go_next(self) -> None:
        custom = self.query_one("#inp-custom", Input).value.strip()
        chosen = custom or self._selected
        if chosen == "latest":
            chosen = ""
        self.app.config[self._config_key] = chosen
        self.app.push_screen(self._next_screen)

    @on(Button.Pressed, "#btn-back")
    def go_back(self) -> None:
        self.app.pop_screen()


class ModelScreen(Screen):
    def __init__(self, serve_model: str) -> None:
        super().__init__()
        self._serve_model = serve_model

    def compose(self) -> ComposeResult:
        yield LogoWidget()
        with ScrollableContainer(classes="page"):
            yield PageTitle("Default Model")
            yield Static(
                "Default model to serve  (SERVE_MODEL)\n"
                "Format: <org>/<model-name>  e.g. openai/gpt-oss-120b",
                classes="hint",
            )
            yield Input(value=self._serve_model, id="inp-model", placeholder="openai/gpt-oss-120b")
            yield Static(
                "A YAML config will be created at:\n"
                "  ${BUILD_PATH}/models/<org>/<name>.yaml",
                classes="hint",
            )
            with Horizontal(classes="btn-row"):
                yield Button("← Back", id="btn-back")
                yield Button("Next →", variant="primary", id="btn-next", classes="-primary")

    @on(Button.Pressed, "#btn-next")
    def go_next(self) -> None:
        val = self.query_one("#inp-model", Input).value.strip() or self._serve_model
        self.app.config["SERVE_MODEL"] = val
        self.app.push_screen("hf_token")

    @on(Button.Pressed, "#btn-back")
    def go_back(self) -> None:
        self.app.pop_screen()


class HFTokenScreen(Screen):
    def compose(self) -> ComposeResult:
        yield LogoWidget()
        with ScrollableContainer(classes="page"):
            yield PageTitle("HuggingFace Token  (optional)")
            yield Static(
                "Required only for gated models (Llama, Gemma, etc).\n"
                "Leave empty to skip — you can add it later in thorllm.conf",
                classes="hint",
            )
            yield Input(password=True, id="inp-token", placeholder="hf_…")
            with Horizontal(classes="btn-row"):
                yield Button("← Back", id="btn-back")
                yield Button("Next →", variant="primary", id="btn-next", classes="-primary")

    @on(Button.Pressed, "#btn-next")
    def go_next(self) -> None:
        val = self.query_one("#inp-token", Input).value.strip()
        self.app.config["HF_TOKEN"] = val
        self.app.push_screen("service_user")

    @on(Button.Pressed, "#btn-back")
    def go_back(self) -> None:
        self.app.pop_screen()


class ServiceUserScreen(Screen):
    def __init__(self, service_user: str) -> None:
        super().__init__()
        self._service_user = service_user

    def compose(self) -> ComposeResult:
        yield LogoWidget()
        with ScrollableContainer(classes="page"):
            yield PageTitle("Service User")
            yield Static(
                "User account to run the vLLM systemd service.\n"
                "This user needs GPU access and read permission on the model files.",
                classes="hint",
            )
            yield Input(value=self._service_user, id="inp-user", placeholder="username")

            yield Static(
                "⚠  Sudo password required", classes="section-title"
            )
            yield Static(
                "thorllm needs root access to:\n"
                "  • Install the systemd service unit  (/etc/systemd/system/vllm.service)\n"
                "  • Create a sudoers rule for cache-drop  (/etc/sudoers.d/vllm-drop-caches)\n"
                "  • Run systemctl daemon-reload\n\n"
                "Your password is used only for these system-level operations and is\n"
                "not stored anywhere.",
                classes="password-hint",
            )
            with Horizontal(classes="btn-row"):
                yield Button("← Back", id="btn-back")
                yield Button("Next →", variant="primary", id="btn-next", classes="-primary")

    @on(Button.Pressed, "#btn-next")
    def go_next(self) -> None:
        val = self.query_one("#inp-user", Input).value.strip() or self._service_user
        self.app.config["SERVICE_USER"] = val
        self.app.push_screen("summary")

    @on(Button.Pressed, "#btn-back")
    def go_back(self) -> None:
        self.app.pop_screen()


class SummaryScreen(Screen):
    def compose(self) -> ComposeResult:
        cfg = self.app.config
        vllm_ver = cfg.get("VLLM_VERSION") or "latest"
        torch_ver = cfg.get("TORCH_VERSION") or "latest"
        hf = "(set)" if cfg.get("HF_TOKEN") else "not set"

        rows = [
            ("BUILD_PATH",    cfg.get("BUILD_PATH", "~")),
            ("CACHE_ROOT",    cfg.get("CACHE_ROOT", "~/.cache/vllm")),
            ("VLLM_VERSION",  vllm_ver),
            ("TORCH_VERSION", torch_ver),
            ("SERVE_MODEL",   cfg.get("SERVE_MODEL", "")),
            ("SERVICE_USER",  cfg.get("SERVICE_USER", "")),
            ("HF_TOKEN",      hf),
        ]

        yield LogoWidget()
        with ScrollableContainer(classes="page"):
            yield PageTitle("── Configuration Summary")
            yield Static(
                "Review your settings before installation starts.\n"
                "Press Install to save config and begin, or Back to change anything.",
                classes="hint",
            )
            for key, val in rows:
                with Horizontal(classes="summary-row"):
                    yield Static(f"{key}", classes="summary-key")
                    yield Static(val, classes="summary-val")
            with Horizontal(classes="btn-row"):
                yield Button("← Back", id="btn-back")
                yield Button("✓ Install", variant="primary", id="btn-install", classes="-primary")
                yield Button("✗ Cancel", id="btn-cancel", classes="-destructive")

    @on(Button.Pressed, "#btn-install")
    def do_install(self) -> None:
        self.app.exit(self.app.config)

    @on(Button.Pressed, "#btn-back")
    def go_back(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#btn-cancel")
    def do_cancel(self) -> None:
        self.app.exit(None)


class ModelSelectScreen(Screen):
    """Interactive model selection (for `thorllm model select`)."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, models: list[str], active: str) -> None:
        super().__init__()
        self._models = models
        self._active = active

    def compose(self) -> ComposeResult:
        yield LogoWidget()
        with ScrollableContainer(classes="page"):
            yield PageTitle("Select Active Model")
            yield Static(
                "Use arrow keys to highlight, Enter to select.",
                classes="hint",
            )
            ol = OptionList(id="model-list")
            for m in self._models:
                marker = "●" if m == self._active else "○"
                label = f"{marker}  {m}"
                ol.add_option(Option(label, id=m))
            yield ol
            with Horizontal(classes="btn-row"):
                yield Button("Cancel", id="btn-cancel", classes="-destructive")
                yield Button("✓ Select", variant="primary", id="btn-select", classes="-primary")

    @on(Button.Pressed, "#btn-select")
    def do_select(self) -> None:
        ol = self.query_one("#model-list", OptionList)
        opt = ol.highlighted_option
        if opt is not None:
            self.app.exit({"selected": opt.id})
        else:
            self.app.exit(None)

    @on(Button.Pressed, "#btn-cancel")
    def action_cancel(self) -> None:
        self.app.exit(None)


# ── App ────────────────────────────────────────────────────────────────────────

class ThorllmWizard(App):
    CSS = CSS
    TITLE = "thorllm setup"

    def __init__(self, defaults: dict, version: str, mode: str = "wizard") -> None:
        super().__init__()
        self.defaults = defaults
        self.version = version
        self.mode = mode
        self.config: dict = {}

    def on_mount(self) -> None:
        # Footer bar
        self.screen.mount(AppFooterBar(self.version))
        if self.mode == "wizard":
            self._install_wizard_screens()
        # model select mode is handled separately

    def _install_wizard_screens(self) -> None:
        d = self.defaults
        bp = d.get("BUILD_PATH", os.path.expanduser("~/thorllm"))
        cr = d.get("CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
        su = d.get("SERVICE_USER", os.environ.get("USER", "ubuntu"))
        sm = d.get("SERVE_MODEL", "openai/gpt-oss-120b")
        vv = d.get("VLLM_VERSION", "")
        tv = d.get("TORCH_VERSION", "2.10.0")

        self.install_screen(WelcomeScreen(self.version, bp), name="welcome")
        self.install_screen(PathsScreen(bp, cr), name="paths")
        self.install_screen(
            VersionScreen(
                title="vLLM Version",
                description="Select vLLM version to install.\nChoose 'latest' to always install the newest stable release.",
                default_version=vv,
                fetch_fn=fetch_vllm_versions,
                config_key="VLLM_VERSION",
                next_screen="torch_version",
            ),
            name="vllm_version",
        )
        self.install_screen(
            VersionScreen(
                title="PyTorch Version",
                description="Select PyTorch version to install.\nMust have a cu130 build at download.pytorch.org.",
                default_version=tv,
                fetch_fn=fetch_torch_versions,
                config_key="TORCH_VERSION",
                next_screen="model",
            ),
            name="torch_version",
        )
        self.install_screen(ModelScreen(sm), name="model")
        self.install_screen(HFTokenScreen(), name="hf_token")
        self.install_screen(ServiceUserScreen(su), name="service_user")
        self.install_screen(SummaryScreen(), name="summary")
        self.push_screen("welcome")


class ThorllmModelSelect(App):
    CSS = CSS
    TITLE = "thorllm — model select"

    def __init__(self, models: list[str], active: str, version: str) -> None:
        super().__init__()
        self._models = models
        self._active = active
        self.version = version

    def on_mount(self) -> None:
        self.screen.mount(AppFooterBar(self.version))
        self.push_screen(ModelSelectScreen(self._models, self._active))


# ── Entry points ───────────────────────────────────────────────────────────────

def run_wizard(defaults: dict, version: str) -> dict | None:
    app = ThorllmWizard(defaults, version, mode="wizard")
    return app.run()


def run_model_select(models: list[str], active: str, version: str) -> dict | None:
    app = ThorllmModelSelect(models, active, version)
    return app.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="thorllm TUI")
    parser.add_argument("--mode", choices=["wizard", "model-select"], default="wizard")
    parser.add_argument("--defaults", default="{}")
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--models", default="[]")
    parser.add_argument("--active", default="")
    args = parser.parse_args()

    defaults = json.loads(args.defaults)
    version = args.version

    if args.mode == "wizard":
        result = run_wizard(defaults, version)
    else:
        models = json.loads(args.models)
        result = run_model_select(models, args.active, version)

    if result:
        print(json.dumps(result))
        sys.exit(0)
    else:
        sys.exit(1)
