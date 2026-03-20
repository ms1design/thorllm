#!/usr/bin/env python3
"""
thorllm TUI wizard — built with Textual (textualize.io)

Correct architecture:
  - All version data is fetched BEFORE the TUI starts (no background threads)
  - App pushes Screen instances (never install_screen / string names)
  - Each Screen uses compose() — never on_mount mounting
  - self.app.exit(result) returns directly from run()
  - Result dict is written to --output file; stdout/stderr are untouched so
    the terminal is fully available to Textual
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from typing import Any

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.screen import Screen
from textual.widgets import Button, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

# ── Palette ───────────────────────────────────────────────────────────────────

LOGO = """\
  ▗ ▌     ▜ ▜    
  ▜▘▛▌▛▌▛▘▐ ▐ ▛▛▌
  ▐▖▌▌▙▌▌ ▐▖▐▖▌▌▌"""

CSS = """
$ng:   #76b900;   /* NVIDIA green     */
$nl:   #a0d832;   /* NVIDIA light     */
$nd:   #1a2700;   /* NVIDIA dark bg   */
$nm:   #2d4700;   /* NVIDIA mid       */
$sur:  #0e1a00;   /* surface          */
$pan:  #162000;   /* panel            */
$bdr:  #2a3d00;   /* border           */
$txt:  #c8e88a;   /* body text        */
$mut:  #5a7a20;   /* muted            */
$wht:  #e8f5cc;   /* bright text      */
$err:  #ff5f57;   /* error red        */
$wrn:  #ffbd2e;   /* warning yellow   */

Screen {
    background: $sur;
    color: $txt;
}

/* ── Logo ── */
.logo {
    width: 100%;
    color: $ng;
    text-style: bold;
    text-align: center;
    padding: 1 0 0 0;
}

/* ── Footer ── */
.footer {
    width: 100%;
    color: $mut;
    text-align: center;
    padding: 1 0;
    border-top: solid $bdr;
}

/* ── Page wrapper ── */
.page {
    width: 100%;
    height: 1fr;
    padding: 1 4;
}

/* ── Text helpers ── */
.title {
    color: $ng;
    text-style: bold;
    border-bottom: solid $bdr;
    padding-bottom: 1;
    margin-bottom: 1;
    width: 100%;
}
.hint {
    color: $mut;
    margin-bottom: 1;
}
.label {
    color: $txt;
    margin-top: 1;
}
.warn-text {
    color: $wrn;
    text-style: italic;
    margin-bottom: 1;
}
.summary-key {
    width: 26;
    color: $mut;
    padding: 0 1;
}
.summary-val {
    color: $ng;
    text-style: bold;
}

/* ── Inputs ── */
Input {
    background: $pan;
    border: solid $bdr;
    color: $wht;
    margin-bottom: 1;
}
Input:focus {
    border: solid $ng;
}

/* ── Buttons ── */
Button {
    background: $nm;
    color: $nl;
    border: solid $bdr;
    margin: 0 1;
    min-width: 12;
}
Button:hover, Button:focus {
    background: $ng;
    color: $nd;
    border: solid $nl;
}
Button.-primary {
    background: $ng;
    color: $nd;
    text-style: bold;
    border: solid $nl;
}
Button.-primary:hover {
    background: $nl;
}
Button.-error {
    color: $err;
    border: solid $err;
    background: $nd;
}

/* ── Button row ── */
.btn-row {
    height: auto;
    align: center middle;
    padding-top: 1;
    border-top: solid $bdr;
    margin-top: 1;
}

/* ── OptionList ── */
OptionList {
    background: $pan;
    border: solid $bdr;
    max-height: 10;
}
OptionList:focus {
    border: solid $ng;
}
OptionList > .option-list--option-highlighted {
    background: $nm;
    color: $nl;
    text-style: bold;
}
"""

# ── Pre-flight version fetching (synchronous, before TUI starts) ──────────────

def _get(url: str, timeout: int = 5) -> Any:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "thorllm/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def fetch_vllm_versions() -> tuple[str, list[str]]:
    data = _get("https://api.github.com/repos/vllm-project/vllm/releases?per_page=6")
    versions = [r.get("tag_name", "").lstrip("v") for r in (data or []) if r.get("tag_name")]
    versions = [v for v in versions if v][:5]
    latest = versions[0] if versions else "unknown"
    return latest, versions


def fetch_torch_versions() -> tuple[str, list[str]]:
    data = _get("https://pypi.org/pypi/torch/json")
    if not data:
        return "2.10.0", ["2.10.0", "2.9.1", "2.8.0"]
    releases = sorted(data.get("releases", {}).keys(), reverse=True)
    stable = [v for v in releases if all(x not in v for x in ("rc", "dev", "a", "b", "post"))][:5]
    return (stable[0] if stable else "2.10.0"), stable


# ── Reusable helpers ──────────────────────────────────────────────────────────

def _logo() -> Static:
    return Static(LOGO, classes="logo")


def _footer(version: str) -> Static:
    return Static(
        f"thorllm · v{version} · made by ms1design",
        classes="footer",
    )


# ── Screens ───────────────────────────────────────────────────────────────────

class WelcomeScreen(Screen):
    def __init__(self, version: str, build_path: str) -> None:
        super().__init__()
        self._version = version
        self._build_path = build_path

    def compose(self) -> ComposeResult:
        yield _logo()
        with ScrollableContainer(classes="page"):
            yield Static("Welcome to thorllm setup", classes="title")
            yield Static(
                "thorllm — vLLM manager for NVIDIA Jetson Thor\n\n"
                "This wizard will guide you through:\n"
                "  • Installation and cache directories\n"
                "  • vLLM version to install\n"
                "  • PyTorch version to install\n"
                "  • Default model to serve\n"
                "  • HuggingFace token (optional)\n"
                "  • systemd service user\n\n"
                f"Settings saved to {self._build_path}/thorllm.conf"
            )
            with Horizontal(classes="btn-row"):
                yield Button("Quit", id="quit", classes="-error")
                yield Button("Continue →", id="next", classes="-primary")
        yield _footer(self._version)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "next":
            self.app.push_screen(self.app._wizard_screens[1])  # PathsScreen
        elif event.button.id == "quit":
            self.app.exit(None)


class PathsScreen(Screen):
    def __init__(self, build_path: str, cache_root: str, version: str) -> None:
        super().__init__()
        self._bp = build_path
        self._cr = cache_root
        self._version = version

    def compose(self) -> ComposeResult:
        yield _logo()
        with ScrollableContainer(classes="page"):
            yield Static("Installation Paths", classes="title")
            yield Static("Installation directory (BUILD_PATH)", classes="label")
            yield Static(
                "All vLLM files, venv, and model configs will live here.",
                classes="hint",
            )
            yield Input(value=self._bp, id="inp-build", placeholder="~/thorllm")
            yield Static("Cache root directory (CACHE_ROOT)", classes="label")
            yield Static(
                "Triton / FlashInfer / HuggingFace caches. Use fastest storage.",
                classes="hint",
            )
            yield Input(value=self._cr, id="inp-cache", placeholder="~/.cache/vllm")
            with Horizontal(classes="btn-row"):
                yield Button("← Back", id="back")
                yield Button("Next →", id="next", classes="-primary")
        yield _footer(self._version)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "next":
            bp = self.query_one("#inp-build", Input).value.strip() or self._bp
            cr = self.query_one("#inp-cache", Input).value.strip() or self._cr
            self.app.config["BUILD_PATH"] = bp
            self.app.config["CACHE_ROOT"] = cr
            self.app.push_screen(self.app._wizard_screens[2])  # vLLM version
        elif event.button.id == "back":
            self.app.pop_screen()


class VersionScreen(Screen):
    """Reusable version picker for vLLM or PyTorch."""

    def __init__(
        self,
        title: str,
        hint: str,
        config_key: str,
        current: str,
        latest: str,
        versions: list[str],
        version: str,
        next_screen_idx: int,
    ) -> None:
        super().__init__()
        self._title = title
        self._hint = hint
        self._config_key = config_key
        self._current = current
        self._latest = latest
        self._versions = versions
        self._version = version
        self._next_idx = next_screen_idx
        self._selected: str = "latest"

    def compose(self) -> ComposeResult:
        yield _logo()
        with ScrollableContainer(classes="page"):
            yield Static(self._title, classes="title")
            yield Static(
                f"{self._hint}\nLatest: {self._latest}",
                classes="hint",
            )
            ol = OptionList(id="ol")
            ol.add_option(Option("latest  (always newest stable)", id="latest"))
            ol.add_option(None)
            for v in self._versions:
                ol.add_option(Option(f"v{v}", id=v))
            yield ol
            yield Static("Or type a custom version:", classes="label")
            yield Input(placeholder="e.g. 2.10.0", id="inp-custom")
            with Horizontal(classes="btn-row"):
                yield Button("← Back", id="back")
                yield Button("Next →", id="next", classes="-primary")
        yield _footer(self._version)

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        if event.option_id:
            self._selected = event.option_id
            self.query_one("#inp-custom", Input).value = ""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "next":
            custom = self.query_one("#inp-custom", Input).value.strip()
            chosen = custom if custom else self._selected
            self.app.config[self._config_key] = "" if chosen == "latest" else chosen
            self.app.push_screen(self.app._wizard_screens[self._next_idx])
        elif event.button.id == "back":
            self.app.pop_screen()


class ModelScreen(Screen):
    def __init__(self, serve_model: str, version: str) -> None:
        super().__init__()
        self._model = serve_model
        self._version = version

    def compose(self) -> ComposeResult:
        yield _logo()
        with ScrollableContainer(classes="page"):
            yield Static("Default Model", classes="title")
            yield Static(
                "Model to serve  (SERVE_MODEL)\n"
                "Format: <org>/<model-name>  e.g. openai/gpt-oss-120b\n"
                "A YAML config will be created at ${BUILD_PATH}/models/<org>/<name>.yaml",
                classes="hint",
            )
            yield Input(value=self._model, id="inp-model", placeholder="openai/gpt-oss-120b")
            with Horizontal(classes="btn-row"):
                yield Button("← Back", id="back")
                yield Button("Next →", id="next", classes="-primary")
        yield _footer(self._version)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "next":
            val = self.query_one("#inp-model", Input).value.strip() or self._model
            self.app.config["SERVE_MODEL"] = val
            self.app.push_screen(self.app._wizard_screens[5])  # HF token
        elif event.button.id == "back":
            self.app.pop_screen()


class HFTokenScreen(Screen):
    def __init__(self, version: str) -> None:
        super().__init__()
        self._version = version

    def compose(self) -> ComposeResult:
        yield _logo()
        with ScrollableContainer(classes="page"):
            yield Static("HuggingFace Token  (optional)", classes="title")
            yield Static(
                "Required only for gated models (Llama, Gemma, etc).\n"
                "Leave empty to skip — you can add it later in thorllm.conf",
                classes="hint",
            )
            yield Input(password=True, id="inp-token", placeholder="hf_…")
            with Horizontal(classes="btn-row"):
                yield Button("← Back", id="back")
                yield Button("Next →", id="next", classes="-primary")
        yield _footer(self._version)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "next":
            self.app.config["HF_TOKEN"] = self.query_one("#inp-token", Input).value.strip()
            self.app.push_screen(self.app._wizard_screens[6])  # Service user
        elif event.button.id == "back":
            self.app.pop_screen()


class ServiceUserScreen(Screen):
    def __init__(self, service_user: str, version: str) -> None:
        super().__init__()
        self._user = service_user
        self._version = version

    def compose(self) -> ComposeResult:
        yield _logo()
        with ScrollableContainer(classes="page"):
            yield Static("Service User", classes="title")
            yield Static(
                "User account that will run the vLLM systemd service.",
                classes="hint",
            )
            yield Input(value=self._user, id="inp-user", placeholder="username")
            yield Static("⚠  Why we need sudo", classes="label")
            yield Static(
                "thorllm will ask for your sudo password to:\n"
                "  • Install the systemd unit  /etc/systemd/system/vllm.service\n"
                "  • Add a sudoers rule for cache-drop  /etc/sudoers.d/vllm-drop-caches\n"
                "  • Run systemctl daemon-reload\n\n"
                "Your password is used only for these operations and is never stored.",
                classes="warn-text",
            )
            with Horizontal(classes="btn-row"):
                yield Button("← Back", id="back")
                yield Button("Next →", id="next", classes="-primary")
        yield _footer(self._version)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "next":
            val = self.query_one("#inp-user", Input).value.strip() or self._user
            self.app.config["SERVICE_USER"] = val
            self.app.push_screen(self.app._wizard_screens[7])  # Summary
        elif event.button.id == "back":
            self.app.pop_screen()


class SummaryScreen(Screen):
    def __init__(self, version: str) -> None:
        super().__init__()
        self._version = version

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
        yield _logo()
        with ScrollableContainer(classes="page"):
            yield Static("── Configuration Summary", classes="title")
            yield Static(
                "Review your settings. Press Install to save and begin, or Back to change.",
                classes="hint",
            )
            for key, val in rows:
                with Horizontal():
                    yield Static(key, classes="summary-key")
                    yield Static(val, classes="summary-val")
            with Horizontal(classes="btn-row"):
                yield Button("← Back", id="back")
                yield Button("✗ Cancel", id="cancel", classes="-error")
                yield Button("✓ Install", id="install", classes="-primary")
        yield _footer(self._version)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "install":
            self.app.exit(self.app.config)
        elif event.button.id == "cancel":
            self.app.exit(None)
        elif event.button.id == "back":
            self.app.pop_screen()


# ── Model select screen ───────────────────────────────────────────────────────

class ModelSelectScreen(Screen):
    def __init__(self, models: list[str], active: str, version: str) -> None:
        super().__init__()
        self._models = models
        self._active = active
        self._version = version

    def compose(self) -> ComposeResult:
        yield _logo()
        with ScrollableContainer(classes="page"):
            yield Static("Select Active Model", classes="title")
            yield Static(
                "Use arrow keys to highlight, Enter or click to select.",
                classes="hint",
            )
            ol = OptionList(id="ol")
            for m in self._models:
                marker = "●" if m == self._active else "○"
                ol.add_option(Option(f"{marker}  {m}", id=m))
            yield ol
            with Horizontal(classes="btn-row"):
                yield Button("Cancel", id="cancel", classes="-error")
                yield Button("✓ Select", id="select", classes="-primary")
        yield _footer(self._version)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "select":
            ol = self.query_one("#ol", OptionList)
            opt = ol.highlighted_option
            if opt is not None:
                self.app.exit({"selected": opt.id})
            else:
                self.app.exit(None)
        elif event.button.id == "cancel":
            self.app.exit(None)


# ── Apps ─────────────────────────────────────────────────────────────────────

class WizardApp(App[dict | None]):
    """Setup wizard. Returns config dict on install, None on cancel."""

    CSS = CSS

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
        self._vllm_latest = vllm_latest
        self._vllm_versions = vllm_versions
        self._torch_latest = torch_latest
        self._torch_versions = torch_versions

    def on_mount(self) -> None:
        d = self.config
        bp = d.get("BUILD_PATH", os.path.expanduser("~/thorllm"))
        cr = d.get("CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
        su = d.get("SERVICE_USER", os.environ.get("USER", "ubuntu"))

        # Build the ordered screen list up front
        # screens[0] = WelcomeScreen
        # screens[1] = PathsScreen
        # screens[2] = VersionScreen (vLLM)
        # screens[3] = VersionScreen (PyTorch)
        # screens[4] = ModelScreen
        # screens[5] = HFTokenScreen
        # screens[6] = ServiceUserScreen
        # screens[7] = SummaryScreen
        self._wizard_screens = [
            WelcomeScreen(self._version, bp),
            PathsScreen(bp, cr, self._version),
            VersionScreen(
                title="vLLM Version",
                hint="Select vLLM version to install.",
                config_key="VLLM_VERSION",
                current=d.get("VLLM_VERSION", ""),
                latest=self._vllm_latest,
                versions=self._vllm_versions,
                version=self._version,
                next_screen_idx=3,
            ),
            VersionScreen(
                title="PyTorch Version",
                hint="Select PyTorch version (cu130 builds required).",
                config_key="TORCH_VERSION",
                current=d.get("TORCH_VERSION", "2.10.0"),
                latest=self._torch_latest,
                versions=self._torch_versions,
                version=self._version,
                next_screen_idx=4,
            ),
            ModelScreen(d.get("SERVE_MODEL", "openai/gpt-oss-120b"), self._version),
            HFTokenScreen(self._version),
            ServiceUserScreen(su, self._version),
            SummaryScreen(self._version),
        ]
        self.push_screen(self._wizard_screens[0])


class ModelSelectApp(App[dict | None]):
    """Interactive model picker. Returns {selected: name} or None."""

    CSS = CSS

    def __init__(self, models: list[str], active: str, version: str) -> None:
        super().__init__()
        self._models = models
        self._active = active
        self._version = version

    def on_mount(self) -> None:
        self.push_screen(ModelSelectScreen(self._models, self._active, self._version))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="thorllm TUI")
    parser.add_argument("--mode", choices=["wizard", "model-select"], default="wizard")
    parser.add_argument("--defaults", default="{}")
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--models", default="[]")
    parser.add_argument("--active", default="")
    parser.add_argument("--output", default="", help="Write result JSON here instead of stdout")
    args = parser.parse_args()

    defaults = json.loads(args.defaults)
    version = args.version

    if args.mode == "wizard":
        # Fetch versions BEFORE starting TUI so there's no background thread race
        print("Fetching vLLM releases…", flush=True)
        vllm_latest, vllm_versions = fetch_vllm_versions()
        print("Fetching PyTorch releases…", flush=True)
        torch_latest, torch_versions = fetch_torch_versions()

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
