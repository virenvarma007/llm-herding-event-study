"""Config loading (YAML + CLI overrides). Plain dicts with attribute access."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class Cfg(dict):
    """dict with attribute access, recursive."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v


ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | None = None) -> Cfg:
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p) as f:
        cfg = Cfg(yaml.safe_load(f))
    cfg["_config_path"] = str(p)
    # resolve relative paths against the project root
    for key in ("data_dir", "out_dir"):
        v = Path(cfg[key])
        if not v.is_absolute():
            cfg[key] = str((ROOT / v).resolve())
    return cfg


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (KEY=VALUE lines) -- does not override existing env vars."""
    p = path or ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def run_name(arm: str, share: float, provider_override: str | None = None, prompt_variant: str = "free") -> str:
    tag = f"{arm}_share{share:.2f}"
    if prompt_variant and prompt_variant != "free":
        tag += f"_{prompt_variant}"
    if provider_override:
        tag += f"_{provider_override}"
    return tag
