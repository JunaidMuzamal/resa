"""Configuration loading with dotted command-line overrides."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class Config(dict):
    """A dict with attribute access and resolved project paths."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - attribute errors are programming errors
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def path(self, key: str) -> Path:
        """Return an absolute path for ``paths.<key>`` (relative paths resolve to the project root)."""
        raw = Path(self["paths"][key])
        root = Path(os.environ.get("RESA_ROOT", PROJECT_ROOT))
        return raw if raw.is_absolute() else root / raw

    def get_in(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _set_in(data: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: Optional[str | Path] = None, overrides: Optional[Iterable[str]] = None) -> Config:
    """Load the default config, merge an optional extra YAML file, then apply ``key=value`` overrides."""
    with open(DEFAULT_CONFIG, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if path and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        with open(path, encoding="utf-8") as handle:
            data = _deep_merge(data, yaml.safe_load(handle) or {})
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        _set_in(data, key.strip(), yaml.safe_load(raw))
    return Config(data)


def add_config_args(parser) -> None:
    """Attach the standard --config / --set arguments to an argparse parser."""
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config (merged over defaults)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set student.epochs=3 (repeatable)",
    )


def config_from_args(args) -> Config:
    return load_config(args.config, args.overrides)
