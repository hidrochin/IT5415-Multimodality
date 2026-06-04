"""Configuration loading: merges config.yaml with .env secrets."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# src/mmrag/config.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    """Thin wrapper around the parsed config.yaml plus path/env helpers."""

    raw: dict[str, Any]
    root: Path

    # dict-style access for sub-sections, e.g. config["chunking"]["chunk_size"]
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def _resolve(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (self.root / p)

    @property
    def data_raw(self) -> Path:
        return self._resolve(self.raw["paths"]["data_raw"])

    @property
    def data_processed(self) -> Path:
        return self._resolve(self.raw["paths"]["data_processed"])

    @property
    def index_dir(self) -> Path:
        return self._resolve(self.raw["paths"]["index_dir"])

    @property
    def gemini_api_key(self) -> str | None:
        return os.getenv("GEMINI_API_KEY")

    def resolve_device(self) -> str:
        """Return 'cuda'/'cpu' honoring config; 'auto' detects a GPU if present."""
        dev = self.raw.get("embeddings", {}).get("device", "auto")
        if dev and dev != "auto":
            return dev
        try:
            import torch  # local import: torch may not be installed yet

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


def load_config(config_path: str | Path | None = None) -> Config:
    """Load .env (for secrets) and config.yaml into a :class:`Config`."""
    load_dotenv(PROJECT_ROOT / ".env")
    cfg_file = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    with open(cfg_file, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config(raw=raw, root=PROJECT_ROOT)
