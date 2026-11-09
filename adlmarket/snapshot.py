"""<output>/config.yaml: the economy or market config file the run played.

It is the file as given (comments kept). `--resume <output>` reads it back, with the
script and phase recorded in <output>/metadata.json.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml


def write_run_config(runner: Any) -> None:
    cfg = runner.cfg
    origin = cfg.origin or {}
    dest = runner.out / "config.yaml"
    source = Path(cfg.source) if cfg.source else None
    if origin.get("overridden") or not (source and source.exists()):
        # settings changed from Python (tests, scripts): save what was played
        if origin.get("file"):
            dest.write_text(yaml.safe_dump(origin["file"], sort_keys=False, allow_unicode=True,
                                           width=100), encoding="utf-8")
        return
    if source.resolve() != dest.resolve():
        shutil.copyfile(source, dest)
