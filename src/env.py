"""
Единая точка входа для ноутбуков: где корень репозитория и где лежат данные.

Архив test.zip распаковывается в корень репозитория, т.е.
    <ROOT>/sample_submission.csv
    <ROOT>/test/images/*.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def setup(root: Path | None = None) -> dict[str, Path]:
    """
    - chdir в корень, добавить его в sys.path, вернуть словарь путей
    - проверяет, что данные на месте
    """
    
    def find_root(start: Path | None = None) -> Path:
        """
        поднимается от cwd вверх, пока не найдёт папку с src/ и requirements.txt.
        """
        p = (start or Path.cwd()).resolve()
        for cand in (p, *p.parents):
            if (cand / "src").is_dir() and (cand / "requirements.txt").is_file():
                return cand
        raise FileNotFoundError("repo root not found: run from inside the repository")
    
    root = (root or find_root()).resolve()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    paths = {
        "root": root,
        "images": root / "test" / "images",
        "sample_submission": root / "sample_submission.csv",
        "outputs": root / "outputs",
        "weights": root / "weights",
    }
    paths["outputs"].mkdir(exist_ok=True)

    if not paths["images"].is_dir() or not any(paths["images"].glob("*.png")):
        raise FileNotFoundError(
            f"no images in {paths['images']}\n"
            f"unzip test.zip into {root} so that {root / 'test' / 'images'} contains the PNGs"
        )
    if not paths["sample_submission"].is_file():
        print(f"[env] {paths['sample_submission']} not found -> ids will be taken from the images folder")
        paths["sample_submission"] = None  # type: ignore[assignment]
    return paths
