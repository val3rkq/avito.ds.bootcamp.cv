from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def read_image(path: str | Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rot180(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[::-1, ::-1])


def resize_pad(img: np.ndarray, h: int, w: int, pad_value: int | None = None) -> np.ndarray:
    """масштабирует изображение под заданную высоту h с сохранением пропорций 
    (aspect ratio) и дополняет его справа до ширины w.

    pad_value=None (цвет явно не задан) заполняет медианным значением цвета пикселей с левой и правой
    границ изображения, иначе заполняет указанным цветом pad_value (int или tuple).
    """
    ih, iw = img.shape[:2]
    new_w = min(w, max(1, int(round(h * iw / ih))))
    interp = cv2.INTER_AREA if new_w < iw else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, h), interpolation=interp)
    if new_w == w:
        return resized
    if pad_value is None:
        border = np.concatenate([resized[:, 0], resized[:, -1]], axis=0)
        pad_value = np.median(border, axis=0)
    out = np.empty((h, w) + img.shape[2:], dtype=img.dtype)
    out[...] = pad_value
    out[:, :new_w] = resized
    return out


def list_image_ids(images_dir: str | Path, sample_submission: str | Path | None = None) -> list[str]:
    images_dir = Path(images_dir)
    on_disk = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    if sample_submission is not None:
        ids = pd.read_csv(sample_submission)["image_id"].astype(str).tolist()
        missing = [i for i in ids if i not in on_disk]
        if missing:
            raise FileNotFoundError(f"{len(missing)} ids from sample_submission not found, e.g. {missing[:3]}")
        return ids
    return sorted(on_disk)


def image_path(images_dir: str | Path, image_id: str) -> Path:
    images_dir = Path(images_dir)
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        p = images_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{image_id} not found in {images_dir}")


def write_submission(ids: list[str], p_180: np.ndarray, out_path: str | Path, expected_n: int | None = 20_000) -> pd.DataFrame:
    p = np.asarray(p_180, dtype=np.float64)
    assert len(ids) == len(p), f"ids ({len(ids)}) and p_180 ({len(p)}) length mismatch"
    if expected_n is not None:
        assert len(ids) == expected_n, f"expected {expected_n} rows, got {len(ids)}"
    assert not np.isnan(p).any(), "NaN in p_180"
    assert len(set(ids)) == len(ids), "duplicate image_id"
    
    p = np.clip(p, 0.0, 1.0)
    df = pd.DataFrame({"image_id": ids, "p_180": p})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")
    return df


def summarize_probs(p: np.ndarray, name: str = "p_180") -> str:
    p = np.asarray(p)
    lines = [
        f"{name}: n={len(p)}  mean={p.mean():.3f}  share>0.5={float((p > 0.5).mean()):.3f}",
        f"  uncertain (0.3..0.7): {float(((p > 0.3) & (p < 0.7)).mean()):.3f}",
        f"  confident (<0.05 | >0.95): {float(((p < 0.05) | (p > 0.95)).mean()):.3f}",
    ]
    hist, edges = np.histogram(p, bins=10, range=(0, 1))
    lines.append("  hist: " + " ".join(f"[{edges[i]:.1f}-{edges[i+1]:.1f}):{hist[i]}" for i in range(10)))
    return "\n".join(lines)
