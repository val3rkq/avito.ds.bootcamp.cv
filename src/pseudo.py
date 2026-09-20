"""
Псевдо-разметка теста по уверенным предсказаниям попытки 1.
Уверенная метка превращает тестовый кроп в кроп с известной ориентацией:
    p < thr      -> кроп прямой как есть
    p > 1 - thr  -> кроп прямой после поворота на 180 градусов

meta.csv: image_id, path, p_src, rotate, y_soft, w, h
    rotate --- нужно ли повернуть исходник, чтобы получить прямой кроп
    y_soft = min(p, 1 - p) --- мягкая метка «прямой кроп на самом деле перевёрнут» (шум учителя)
Файлы не копируются --- только ссылки на исходные кропы.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.common import image_path, list_image_ids, read_image, write_submission

META_COLS = ["image_id", "path", "p_src", "rotate", "y_soft", "w", "h"]


def load_or_build_preds(preds: Path, images: Path, sample_submission: Path | None) -> pd.DataFrame:
    if preds.is_file():
        df = pd.read_csv(preds)
        assert {"image_id", "p_180"} <= set(df.columns), f"{preds}: expected columns image_id, p_180"
        print(f"[preds] loaded {len(df)} from {preds}")
        return df
    from src.paddle_cls import predict_paddle  # ленивый импорт: onnxruntime нужен только здесь

    print(f"[preds] {preds} not found -> running attempt-1 baseline")
    ids = list_image_ids(images, sample_submission if sample_submission and sample_submission.is_file() else None)
    raw = predict_paddle(images, ids)
    return write_submission(ids, raw["p_tta"].values, preds, expected_n=None)


def build_meta(preds: pd.DataFrame, images: Path, thr: float) -> pd.DataFrame:
    p = preds["p_180"].values
    keep = (p < thr) | (p > 1 - thr)
    sel = preds.loc[keep, ["image_id", "p_180"]].rename(columns={"p_180": "p_src"}).reset_index(drop=True)
    sel["rotate"] = sel["p_src"] > 0.5
    sel["y_soft"] = np.minimum(sel["p_src"], 1 - sel["p_src"])
    paths, ws, hs = [], [], []
    for iid in sel["image_id"]:
        pth = image_path(images, iid)
        with Image.open(pth) as im:
            ws.append(im.width); hs.append(im.height)
        paths.append(pth.as_posix())
    sel["path"] = paths; sel["w"] = ws; sel["h"] = hs
    return sel[META_COLS]


def load_or_build_meta(meta_path: Path, preds_path: Path, images: Path, thr: float,
                       sample_submission: Path | None = None, sheet: Path | None = None) -> pd.DataFrame:
    if meta_path.is_file():
        meta = pd.read_csv(meta_path)
        assert list(meta.columns) == META_COLS, f"{meta_path}: unexpected columns {list(meta.columns)}"
        print(f"[meta] loaded {len(meta)} from {meta_path}")
        return meta
    preds = load_or_build_preds(preds_path, images, sample_submission)
    meta = build_meta(preds, images, thr)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(meta_path, index=False, float_format="%.5f")
    n, n_rot = len(meta), int(meta["rotate"].sum())
    print(f"[meta] thr={thr}: kept {n}/{len(preds)} ({n / len(preds):.1%}); rotated to upright: {n_rot}; "
          f"y_soft mean={meta['y_soft'].mean():.4f} -> {meta_path}")
    if sheet is not None:
        contact_sheet(meta, sheet)
    return meta


def contact_sheet(meta: pd.DataFrame, out: Path, n: int = 24, seed: int = 42) -> None:
    """
    случайные псевдо-размеченные кропы, приведённые к прямой ориентации --- для проверки на глаз
    """
    rows = meta.sample(min(n, len(meta)), random_state=seed)
    W, H = 420, 64
    sheet = Image.new("RGB", (W * 2, H * ((len(rows) + 1) // 2)), "white")
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(rows.itertuples()):
        im = read_image(r.path)
        if r.rotate:
            im = im[::-1, ::-1]
        im = Image.fromarray(im)
        s = min((W - 110) / im.width, (H - 8) / im.height)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
        x0, y0 = (i % 2) * W, (i // 2) * H
        sheet.paste(im, (x0 + 4, y0 + 4))
        d.text((x0 + W - 100, y0 + 24), f"p={r.p_src:.3f}{' R' if r.rotate else ''}", fill="red")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
