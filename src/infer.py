"""
Инференс OrientNet на папке кропов: окна для длинных строк, усреднение логитов, потоковая обработка.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.common import image_path, resize_pad
from src.dataset import IN_H, IN_W, load_upright, to_tensor


def windows(img: np.ndarray, max_windows: int = 4) -> list[np.ndarray]:
    """
    кроп высотой IN_H -> список окон IN_H x IN_W (одно с паддингом, если строка короткая)
    """
    w = img.shape[1]
    if w <= IN_W:
        return [resize_pad(img, IN_H, IN_W, side="center")]
    n = min(max_windows, int(np.ceil(w / IN_W)))
    starts = np.linspace(0, w - IN_W, n).round().astype(int)
    return [img[:, s : s + IN_W] for s in starts]


@torch.no_grad()
def predict_logits(model: torch.nn.Module, images_dir: Path, ids: list[str], batch_size: int = 128,
                   chunk: int = 512) -> np.ndarray:
    """Средний по окнам антисимметричный логит z для каждого id"""
    model.eval()
    n = len(ids)
    z_sum, cnt = np.zeros(n), np.zeros(n)
    for c0 in tqdm(range(0, n, chunk), desc="predict"):
        tensors, owner = [], []
        for idx in range(c0, min(n, c0 + chunk)):
            img = load_upright(str(image_path(images_dir, ids[idx])), rotate=False)
            for wnd in windows(img):
                tensors.append(to_tensor(wnd)); owner.append(idx)
        owner = np.array(owner)
        z = np.concatenate([model(torch.stack(tensors[i : i + batch_size])).numpy()
                            for i in range(0, len(tensors), batch_size)])
        np.add.at(z_sum, owner, z); np.add.at(cnt, owner, 1)
    return z_sum / cnt
