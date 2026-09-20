"""
Датасет для обучения классификатора ориентации.

Вход:
    meta.csv с прямыми кропами (path, rotate, y_soft). 
    Все кропы один раз читаются в память, приводятся к прямой ориентации и масштабируются к высоте IN_H 
    (ширина по пропорции, не больше MAX_W)

Модель антисимметрична (loss(x, y) == loss(rot180 x, 1 - y)), поэтому подавать повёрнутые копии не нужно,
достаточно прямых кропов с мягкой меткой y_soft.

Аугментации --- только те, что не меняют ориентацию: масштаб, лёгкий поворот, blur, шум, JPEG, яркость/контраст,
случайное окно по ширине
"""

from __future__ import annotations

import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.common import read_image, resize_pad, rot180

IN_H, IN_W = 32, 256
MAX_W = 1024  # длинные строки храним не длиннее этого (при IN_H), окно всё равно IN_W

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet, под претрейн timm
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 RGB -> 3xHxW float, ImageNet-нормализация."""
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def load_upright(path: str, rotate: bool) -> np.ndarray:
    """
    прямой кроп высотой IN_H (ширина по пропорции, обрезка до MAX_W)
    """
    img = read_image(path)
    if rotate:
        img = rot180(img)
    h, w = img.shape[:2]
    new_w = int(np.clip(round(IN_H * w / h), 8, MAX_W))
    interp = cv2.INTER_AREA if new_w < w else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, IN_H), interpolation=interp)


def random_window(img: np.ndarray, w: int = IN_W) -> np.ndarray:
    """
    cлучайное окно ширины w; короткие строки паддятся со случайным делением между сторонами
    """
    if img.shape[1] <= w:
        return resize_pad(img, IN_H, w, side="random")
    s = random.randint(0, img.shape[1] - w)
    return img[:, s : s + w]


def augment(img: np.ndarray) -> np.ndarray:
    """Аугментации на прямом кропе высотой IN_H. Порядок: геометрия -> деградация -> цвет."""
    h, w = img.shape[:2]
    # масштаб: имитируем мелкий текст (down -> up) и лёгкое растяжение по ширине
    if random.random() < 0.5:
        f = random.uniform(0.4, 1.0)
        small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if random.random() < 0.3:
        img = cv2.resize(img, (max(8, int(w * random.uniform(0.8, 1.25))), h), interpolation=cv2.INTER_LINEAR)
    # небольшой поворот (±3°) — детектор редко даёт идеально горизонтальный бокс
    if random.random() < 0.3:
        ang = random.uniform(-3, 3)
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, h / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (img.shape[1], h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # деградации
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.3, 1.2))
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(2, 10), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        q = random.randint(30, 90)
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    # цвет
    if random.random() < 0.5:
        alpha, beta = random.uniform(0.7, 1.3), random.uniform(-30, 30)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.15:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    if random.random() < 0.1:
        img = 255 - img  # инверсия: светлый текст на тёмном фоне и наоборот
    return img


def first_window(img: np.ndarray) -> np.ndarray:
    """Val/инференс: центрированный паддинг для коротких строк, первое окно для длинных."""
    return resize_pad(img, IN_H, IN_W, side="center") if img.shape[1] <= IN_W else img[:, :IN_W]


class OrientDataset(Dataset):
    """
    train: 
        прямые кропы с аугментациями и случайным окном, метка y_soft
    val:
        каждый кроп в обеих ориентациях, построенных из СЫРОГО кропа ровно как на инференсе
        (поворот до resize/паддинга): (x, y_soft) и (rot180 x, 1 - y_soft)
        
    """

    def __init__(self, meta: pd.DataFrame, train: bool):
        self.train = train
        self.y = meta["y_soft"].values.astype(np.float32)
        self.imgs = [load_upright(p, r) for p, r in zip(meta["path"], meta["rotate"])]
        if not train:
            self.imgs_rot = [load_upright(p, not r) for p, r in zip(meta["path"], meta["rotate"])]

    def __len__(self) -> int:
        return len(self.imgs) if self.train else 2 * len(self.imgs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train:
            return to_tensor(random_window(augment(self.imgs[i]))), torch.tensor(self.y[i])
        j, flip = i // 2, i % 2
        img, y = (self.imgs_rot[j], 1.0 - self.y[j]) if flip else (self.imgs[j], self.y[j])
        return to_tensor(first_window(img)), torch.tensor(np.float32(y))
