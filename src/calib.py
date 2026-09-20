"""
Метрики и temperature scaling.

Brier = mean((p - y)^2)
Temperature scaling: p = sigmoid(z / T), T подбирается минимизацией Brier на val

T не меняет ранжирование (accuracy/AUC те же), только уверенность
для антисимметричной модели деление на T сохраняет p(x) + p(rot180 x) = 1.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error по уверенности max(p, 1-p) и жёсткой метке y > 0.5"""
    conf = np.maximum(p, 1 - p)
    correct = ((p > 0.5) == (y > 0.5)).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(out)


def metrics(z: np.ndarray, y: np.ndarray, T: float = 1.0) -> dict[str, float]:
    p = sigmoid(z / T)
    hard = (y > 0.5).astype(float)
    return {
        "brier_soft": brier(p, y),        # против мягких меток учителя
        "brier": brier(p, hard),          # то, что считает платформа
        "acc": float(np.mean((p > 0.5) == (hard > 0.5))),
        "ece": ece(p, hard),
        "conf_err": float(np.mean(((p > 0.9) & (hard == 0)) | ((p < 0.1) & (hard == 1)))),  # уверенные ошибки
    }


def fit_temperature(z: np.ndarray, y: np.ndarray, lo: float = 0.25, hi: float = 8.0) -> float:
    """
    минимизация Brier(sigmoid(z/T), y) по T --- одномерный поиск по сетке в log-шкале + уточнение
    """
    hard = (y > 0.5).astype(float)
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), 200))
    losses = [brier(sigmoid(z / T), hard) for T in grid]
    T = grid[int(np.argmin(losses))]
    for step in (0.1, 0.02, 0.005):  # локальное уточнение
        cands = T * np.exp(np.linspace(-step, step, 21))
        T = cands[int(np.argmin([brier(sigmoid(z / t), hard) for t in cands]))]
    return float(T)
