"""
Zero-shot (без дообучения) бейзлайн: PaddleOCR classifier (ch_ppocr_mobile_v2.0_cls)

Модель MobileNetV3 из PaddleOCR / RapidOCR предсказывает 0 / 180 градусов для каждого текстового кропа. 

Вход:
    3x48x192, BGR, нормализованный к [-1, 1], справа паддинг нулями
    
Берется softmax[180] как p_180 + антисимметричный TTA:
    z = (logit(x) - logit(rot180(x))) / 2, 
    p = sigmoid(z), так что p(x) + p(rot180(x)) == 1

Использование:
    python -m src.baseline_paddle --images test/images --out submission.csv
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from tqdm import tqdm

from src.common import image_path, list_image_ids, read_image, rot180, summarize_probs, write_submission

CLS_H, CLS_W = 48, 192
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "weights" / "paddle_cls" / "ch_ppocr_mobile_v2.0_cls_infer.onnx"


def preprocess_squeeze(img_bgr: np.ndarray) -> np.ndarray:
    """
    сохраняет пропорции, масштабирует до высоты 48 и паддит справа до ширины 192
    """
    h, w = img_bgr.shape[:2]
    new_w = min(CLS_W, int(math.ceil(CLS_H * w / h)))
    x = cv2.resize(img_bgr, (new_w, CLS_H)).astype(np.float32)
    x = (x.transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
    out = np.zeros((3, CLS_H, CLS_W), dtype=np.float32)
    out[:, :, :new_w] = x
    return out


def preprocess_windows(img_bgr: np.ndarray, max_windows: int = 4) -> list[np.ndarray]:
    """
    масштабирует изображение до высоты 48, затем скользящим окном шириной 192 делает N кропов (N <= max_windows)
    """
    h, w = img_bgr.shape[:2]
    new_w = max(1, int(round(CLS_H * w / h)))
    if new_w <= CLS_W:
        return [preprocess_squeeze(img_bgr)]
    x = cv2.resize(img_bgr, (new_w, CLS_H)).astype(np.float32)
    x = (x.transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
    n = min(max_windows, int(math.ceil(new_w / CLS_W)))
    starts = np.linspace(0, new_w - CLS_W, n).round().astype(int) if n > 1 else [0]
    return [np.ascontiguousarray(x[:, :, s : s + CLS_W]) for s in starts]


class PaddleCls:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL, threads: int = 0):
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def logits(self, batch: np.ndarray) -> np.ndarray:
        """
        батч: Nx3x48x192 float32 -> N логитов класса 180 (log p180 - log p0)
        """
        prob = self.sess.run(None, {self.input_name: batch})[0]  # выход softmax размера N x 2
        prob = np.clip(prob, 1e-7, 1 - 1e-7)
        return np.log(prob[:, 1]) - np.log(prob[:, 0])


def predict_logits(model: PaddleCls, tensors: list[np.ndarray], batch_size: int) -> np.ndarray:
    out = np.empty(len(tensors), dtype=np.float32)
    for i in range(0, len(tensors), batch_size):
        out[i : i + batch_size] = model.logits(np.stack(tensors[i : i + batch_size]))
    return out


def run(images_dir: Path, ids: list[str], model: PaddleCls, mode: str, batch_size: int,
        chunk: int = 256) -> pd.DataFrame:
    """
    возвращает датафрейм с логитами для x и rot180(x) для каждого изображения с производными вероятностей

    Обработка потоковая, чанками по `chunk` изображений: препроцессинг -> инференс -> накопление логитов.
    Иначе для 20k кропов с окнами и TTA пришлось бы держать ~80k тензоров (~9 GB) в памяти.
    """
    prep = preprocess_windows if mode == "windows" else (lambda im: [preprocess_squeeze(im)])

    n = len(ids)
    z_x = np.zeros(n); z_rx = np.zeros(n); cnt = np.zeros(n)
    n_tensors, t_infer = 0, 0.0

    for c0 in tqdm(range(0, n, chunk), desc="chunks"):
        # Развернуть все окна изображений чанка (+повороты) в один список тензоров, запомнить принадлежность
        tensors, owner, is_rot = [], [], []
        for idx in range(c0, min(n, c0 + chunk)):
            img = cv2.cvtColor(read_image(image_path(images_dir, ids[idx])), cv2.COLOR_RGB2BGR)
            for r, im in ((0, img), (1, rot180(img))):
                for t in prep(im):
                    tensors.append(t); owner.append(idx); is_rot.append(r)
        owner, is_rot = np.array(owner), np.array(is_rot)

        t0 = time.perf_counter()
        z = predict_logits(model, tensors, batch_size)
        t_infer += time.perf_counter() - t0
        n_tensors += len(tensors)

        # накопить сумму логитов окон для каждого (изображение, поворот)
        np.add.at(z_x, owner[is_rot == 0], z[is_rot == 0])
        np.add.at(z_rx, owner[is_rot == 1], z[is_rot == 1])
        np.add.at(cnt, owner[is_rot == 0], 1)

    print(f"inference: {n_tensors} tensors in {t_infer:.1f}s -> {1000 * t_infer / n_tensors:.2f} ms/tensor, "
          f"{1000 * t_infer / n:.2f} ms/image (incl. rot TTA)")
    z_x /= cnt; z_rx /= cnt

    sigmoid = lambda v: 1 / (1 + np.exp(-v))
    return pd.DataFrame({
        "image_id": ids,
        "z_x": z_x, "z_rx": z_rx, "n_windows": cnt.astype(int),
        "p_x": sigmoid(z_x),                 # обычное предсказание, без TTA
        "p_rx": sigmoid(z_rx),               # предсказание для повернутого кропа
        "p_tta": sigmoid((z_x - z_rx) / 2),  # антисимметричное TTA
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--sample-submission", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    ap.add_argument("--raw-out", type=Path, default=None, help="csv with logits / no-TTA probs for analysis")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--mode", choices=["squeeze", "windows"], default="windows")
    ap.add_argument("--no-tta", action="store_true", help="use p(x) instead of antisymmetric TTA")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="only first N images (local smoke test)")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    ids = list_image_ids(args.images, args.sample_submission)
    if args.limit:
        ids = ids[: args.limit]
    print(f"{len(ids)} images, mode={args.mode}, tta={not args.no_tta}")

    df = run(args.images, ids, PaddleCls(args.model, args.threads), args.mode, args.batch_size)
    p = df["p_x"].values if args.no_tta else df["p_tta"].values

    print(summarize_probs(df["p_x"].values, "p_x (no TTA)"))
    print(summarize_probs(p, "p_final"))
    inconsistency = np.abs(df["p_x"] + df["p_rx"] - 1)
    print(f"|p(x)+p(rot x)-1|: mean={inconsistency.mean():.3f}  p90={inconsistency.quantile(0.9):.3f}  "
          f"share>0.5={float((inconsistency > 0.5).mean()):.3f}")
    print(f"share of images with >1 window: {float((df['n_windows'] > 1).mean()):.3f}")

    if args.raw_out:
        args.raw_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.raw_out, index=False, float_format="%.5f")
    write_submission(ids, p, args.out, expected_n=None if args.limit else 20_000)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
