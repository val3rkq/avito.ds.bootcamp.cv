"""
Попытка 2: своя модель (MobileNetV3-small + антисимметричная голова) на псевдо-размеченном тесте.

Этапы (каждый промежуточный артефакт):
    1. outputs/submission_v1.csv   - предсказания попытки 1 (PaddleOCR cls); иначе считаем бейзлайн
    2. data/pseudo_test/meta.csv   - уверенное подмножество теста как прямые кропы; иначе строим из п.1
    3. weights/v2/orient_mnv3s.pt  — обучение на train-части meta (BCE с мягкими метками), val — 15% meta;
                                     OneCycle на --epochs 20, остановка после --stop-after 13 (так получены
                                     итоговые веса); если файл весов есть — обучение пропускается (--retrain)
    4. temperature scaling на val (минимизация Brier)
    5. инференс на всех 20k кропах; p = (1 - w) * p_student + w * p_v1, w = --blend-v1 (0 — чистый студент)
       Brier выпукл по p, поэтому Brier смеси <= среднего Brier'ов моделей; на лёгких кропах доминирует
       уверенный v1, на трудных студент добавляет знание домена. -> --out

Val из псевдо-меток измеряет согласие с учителем на «лёгких» кропах --- это контроль переобучения и
подбор T, но не честная оценка качества. 
Поэтому дополнительно печатается label-free диагностика на тесте: согласие с попыткой 1 на уверенном подмножестве 
и распределение p на неуверенном.

Запуск:
    docker compose run --rm app submission_v2 --out outputs/submission_v2.csv
    (быстрая проверка пайплайна: --epochs 1 --limit-train 500 --retrain --weights outputs/smoke.pt)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.calib import fit_temperature, metrics, sigmoid
from src.common import list_image_ids, seed_everything, summarize_probs, write_submission
from src.dataset import OrientDataset
from src.infer import predict_logits
from src.model import OrientNet, count_params
from src.pseudo import load_or_build_meta


def split_meta(meta: pd.DataFrame, val_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(meta))
    n_val = int(len(meta) * val_frac)
    return meta.iloc[idx[n_val:]].reset_index(drop=True), meta.iloc[idx[:n_val]].reset_index(drop=True)


@torch.no_grad()
def eval_logits(model: OrientNet, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    zs, ys = [], []
    for x, y in loader:
        zs.append(model(x).numpy()); ys.append(y.numpy())
    return np.concatenate(zs), np.concatenate(ys)


def train(model: OrientNet, train_ds: OrientDataset, val_ds: OrientDataset, epochs: int, lr: float,
          batch_size: int, workers: int, ckpt: Path, stop_after: int | None = None) -> dict:
    """OneCycle строится на `epochs`; `stop_after` прерывает обучение раньше (расписание при этом не меняется)."""
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True,
                          persistent_workers=workers > 0)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)

    best = {"brier": 1.0}
    for ep in range(1, epochs + 1):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for x, y in train_dl:
            z = model(x)
            loss = F.binary_cross_entropy_with_logits(z, y)  # мягкие метки y_soft
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            loss_sum += loss.item() * len(y); n += len(y)
        z_val, y_val = eval_logits(model, val_dl)
        m = metrics(z_val, y_val)
        print(f"ep {ep:2d}/{epochs}  train_bce={loss_sum / n:.4f}  val: brier={m['brier']:.4f} acc={m['acc']:.4f} "
              f"ece={m['ece']:.4f} conf_err={m['conf_err']:.4f}  ({time.time() - t0:.0f}s)")
        if m["brier"] < best["brier"]:
            best = {**m, "epoch": ep}
            torch.save(model.state_dict(), ckpt)
        if stop_after and ep >= stop_after:
            print(f"[train] stopped after epoch {ep} (--stop-after)")
            break
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return best


def test_diagnostics(p2: np.ndarray, ids: list[str], preds_v1: pd.DataFrame, thr: float) -> None:
    """Label-free сравнение с попыткой 1 на тесте."""
    p1 = preds_v1.set_index("image_id").loc[ids, "p_180"].values
    conf = (p1 < thr) | (p1 > 1 - thr)
    print(summarize_probs(p2, "p_v2"))
    agree = float(np.mean((p2[conf] > 0.5) == (p1[conf] > 0.5)))
    print(f"agreement with v1 (class) on confident v1 subset: {agree:.4f}")
    if agree < 0.95:
        print("!!! WARNING: low agreement with v1 on its confident subset -> train/inference mismatch or shortcut; "
              "do not submit")
    print(f"on UNconfident v1 subset ({(~conf).sum()}): v2 confident(<0.05|>0.95)={np.mean((p2[~conf] < 0.05) | (p2[~conf] > 0.95)):.3f}  "
          f"class agreement with v1={np.mean((p2[~conf] > 0.5) == (p1[~conf] > 0.5)):.3f}")
    print(f"mean |p_v2 - p_v1| = {np.mean(np.abs(p2 - p1)):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="submission попытки 2")
    ap.add_argument("--images", type=Path, default=Path("test/images"))
    ap.add_argument("--sample-submission", type=Path, default=Path("sample_submission.csv"))
    ap.add_argument("--preds-v1", type=Path, default=Path("outputs/submission_v1.csv"))
    ap.add_argument("--meta", type=Path, default=Path("data/pseudo_test/meta.csv"))
    ap.add_argument("--thr", type=float, default=0.03, help="порог псевдо-разметки")
    ap.add_argument("--backbone", default="mobilenetv3_small_100")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=20, help="длина OneCycle-расписания")
    ap.add_argument("--stop-after", type=int, default=13, help="остановить обучение после этой эпохи (0 — не останавливать)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 — детерминированно и без spawn-проблем)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = по умолчанию)")
    ap.add_argument("--limit-train", type=int, default=None, help="урезать train (smoke test)")
    ap.add_argument("--weights", type=Path, default=Path("weights/v2/orient_mnv3s.pt"),
                    help="если файл есть — обучение пропускается")
    ap.add_argument("--retrain", action="store_true", help="обучить заново, даже если веса есть")
    ap.add_argument("--blend-v1", type=float, default=0.0, help="вес попытки 1 в смеси вероятностей (0 — чистый студент)")
    ap.add_argument("--student-out", type=Path, default=None, help="куда дополнительно записать чистого студента")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    sample = args.sample_submission if args.sample_submission.is_file() else None

    # предсказания попытки 1 и псевдо-разметка (load-or-build)
    meta = load_or_build_meta(args.meta, args.preds_v1, args.images, args.thr, sample,
                              sheet=Path("outputs/pseudo_test_sheet.png"))
    preds_v1 = pd.read_csv(args.preds_v1)

    # обучение
    train_meta, val_meta = split_meta(meta, args.val_frac, args.seed)
    if args.limit_train:
        train_meta = train_meta.head(args.limit_train)
    print(f"[data] train={len(train_meta)} val={len(val_meta)}; loading crops into memory...")
    train_ds, val_ds = OrientDataset(train_meta, train=True), OrientDataset(val_meta, train=False)

    model = OrientNet(args.backbone, pretrained=not args.no_pretrained)
    print(f"[model] {args.backbone}: {count_params(model) / 1e6:.2f}M params")
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    if args.weights.is_file() and not args.retrain:
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        print(f"[train] loaded weights from {args.weights} (use --retrain to train again)")
        best = {"epoch": None}
    else:
        best = train(model, train_ds, val_ds, args.epochs, args.lr, args.batch_size, args.workers, args.weights,
                     stop_after=args.stop_after or None)
        print(f"[train] best epoch {best['epoch']}: val brier={best['brier']:.4f} acc={best['acc']:.4f}")

    # калибровка
    z_val, y_val = eval_logits(model, DataLoader(val_ds, batch_size=256))
    T = fit_temperature(z_val, y_val)
    before, after = metrics(z_val, y_val, 1.0), metrics(z_val, y_val, T)
    print(f"[calib] T={T:.3f}: val brier {before['brier']:.4f} -> {after['brier']:.4f}, ece {before['ece']:.4f} -> {after['ece']:.4f}")
    json.dump({"backbone": args.backbone, "T": T, "val": after, "best_epoch": best["epoch"], "seed": args.seed,
               "epochs": args.epochs, "stop_after": args.stop_after},
              open(args.weights.with_suffix(".json"), "w"), indent=2)

    # инференс на всём тесте
    ids = list_image_ids(args.images, sample)
    t0 = time.perf_counter()
    z = predict_logits(model, args.images, ids)
    print(f"[infer] {len(ids)} crops in {time.perf_counter() - t0:.0f}s ({1000 * (time.perf_counter() - t0) / len(ids):.1f} ms/crop, CPU)")
    p_student = sigmoid(z / T)
    test_diagnostics(p_student, ids, preds_v1, args.thr)
    if args.student_out:
        write_submission(ids, p_student, args.student_out)
        print(f"wrote student-only -> {args.student_out}")

    p_v1 = preds_v1.set_index("image_id").loc[ids, "p_180"].values
    p = (1 - args.blend_v1) * p_student + args.blend_v1 * p_v1
    print(f"[blend] w_v1={args.blend_v1}: " + summarize_probs(p, "p_final").split(chr(10))[0])
    write_submission(ids, p, args.out)
    print(f"wrote {args.out}; weights -> {args.weights}")


if __name__ == "__main__":
    main()
