"""
Попытка 1: zero-shot бейзлайн (PaddleOCR cls + окна + антисимметричный TTA) -> submission.

Запуск:
    docker compose run --rm app baseline_paddle --out outputs/submission_v1.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.common import list_image_ids, write_submission
from src.paddle_cls import DEFAULT_MODEL, predict_paddle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=Path("test/images"))
    ap.add_argument("--sample-submission", type=Path, default=Path("sample_submission.csv"),
                    help="порядок id; если файла нет — id берутся из папки с картинками")
    ap.add_argument("--out", type=Path, required=True, help="куда писать submission (image_id, p_180)")
    ap.add_argument("--raw-out", type=Path, default=None, help="csv с логитами и p без TTA для анализа")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--mode", choices=["squeeze", "windows"], default="windows")
    ap.add_argument("--no-tta", action="store_true", help="p(x) вместо антисимметричного TTA")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="только первые N картинок (smoke test)")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    sample = args.sample_submission if args.sample_submission.is_file() else None
    ids = list_image_ids(args.images, sample)
    if args.limit:
        ids = ids[: args.limit]
    print(f"{len(ids)} images, mode={args.mode}, tta={not args.no_tta}, ids from {'sample_submission' if sample else 'folder'}")

    df = predict_paddle(args.images, ids, args.model, args.mode, args.batch_size, args.threads)
    p = df["p_x"].values if args.no_tta else df["p_tta"].values

    if args.raw_out:
        args.raw_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.raw_out, index=False, float_format="%.5f")
    write_submission(ids, p, args.out, expected_n=None if args.limit else 20_000)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
