from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _read_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns)[:2] != ["id", "sentiment"]:
        raise ValueError(f"{path} must have columns: id,sentiment")
    df = df[["id", "sentiment"]].copy()
    df["id"] = df["id"].astype(str)
    df["sentiment"] = df["sentiment"].astype(float)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Blend multiple Kaggle submissions by weighted average.")
    ap.add_argument("--out", type=Path, required=True, help="Output csv path.")
    ap.add_argument(
        "--method",
        choices=("mean", "rank_mean", "logit_mean"),
        default="mean",
        help="Blending method.",
    )
    ap.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Input submission CSVs (each must be id,sentiment).",
    )
    ap.add_argument(
        "--weights",
        type=float,
        nargs="*",
        default=None,
        help="Optional weights, same count as inputs. Defaults to equal weights.",
    )
    args = ap.parse_args()

    inputs = args.inputs
    if args.weights is None or len(args.weights) == 0:
        weights = np.ones(len(inputs), dtype=np.float64) / float(len(inputs))
    else:
        if len(args.weights) != len(inputs):
            raise ValueError("--weights count must match --inputs count")
        weights = np.asarray(args.weights, dtype=np.float64)
        if np.all(weights == 0):
            raise ValueError("--weights cannot be all zeros")
        weights = weights / weights.sum()

    dfs = [_read_submission(p) for p in inputs]
    base_ids = dfs[0]["id"].to_numpy()
    for p, df in zip(inputs[1:], dfs[1:], strict=False):
        if not np.array_equal(base_ids, df["id"].to_numpy()):
            raise ValueError(f"id column mismatch vs first input: {p}")

    preds = np.vstack([df["sentiment"].to_numpy(dtype=np.float64) for df in dfs])
    if args.method == "mean":
        blended = (weights.reshape(-1, 1) * preds).sum(axis=0)
    elif args.method == "rank_mean":
        # Convert each prediction vector to rank-percentile then average.
        # This is often robust for AUC since calibration differences are removed.
        n = preds.shape[1]
        blended = np.zeros(n, dtype=np.float64)
        for w, p in zip(weights, preds, strict=False):
            order = np.argsort(p, kind="mergesort")
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(n, dtype=np.float64)
            ranks /= max(1.0, float(n - 1))
            blended += w * ranks
    else:
        # logit mean: average in log-odds space, then sigmoid
        eps = 1e-6
        p = np.clip(preds, eps, 1.0 - eps)
        logits = np.log(p / (1.0 - p))
        avg_logits = (weights.reshape(-1, 1) * logits).sum(axis=0)
        avg_logits = np.clip(avg_logits, -50.0, 50.0)
        blended = 1.0 / (1.0 + np.exp(-avg_logits))

    out = pd.DataFrame({"id": base_ids, "sentiment": blended})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote: {args.out}")
    print("weights:", ",".join(f"{w:.6f}" for w in weights))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
