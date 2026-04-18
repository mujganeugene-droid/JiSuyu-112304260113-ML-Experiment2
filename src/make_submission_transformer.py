from __future__ import annotations

import argparse
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import warnings
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)


def _resolve_data_file(data_dir: Path, base_name: str) -> Path | None:
    candidates = [
        data_dir / base_name,
        data_dir / base_name / base_name,  # e.g. ./labeledTrainData.tsv/labeledTrainData.tsv
        data_dir / "data" / base_name,
        data_dir / "input" / base_name,
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


@dataclass(frozen=True)
class DataPaths:
    labeled: Path
    test: Path


def _find_data_paths(data_dir: Path) -> DataPaths:
    labeled = _resolve_data_file(data_dir, "labeledTrainData.tsv")
    test = _resolve_data_file(data_dir, "testData.tsv")
    if labeled is None:
        raise FileNotFoundError("找不到 labeledTrainData.tsv（请检查 --data-dir）")
    if test is None:
        raise FileNotFoundError("找不到 testData.tsv（请检查 --data-dir）")
    return DataPaths(labeled=labeled, test=test)


def _read_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=0, delimiter="\t", quoting=3)
    for col in ("id", "review"):
        if col in df.columns and df[col].dtype == object:
            s = df[col].astype(str)
            mask = s.str.startswith('"') & s.str.endswith('"') & (s.str.len() >= 2)
            df[col] = s.where(~mask, s.str.slice(1, -1))
    return df


_SPACE_RE = re.compile(r"\s+")


def _strip_html_keep_text(review_html: str) -> str:
    text = BeautifulSoup(review_html, "lxml").get_text(" ")
    return _SPACE_RE.sub(" ", text).strip()


warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int] | None):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        item = {"text": self.texts[idx]}
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune a Transformer and write Kaggle submission.")
    ap.add_argument("--data-dir", type=Path, default=Path("."), help="Directory containing TSV files.")
    ap.add_argument("--out", type=Path, default=Path("submission_transformer.csv"))
    ap.add_argument("--artifacts-dir", type=Path, default=Path("artifacts_transformer"))

    ap.add_argument("--model", type=str, default="distilbert-base-uncased")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps.")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--fp16", action="store_true", help="Use mixed precision on CUDA.")
    ap.add_argument("--grad-checkpointing", action="store_true", help="Enable gradient checkpointing to save VRAM.")
    ap.add_argument("--offline", action="store_true", help="Use local cache only (no network).")
    ap.add_argument("--hf-home", type=Path, default=None, help="Optional HF cache dir (sets HF_HOME).")
    ap.add_argument("--max-train-rows", type=int, default=None, help="Debug: limit labeled rows.")
    ap.add_argument("--max-test-rows", type=int, default=None, help="Debug: limit test rows.")

    ap.add_argument("--valid-ratio", type=float, default=0.1, help="Holdout validation ratio (for AUC print).")
    ap.add_argument("--no-eval", action="store_true", help="Skip validation AUC.")

    args = ap.parse_args()
    _set_seed(args.seed)

    # Cache / telemetry
    if args.hf_home is not None:
        os.environ["HF_HOME"] = str(args.hf_home)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    paths = _find_data_paths(args.data_dir)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    )
    print(f"[device] {device}")
    print(f"[model] {args.model}")

    train_df = _read_tsv(paths.labeled)
    test_df = _read_tsv(paths.test)
    if args.max_train_rows:
        train_df = train_df.head(args.max_train_rows)
    if args.max_test_rows:
        test_df = test_df.head(args.max_test_rows)

    train_texts = [_strip_html_keep_text(x) for x in train_df["review"].astype(str).tolist()]
    y = train_df["sentiment"].astype(int).tolist()
    test_texts = [_strip_html_keep_text(x) for x in test_df["review"].astype(str).tolist()]

    n = len(train_texts)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_valid = int(round(n * float(args.valid_ratio)))
    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]

    train_ds = _TextDataset([train_texts[i] for i in train_idx], [y[i] for i in train_idx])
    valid_ds = _TextDataset([train_texts[i] for i in valid_idx], [y[i] for i in valid_idx])
    test_ds = _TextDataset(test_texts, None)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=args.offline)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    def encode_batch(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=args.max_length)
        if "labels" in batch:
            enc["labels"] = batch["labels"]
        return enc

    def collate_fn(features):
        texts = [f["text"] for f in features]
        enc = tokenizer(texts, truncation=True, max_length=args.max_length)
        if "labels" in features[0]:
            enc["labels"] = [f["labels"] for f in features]
        return collator(enc)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, local_files_only=args.offline
    )
    if args.grad_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False
            print("[train] gradient checkpointing enabled")
        except Exception:
            print("[train] gradient checkpointing not supported; skip")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * float(args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device.type == "cuda"))

    model.train()
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, desc=f"train e{epoch}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(args.fp16 and device.type == "cuda")):
                out = model(**batch)
                loss = out.loss / max(1, int(args.grad_accum))
            scaler.scale(loss).backward()

            if step % max(1, int(args.grad_accum)) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix(loss=float(loss.detach().cpu().item()) * max(1, int(args.grad_accum)))

    save_dir = args.artifacts_dir / "final"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[save] {save_dir}")

    if not args.no_eval and len(valid_ds) > 0:
        try:
            from sklearn.metrics import roc_auc_score
        except Exception:
            roc_auc_score = None

        model.eval()
        probs = []
        labels = []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="valid"):
                labels.extend(batch["labels"].tolist())
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=(args.fp16 and device.type == "cuda")):
                    logits = model(**batch).logits
                p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
                probs.append(p)
        p = np.concatenate(probs)
        if roc_auc_score is not None:
            auc = roc_auc_score(np.asarray(labels), p)
            print(f"[valid] AUC: {auc:.5f}")

    model.eval()
    test_probs = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="test"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(args.fp16 and device.type == "cuda")):
                logits = model(**batch).logits
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            test_probs.append(p)
    test_p = np.concatenate(test_probs)

    submission = pd.DataFrame({"id": test_df["id"].astype(str), "sentiment": test_p})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    print(f"[submit] wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
