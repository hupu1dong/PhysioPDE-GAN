"""Stratified record selection from the PTB-XL metadata, producing a download list and splits.

Split protocol (paper §4.1):
    * 20% of patients form the independent test set (strat_fold 9/10)
    * the remaining 80% are split into 5 cross-validation folds with no patient overlap
    * paced and extra-beat records are excluded

Output:
    data/splits/selected_records.csv   selected records (fold / split / labels)
    data/splits/download_ids.txt       filename_hr entries to download
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.labels import ALL_LABELS, CLASSES, codes_to_multihot, is_norm, parse_scp_codes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT / "data" / "raw" / "ptbxl"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "splits"))
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--norm-records", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.data_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    db = pd.read_csv(root / "ptbxl_database.csv")
    db = db[db["filename_hr"].notna()].copy()
    n0 = len(db)
    db = db[(db.get("pacemaker", 0) != 1) & (db.get("extra_beats", 0) != 1)]
    print(f"[select] records {n0} -> {len(db)} after removing paced/ectopic")

    codes = db["scp_codes"].map(parse_scp_codes)
    hot = np.stack(codes.map(lambda c: codes_to_multihot(c)).to_numpy())          # (R,7)
    norm = codes.map(is_norm).to_numpy().astype(np.uint8)
    db["label_" + ALL_LABELS[-1]] = norm
    for i, c in enumerate(CLASSES):
        db["label_" + c] = hot[:, i]

    # patient-level split: strat_fold 9/10 -> test
    db["split"] = np.where(db["strat_fold"].isin([9, 10]), "test", "train")
    # random 5-fold split within the training pool, by patient
    train_pat = db.loc[db["split"] == "train", "patient_id"].unique()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(train_pat)
    pat2fold = {p: i % 5 for i, p in enumerate(train_pat)}
    db["cv_fold"] = db["patient_id"].map(pat2fold)
    db.loc[db["split"] == "test", "cv_fold"] = -1

    # stratified sampling so every target class appears in both train and test
    label_cols = ["label_" + c for c in ALL_LABELS]
    rng2 = np.random.default_rng(args.seed + 1)
    chosen: list[int] = []
    for c in CLASSES:
        col = "label_" + c
        for sp in ("train", "test"):
            pool = db[(db[col] == 1) & (db["split"] == sp)].index.tolist()
            n = args.per_class if sp == "train" else max(args.per_class // 4, 30)
            if len(pool) > n:
                pool = rng2.choice(pool, size=n, replace=False).tolist()
            chosen += pool
    pool = db[(db["label_NORM"] == 1)].index.tolist()
    n_norm = min(len(pool), args.norm_records)
    chosen += rng2.choice(pool, size=n_norm, replace=False).tolist() if len(pool) > n_norm else pool
    sel = db.loc[sorted(set(chosen))].copy()

    # a patient must not appear in both train and test (strat_fold guarantees this; re-checked here)
    overlap = set(sel.loc[sel.split == "train", "patient_id"]) & set(sel.loc[sel.split == "test", "patient_id"])
    assert not overlap, f"patient leakage across splits: {len(overlap)}"

    sel = sel.sort_values(["split", "cv_fold", "ecg_id"])
    sel_out = sel[["ecg_id", "patient_id", "strat_fold", "split", "cv_fold", "filename_hr"] + label_cols]
    csv_path = out / "selected_records.csv"
    sel_out.to_csv(csv_path, index=False)
    (out / "download_ids.txt").write_text("\n".join(sel["filename_hr"].tolist()))

    print(f"[select] selected {len(sel)} records -> {csv_path}")
    print(f"  train={int((sel.split=='train').sum())}  test={int((sel.split=='test').sum())}")
    for c in ALL_LABELS:
        sub = sel[sel["label_" + c] == 1]
        print(f"  {c:8s} records={len(sub):5d} train={int((sub.split=='train').sum()):4d} test={int((sub.split=='test').sum()):4d}")
    print(f"  folds: {sel[sel.split=='train'].groupby('cv_fold').size().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
