"""PTB-XL 500 Hz -> per-beat 12-lead cache (reproduction of the paper §4.4 preprocessing).

Pipeline per record:
    1) read 10 s / 12 leads / 500 Hz with wfdb
    2) per-record z-score (median / 1.4826*MAD, clipped to +/-8)
    3) detect R peaks on Lead II with NeuroKit2 (the paper specifies Lead II)
    4) cut all 12 leads around R (R-140 to R+260 samples, i.e. -0.28 s to +0.52 s, 400 samples)
    5) delineate P/Q/S/T with NeuroKit2 and store positions relative to R (seconds) for the gating masks

Output (data/processed/*.npz):
  beats   (N,12,L) float16
    labels  (N,8)    uint8   order defined in labels.ALL_LABELS
  rec_ids (N,)     int32
    fold    (N,)     uint8   0..4 (folded from strat_fold 1..10)
    peaks   (N,4)    float32 [P,Q,S,T] seconds relative to R, NaN = not detected
    rr      (N,)     float32 R-R interval of this beat (seconds)
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.labels import ALL_LABELS, CLASSES, codes_to_multihot, is_norm, parse_scp_codes  # noqa: E402

PRE_BEATS = 140    # samples before the R peak (500 Hz -> 0.28 s)
POST_BEATS = 260   # samples after the R peak (0.52 s)
WIN = PRE_BEATS + POST_BEATS
SR = 500
LEAD_II = 1


def record_priority(row: pd.Series) -> bool:
    """Drop paced and ectopic records (pacing breaks P-QRS semantics; ectopics make beat labels inconsistent)."""
    if row.get("pacemaker", 0) == 1:
        return False
    if row.get("extra_beats", 0) == 1:
        return False
    return True


def select_records(df: pd.DataFrame, per_class: int, seed: int = 42) -> pd.DataFrame:
    """Stratified sampling by target class; NORM has its own cap."""
    rng = np.random.default_rng(seed)
    picked: dict[int, str] = {}
    keep = []
    for idx, row in df.iterrows():
        codes = parse_scp_codes(row["scp_codes"])
        hot = codes_to_multihot(codes)
        if sum(hot) == 0 and not is_norm(codes):
            continue
        keep.append(idx)
    df = df.loc[keep].copy()
    df["_multihot"] = df["scp_codes"].map(lambda s: codes_to_multihot(parse_scp_codes(s)))
    df["_norm"] = df["scp_codes"].map(lambda s: is_norm(parse_scp_codes(s)))

    selected: list[int] = []
    used: set[int] = set()
    for ci, cname in enumerate(CLASSES):
        pool = [i for i in df.index if df.at[i, "_multihot"][ci] == 1 and i not in used]
        # keep all rare classes, truncate common ones
        take = pool[:per_class] if per_class > 0 else pool
        if len(pool) > per_class > 0:
            take = list(rng.choice(pool, size=per_class, replace=False))
        selected += list(take)
        used.update(take)
    pool_norm = [i for i in df.index if df.at[i, "_norm"] == 1 and i not in used]
    n_norm = min(len(pool_norm), max(per_class, 1) * 2)
    if len(pool_norm) > n_norm:
        pool_norm = list(rng.choice(pool_norm, size=n_norm, replace=False))
    selected += pool_norm
    out = df.loc[sorted(set(selected))].copy()
    out["_idx"] = out.index
    return out


def zscore_record(sig: np.ndarray) -> np.ndarray:
    """Per-record, per-lead z-score (median / 1.4826*MAD), clipped to +/-8."""
    med = np.median(sig, axis=0, keepdims=True)
    mad = np.median(np.abs(sig - med), axis=0, keepdims=True) * 1.4826
    mad = np.where(mad < 1e-6, 1.0, mad)
    return np.clip((sig - med) / mad, -8, 8) / 8.0


def process_record(rec, nk, wfdb, root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (beats (n,12,W), peaks (n,4), rr (n,), lead_ii_ok)."""
    path = root / rec.filename_hr
    sig, fields = wfdb.rdsamp(str(path))
    if sig is None or sig.shape[1] != 12 or sig.shape[0] < SR * 5:
        return None, None, None, 0
    sig = sig.astype(np.float64)
    if not np.isfinite(sig).all():
        sig = np.nan_to_num(sig, nan=0.0)
    sig = zscore_record(sig)
    lead = sig[:, LEAD_II]

    cleaned = nk.ecg_clean(lead, sampling_rate=SR, method="neurokit")
    _, info = nk.ecg_peaks(cleaned, sampling_rate=SR, method="neurokit", correct_artifacts=True)
    rpeaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    if len(rpeaks) < 3:
        return None, None, None, 0

    try:
        _, d = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=SR, method="dwt", show=False)
        p = np.asarray(d.get("ECG_P_Peaks", []), dtype=float)
        q = np.asarray(d.get("ECG_Q_Peaks", []), dtype=float)
        s = np.asarray(d.get("ECG_S_Peaks", []), dtype=float)
        t = np.asarray(d.get("ECG_T_Peaks", []), dtype=float)
    except Exception:  # noqa: BLE001
        p = q = s = t = np.full(len(rpeaks), np.nan)

    beats, peaks, rrs = [], [], []
    for i, r in enumerate(rpeaks):
        a, b = r - PRE_BEATS, r + POST_BEATS
        if a < 0 or b > sig.shape[0]:
            continue
        seg = sig[a:b].T  # (12, W)
        if np.abs(seg).max() > 6.0:      # extreme artefact
            continue
        beats.append(seg.astype(np.float16))
        rr = (rpeaks[i] - rpeaks[i - 1]) / SR if i > 0 else np.nan
        rrs.append(rr)
        rel = []
        for arr in (p, q, s, t):
            v = arr[i] if i < len(arr) else np.nan
            rel.append((v - r) / SR if np.isfinite(v) else np.nan)
        peaks.append(rel)

    if not beats:
        return None, None, None, 0
    return np.stack(beats), np.asarray(peaks, np.float32), np.asarray(rrs, np.float32), 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT / "data" / "raw" / "ptbxl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "ptbxl_beats.npz"))
    ap.add_argument("--records-csv", default=str(ROOT / "data" / "splits" / "selected_records.csv"),
                    help="output of select_records.py; when present its split/cv_fold take precedence")
    ap.add_argument("--per-class", type=int, default=60, help="max records per class when no records csv is given")
    ap.add_argument("--limit", type=int, default=0, help="debug: process only the first N records")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import neurokit2 as nk
    import wfdb

    root = Path(args.data_root)
    rec_csv = Path(args.records_csv)
    if rec_csv.exists():
        sel = pd.read_csv(rec_csv)
        sel = sel[sel["filename_hr"].map(lambda p: (root / p).with_suffix(".hea").exists())]
        print(f"[pre] read {len(sel)} downloaded records from {rec_csv.name}", flush=True)
    else:
        db = pd.read_csv(root / "ptbxl_database.csv")
        db = db[db["filename_hr"].notna()]
        db = db[db.apply(record_priority, axis=1)]
        db = db[db["filename_hr"].map(lambda p: (root / p).with_suffix(".hea").exists())]
        print(f"[pre] usable records {len(db)}", flush=True)
        sel = select_records(db, args.per_class, args.seed)
        sel["split"] = "train"
        sel["cv_fold"] = (sel["strat_fold"] - 1) % 5
    if args.limit:
        sel = sel.head(args.limit)
    print(f"[pre] records to process {len(sel)}", flush=True)

    all_beats, all_labels, all_rec, all_fold, all_split, all_peaks, all_rr = [], [], [], [], [], [], []
    n_fail = 0
    for k, (_, row) in enumerate(sel.iterrows(), 1):
        if rec_csv.exists():
            hot = [int(row["label_" + c]) for c in ALL_LABELS]
            is_test = str(row.get("split", "train")) == "test"
            fold = -1 if is_test else int(row.get("cv_fold", 0))
        else:
            codes = parse_scp_codes(row["scp_codes"])
            hot = codes_to_multihot(codes) + [is_norm(codes)]
            is_test, fold = False, int((int(row["strat_fold"]) - 1) % 5)
        try:
            beats, peaks, rr, ok = process_record(row, nk, wfdb, root)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            if n_fail <= 10:
                print(f"[pre] failed ecg_id={row['ecg_id']}: {type(e).__name__}: {e}", flush=True)
            continue
        if not ok:
            n_fail += 1
            continue
        n = beats.shape[0]
        all_beats.append(beats)
        all_labels.append(np.tile(np.asarray(hot, np.uint8), (n, 1)))
        all_rec.append(np.full(n, int(row["ecg_id"]), np.int32))
        all_fold.append(np.full(n, fold, np.int8))
        all_split.append(np.full(n, int(is_test), np.uint8))
        all_peaks.append(peaks)
        all_rr.append(rr)
        if k % 50 == 0 or k == len(sel):
            nb = sum(b.shape[0] for b in all_beats)
            print(f"[pre] {k}/{len(sel)} records, {nb} beats, fail={n_fail}", flush=True)

    if not all_beats:
        print("[pre] no beats produced", file=sys.stderr)
        return 1

    beats = np.concatenate(all_beats)
    labels = np.concatenate(all_labels)
    rec_ids = np.concatenate(all_rec)
    fold = np.concatenate(all_fold)
    split = np.concatenate(all_split)
    peaks = np.concatenate(all_peaks)
    rr = np.concatenate(all_rr)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, beats=beats, labels=labels, rec_ids=rec_ids, fold=fold,
                        split=split, peaks=peaks, rr=rr, classes=np.asarray(ALL_LABELS))
    print(f"[pre] saved {out}  beats={beats.shape} labels={labels.shape}", flush=True)
    for i, c in enumerate(ALL_LABELS):
        print(f"      {c:8s} beats={int(labels[:, i].sum()):7d} records={len(np.unique(rec_ids[labels[:, i] == 1]))}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
