"""G12EC (Georgia 12-lead ECG Challenge) -> per-beat cache.

Differs from preprocess.py in two places (beat cutting, normalisation and gating positions are shared):
    1) record traversal: Georgia/*.hea (E00001..E10344), wfdb reads the WFDB-16 payload inside .mat directly;
    2) labels: the `#Dx: <comma separated SNOMED-CT>` annotation in .hea, mapped to the 7 paper classes plus NORM.

Split (G12EC has no official folds):
    stratified by each record primary label; 20% held out as test (split=1), the rest split into 5 folds (0..4).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.labels import ALL_LABELS  # noqa: E402

# Official Challenge 2020 SNOMED-CT mapping (7 paper classes plus NORM)
SNOMED = {
    "IAVB": "270492004",
    "RBBB": "59118001",
    "LBBB": "164909002",
    "NSIVCB": "713426002",
    "LAnFB": "445118002",
    "QAb": "164951009",
    "AFL": "164889003",
    "NORM": "426783006",
}
CODE2CLS = {v: k for k, v in SNOMED.items()}


class Rec:
    __slots__ = ("name", "path", "hot", "primary", "filename_hr")

    def __init__(self, name, path, hot, primary):
        self.name, self.path, self.hot, self.primary = name, path, hot, primary
        self.filename_hr = name


def parse_header(hea: Path) -> tuple[np.ndarray, int]:
    hot = np.zeros(len(ALL_LABELS), np.uint8)
    dx = ""
    for line in hea.read_text(errors="ignore").splitlines():
        if line.startswith("#Dx:"):
            dx = line.split(":", 1)[1].strip()
            break
    for code in (c.strip() for c in dx.split(",") if c.strip()):
        name = CODE2CLS.get(code)
        if name is not None:
            hot[ALL_LABELS.index(name)] = 1
    primary = int(np.argmax(hot)) if hot.sum() else len(ALL_LABELS)   # all-zero -> other
    return hot, primary


def read_g12ec(hea: Path) -> np.ndarray:
    """Parse the MATLAB v4 container directly (WFDB-16 payload plus 24-byte header); return (n_samp, 12) in mV.

    G12EC .hea files put the date before the time (`05-May-2020 14:50:55`), which makes the wfdb header parser raise
    HeaderSyntaxError, so the header is parsed here; data is frame-interleaved (12 leads per frame).
    """
    import struct

    lines = [ln.strip() for ln in hea.read_text(errors="ignore").splitlines() if ln.strip()]
    f = lines[0].split()
    n_sig, n_samp = int(f[1]), int(f[3])
    rows = [ln.split() for ln in lines[1:1 + n_sig]]
    gains = np.asarray([float(r[2].split("/")[0].split("(")[0]) for r in rows], dtype=np.float64)
    baselines = np.asarray([float(r[4]) if len(r) > 4 else 0.0 for r in rows], dtype=np.float64)

    raw = hea.with_suffix(".mat").read_bytes()
    _, mrows, ncols, _, namelen = struct.unpack("<iiiii", raw[:20])
    data = np.frombuffer(raw[20 + namelen:], dtype="<i2")
    n = data.size // n_sig
    assert n >= n_samp, f"{hea.name}: {n} samples < {n_samp}"
    sig = data[: n * n_sig].reshape(n, n_sig).astype(np.float64)
    sig = (sig - baselines) / np.where(gains <= 0, 1.0, gains)
    return sig


def process_record_g12ec(rec: Rec, nk) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    from src.data.preprocess import POST_BEATS, PRE_BEATS, SR, zscore_record

    sig = read_g12ec(rec.path)
    if sig.shape[1] != 12 or sig.shape[0] < SR * 5:
        return None, None, None, 0
    if not np.isfinite(sig).all():
        sig = np.nan_to_num(sig, nan=0.0)
    sig = zscore_record(sig)
    lead = sig[:, 1]

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
        seg = sig[a:b].T
        if np.abs(seg).max() > 6.0:
            continue
        beats.append(seg.astype(np.float16))
        rrs.append((rpeaks[i] - rpeaks[i - 1]) / SR if i > 0 else np.nan)
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
    ap.add_argument("--data-root", default=str(ROOT / "datasets" / "Georgia"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "g12ec_beats.npz"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import neurokit2 as nk
    import wfdb
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

    root = Path(args.data_root)
    heas = sorted(root.glob("*.hea"))
    print(f"[g12] {len(heas)} records @ {root}", flush=True)
    if args.limit:
        heas = heas[: args.limit]

    recs: list[Rec] = []
    for h in heas:
        hot, primary = parse_header(h)
        recs.append(Rec(h.stem, h, hot, primary))

    y = np.asarray([r.primary for r in recs])
    # rare primary labels (<5 records) merged into other, otherwise stratified sampling fails
    vals, counts = np.unique(y, return_counts=True)
    rare = {int(v) for v, c in zip(vals, counts) if c < 5}
    if rare:
        print(f"[g12] rare primary labels merged into other: {sorted(rare)}", flush=True)
        y = np.asarray([8 if r.primary in rare else r.primary for r in recs])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    tr_idx, te_idx = next(sss.split(np.zeros(len(y)), y))
    fold = np.full(len(recs), -1, np.int8)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for f, (_, va) in enumerate(skf.split(tr_idx, y[tr_idx])):
        fold[tr_idx[va]] = f
    split = np.zeros(len(recs), np.uint8)
    split[te_idx] = 1
    print(f"[g12] test={len(te_idx)} records, cv folds={np.bincount(fold[fold >= 0]).tolist()}", flush=True)

    all_beats, all_labels, all_rec, all_fold, all_split, all_peaks, all_rr = [], [], [], [], [], [], []
    n_fail = 0
    for k in range(len(recs)):
        rec = recs[k]
        try:
            beats, peaks, rr, ok = process_record_g12ec(rec, nk)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            if n_fail <= 10:
                print(f"[g12] failed {rec.name}: {type(e).__name__}: {e}", flush=True)
            continue
        if not ok:
            n_fail += 1
            continue
        n = beats.shape[0]
        all_beats.append(beats)
        all_labels.append(np.tile(rec.hot, (n, 1)))
        all_rec.append(np.full(n, k, np.int32))
        all_fold.append(np.full(n, fold[k], np.int8))
        all_split.append(np.full(n, split[k], np.uint8))
        all_peaks.append(peaks)
        all_rr.append(rr)
        if (k + 1) % 500 == 0 or k + 1 == len(recs):
            nb = sum(b.shape[0] for b in all_beats)
            print(f"[g12] {k+1}/{len(recs)} records, {nb} beats, fail={n_fail}", flush=True)

    if not all_beats:
        print("[g12] nothing produced", file=sys.stderr)
        return 1

    beats = np.concatenate(all_beats)
    labels = np.concatenate(all_labels)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, beats=beats, labels=labels,
        rec_ids=np.concatenate(all_rec), fold=np.concatenate(all_fold),
        split=np.concatenate(all_split), peaks=np.concatenate(all_peaks),
        rr=np.concatenate(all_rr), classes=np.asarray(ALL_LABELS),
    )
    print(f"[g12] saved {out}  beats={beats.shape} labels={labels.shape}", flush=True)
    for i, c in enumerate(ALL_LABELS):
        print(f"      {c:8s} beats={int(labels[:, i].sum()):7d}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
