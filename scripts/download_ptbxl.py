#!/usr/bin/env python
"""Download PTB-XL 1.0.3 (500 Hz) into data/raw/ptbxl.

The paper uses 12-lead 10 s / 500 Hz records, so records500 is used here.
PTB-XL is openly available on PhysioNet; no login required.

Usage:
  python scripts/download_ptbxl.py --workers 8
Resume: files that exist with matching size are skipped.
"""
import argparse
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://physionet.org/files/ptb-xl/1.0.3"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) pde-gan-repro"}

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "ptbxl"

META = ["ptbxl_database.csv", "scp_statements.csv", "RECORDS", "LICENSE.txt"]


def fetch(url: str, dst: Path, retries: int = 4) -> tuple[str, bool, str]:
    if dst.exists() and dst.stat().st_size > 0:
        return str(dst), True, "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            return str(dst), True, "ok"
        except Exception as e:  # noqa: BLE001
            if dst.exists():
                dst.unlink(missing_ok=True)
            if i == retries - 1:
                return str(dst), False, f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (i + 1))
    return str(dst), False, "unreachable"


def list_record_ids() -> list[str]:
    """Read 500 Hz record paths from the filename_hr column of ptbxl_database.csv.

    Note: the RECORDS metafile lists records100 only and cannot be used for 500 Hz downloads.
    """
    import csv

    csv_path = OUT / "ptbxl_database.csv"
    if not csv_path.exists():
        fetch(f"{BASE}/ptbxl_database.csv", csv_path)
    ids: list[str] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rel = row["filename_hr"].strip()  # e.g. records500/00000/00001_hr
            ids.append(rel.replace("records500/", ""))
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="download only the first N records (debug)")
    ap.add_argument("--ids-file", default="", help="download only the filename_hr entries listed in this file")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for m in META:
        fetch(f"{BASE}/{m}", OUT / m)

    if args.ids_file:
        ids = [ln.strip().replace("records500/", "")
               for ln in Path(args.ids_file).read_text().splitlines() if ln.strip()]
    else:
        ids = list_record_ids()
    if args.limit:
        ids = ids[: args.limit]
    print(f"[download] records={len(ids)} -> {OUT}", flush=True)

    jobs = []
    for rel in ids:
        for ext in (".hea", ".dat"):
            jobs.append((f"{BASE}/records500/{rel}{ext}", OUT / "records500" / f"{rel}{ext}"))

    t0 = time.time()
    ok = skip = fail = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch, u, p) for u, p in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            path, good, msg = fu.result()
            if good:
                if msg == "skip":
                    skip += 1
                else:
                    ok += 1
            else:
                fail += 1
                failures.append((path, msg))
            if i % 2000 == 0 or i == len(futs):
                el = time.time() - t0
                print(f"[{i}/{len(futs)}] ok={ok} skip={skip} fail={fail} {el/60:.1f} min", flush=True)

    if failures:
        (ROOT / "logs").mkdir(exist_ok=True)
        with open(ROOT / "logs" / "download_failures.txt", "w") as f:
            for p, m in failures:
                f.write(f"{p}\t{m}\n")
        print(f"[download] {len(failures)} failures -> logs/download_failures.txt", flush=True)
    print(f"[download] done in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
