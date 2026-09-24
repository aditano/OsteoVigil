#!/usr/bin/env python3
"""Try to fetch one open full lower-limb CT, and skip it when the files are huge.

The Virtual Skeleton Database mirror on Zenodo (10.5281/zenodo.8270365) is
CC BY-NC-SA. Each whole-body subject is about 1 GB, so this script does not
download or commit it. The committed real scan remains the CC0 distal
tibia/fibula/ankle series in data/demo/normal_real_talocrural.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
ZENODO_RECORD = "https://zenodo.org/api/records/8270365"
MAX_BYTES = 400 * 1024 * 1024
CC0_DISTAL = REPO_ROOT / "data" / "demo" / "normal_real_talocrural"


def fetch_record_sizes(url: str = ZENODO_RECORD, timeout: float = 30.0) -> list[int]:
    with urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    sizes = []
    for item in payload.get("files", []):
        size = item.get("size") or item.get("filesize")
        if size is not None:
            sizes.append(int(size))
    return sizes


def plan_download(sizes: Iterable[int], max_bytes: int = MAX_BYTES, min_bytes: int = 1_000_000) -> str:
    """Decide from listed file sizes. Tiny placeholder files are ignored."""
    values = [int(size) for size in sizes if int(size) >= int(min_bytes)]
    if not values:
        return "skip-unknown"
    if min(values) > max_bytes:
        return "skip-too-large"
    return "download"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check whether a full lower-limb CT is small enough to download.")
    parser.add_argument("--dry-run", action="store_true", help="Print the decision without contacting Zenodo.")
    parser.add_argument("--sizes-mb", type=float, nargs="*", default=None, help="Override file sizes, in megabytes.")
    args = parser.parse_args(argv)
    print(f"CC0 distal tib/fib series already in the repo: {CC0_DISTAL}")
    if args.dry_run and args.sizes_mb is None:
        print("Dry run: VSD whole-body CT files are multi-gigabyte and are not downloaded.")
        print("Decision: skip-too-large")
        return 0
    try:
        sizes = [int(value * 1024 * 1024) for value in args.sizes_mb] if args.sizes_mb is not None else fetch_record_sizes()
    except Exception as exc:
        print(f"Could not read the Zenodo record ({exc}).")
        print("Decision: skip-unknown")
        print("Using the committed CC0 distal scan plus the analytical full-shaft phantom.")
        return 0
    decision = plan_download(sizes)
    candidates = [size for size in sizes if size >= 1_000_000]
    smallest = min(candidates) if candidates else (min(sizes) if sizes else 0)
    print(f"Smallest CT-sized file: {smallest / (1024 * 1024):.0f} MB")
    print(f"Decision: {decision}")
    if decision != "download":
        print("Not downloading. Whole-body CT stays out of the git tree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
