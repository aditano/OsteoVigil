#!/usr/bin/env python3
"""Download public abnormal lower-leg CTs for OsteoVigil testing.

True congenital-pseudarthrosis-of-the-tibia CT series are not available in
open archives. The cases below are the closest public 3D CT volumes that
include tibia/fibula anatomy plus a documented lower-leg abnormality.

Volumes are written to data/downloaded/ and data/external/, which are
gitignored. Do not commit the DICOM files.

Collections:
  Soft-tissue-Sarcoma  DOI 10.7937/K9/TCIA.2015.7GO2GSKS  (CC BY 3.0)
  TCGA-SARC            DOI 10.7937/K9/TCIA.2016.CX6YLSUX  (CC BY 3.0)

Clinical tumor sites for Soft-tissue-Sarcoma come from INFOclinical_STS.xlsx
on the TCIA collection page.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Dict
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = REPO_ROOT / "data" / "downloaded"
EXTERNAL_DIR = REPO_ROOT / "data" / "external"
NBIA_IMAGE = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"

# Target-leg is patient left/right as labeled in the TCIA clinical spreadsheet.
CASES: Dict[str, dict] = {
    "sts010": {
        "collection": "Soft-tissue-Sarcoma",
        "patient_id": "STS_010",
        "site": "left calf myxofibrosarcoma",
        "target_leg": "left",
        "series_uid": "1.3.6.1.4.1.14519.5.2.1.5168.1900.197754415891602187397505258429",
        "zip_name": "sts010_legs.zip",
        "extract_dir": "sts010_legs",
        "why": "Documented left-calf sarcoma next to the tib/fib shaft.",
    },
    "sts025": {
        "collection": "Soft-tissue-Sarcoma",
        "patient_id": "STS_025",
        "site": "right calf malignant fibrous histiocytoma",
        "target_leg": "right",
        "series_uid": "1.3.6.1.4.1.14519.5.2.1.5168.1900.272904213121500367633063763460",
        "zip_name": "sts025_legs.zip",
        "extract_dir": "sts025_legs",
        "why": "Documented right-calf sarcoma; contralateral shaft is a same-scan control.",
    },
    "sts028": {
        "collection": "Soft-tissue-Sarcoma",
        "patient_id": "STS_028",
        "site": "left calf extraskeletal high-grade osteogenic sarcoma",
        "target_leg": "left",
        "series_uid": "1.3.6.1.4.1.14519.5.2.1.5168.1900.913590972108839969190922752912",
        "zip_name": "sts028_legs.zip",
        "extract_dir": "sts028_legs",
        "why": "Best public bone-adjacent calf case (osteogenic sarcoma).",
    },
    "sts032": {
        "collection": "Soft-tissue-Sarcoma",
        "patient_id": "STS_032",
        "site": "right calf pleomorphic liposarcoma",
        "target_leg": "right",
        "series_uid": "1.3.6.1.4.1.14519.5.2.1.5168.1900.225389768618855362748822146086",
        "zip_name": "sts032_legs.zip",
        "extract_dir": "sts032_legs",
        "why": "Additional right-calf sarcoma CT.",
    },
    "sts042": {
        "collection": "Soft-tissue-Sarcoma",
        "patient_id": "STS_042",
        "site": "left knee synovial sarcoma",
        "target_leg": "left",
        "series_uid": "1.3.6.1.4.1.14519.5.2.1.5168.1900.794918299994114817879846530899",
        "zip_name": "sts042_legs.zip",
        "extract_dir": "sts042_legs",
        "why": "Proximal-tibia / knee-region sarcoma.",
    },
    "sts049": {
        "collection": "Soft-tissue-Sarcoma",
        "patient_id": "STS_049",
        "site": "right calf round-cell liposarcoma",
        "target_leg": "right",
        "series_uid": "1.3.6.1.4.1.14519.5.2.1.5168.1900.386262602559235766910800931996",
        "zip_name": "sts049_legs.zip",
        "extract_dir": "sts049_legs",
        "why": "Additional right-calf sarcoma CT.",
    },
    "sts051": {
        "collection": "Soft-tissue-Sarcoma",
        "patient_id": "STS_051",
        "site": "left popliteal fossa synovial sarcoma",
        "target_leg": "left",
        "series_uid": "1.3.6.1.4.1.14519.5.2.1.5168.1900.186653801311189416337649712538",
        "zip_name": "sts051_legs.zip",
        "extract_dir": "sts051_legs",
        "why": "Popliteal / proximal-tib-fib sarcoma on a dedicated legs CT.",
    },
    "tcga_a8vf": {
        "collection": "TCGA-SARC",
        "patient_id": "TCGA-QQ-A8VF",
        "site": "lower-limb sarcoma (BodyPartExamined=LOWERLIMB)",
        "target_leg": "right",
        "series_uid": "1.3.6.1.4.1.14519.5.2.1.3023.4024.303123119074665705294749259172",
        "zip_name": "tcga_a8vf_lowerlimb_ct.zip",
        "extract_dir": "tcga_a8vf_lowerlimb",
        "why": "Single-limb lower-extremity CT with a shank soft-tissue mass.",
    },
}

DEFAULT_CASES = ("sts028", "sts010", "sts025", "tcga_a8vf")


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"  -> {dest}")
    with urlopen(url, timeout=180) as response, dest.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    print(f"  saved {dest.stat().st_size / (1024 * 1024):.1f} MB")


def _extract(zip_path: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)
    dcm_count = sum(1 for path in dest.rglob("*.dcm"))
    print(f"  extracted {dcm_count} DICOM files to {dest}")
    return dcm_count


def download_case(name: str, force: bool = False) -> Path:
    if name not in CASES:
        raise KeyError(f"Unknown case {name!r}. Choose from: {', '.join(CASES)}")
    spec = CASES[name]
    zip_path = DOWNLOAD_DIR / spec["zip_name"]
    extract_dir = EXTERNAL_DIR / spec["extract_dir"]
    if force or not zip_path.exists() or zip_path.stat().st_size < 1024 * 1024:
        url = f"{NBIA_IMAGE}?SeriesInstanceUID={spec['series_uid']}"
        _download(url, zip_path)
    else:
        print(f"Using cached {zip_path}")
    if force or not any(extract_dir.rglob("*.dcm")):
        _extract(zip_path, extract_dir)
    else:
        print(f"Using extracted {extract_dir}")
    return extract_dir


def list_cases() -> None:
    print("Public abnormal lower-leg CT cases (not CPT; see README notes):\n")
    for name, spec in CASES.items():
        default = "  [default]" if name in DEFAULT_CASES else ""
        print(f"{name}{default}")
        print(f"  {spec['patient_id']}  {spec['collection']}")
        print(f"  site: {spec['site']}")
        print(f"  why:  {spec['why']}")
        print(f"  run:  python main.py --dicom-dir data/external/{spec['extract_dir']} "
              f"--target-leg {spec['target_leg']} --internal-fea "
              f"--output-dir outputs/{name}")
        print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        choices=sorted(CASES),
        help="Case id to download. Repeatable. Defaults to the calf-focused set.",
    )
    parser.add_argument("--all", action="store_true", help="Download every catalogued case.")
    parser.add_argument("--list", action="store_true", help="Print the catalogue and exit.")
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        list_cases()
        return 0
    names = list(CASES) if args.all else (args.cases or list(DEFAULT_CASES))
    for name in names:
        spec = CASES[name]
        print(f"=== {name}: {spec['patient_id']} ({spec['site']}) ===")
        download_case(name, force=args.force)
    print("Done. DICOM folders are gitignored under data/external/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
