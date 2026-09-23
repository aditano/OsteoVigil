import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_public_abnormal_cts import CASES, DEFAULT_CASES, main


def test_catalog_has_calf_defaults_and_valid_uids():
    assert set(DEFAULT_CASES).issubset(CASES)
    for name, spec in CASES.items():
        assert spec["series_uid"].startswith("1.")
        assert spec["extract_dir"]
        assert spec["target_leg"] in {"auto", "left", "right"}
        assert spec["patient_id"]
        assert name.replace("_", "") in spec["extract_dir"].replace("_", "") or spec["patient_id"].split("-")[-1].lower() in spec["extract_dir"]


def test_list_cases_exits_zero(capsys):
    assert main(["--list"]) == 0
    captured = capsys.readouterr()
    assert "sts028" in captured.out
    assert "TCGA-QQ-A8VF" in captured.out
