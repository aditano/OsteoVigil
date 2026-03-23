from __future__ import annotations

import json
import os
import platform
from pathlib import Path

from cpt_predictor.utils.febio_manager import (
    _select_release_asset,
    managed_febio_metadata_path,
    managed_febio_root,
    resolve_managed_febio_executable,
)


def test_select_release_asset_prefers_platform_binary() -> None:
    system = platform.system().lower()
    platform_label = {
        "windows": "windows",
        "darwin": "macos",
        "linux": "linux",
    }.get(system, system)
    release = {
        "assets": [
            {"name": "Source code (zip)", "browser_download_url": "https://example.com/source.zip"},
            {
                "name": f"FEBio-4.12-{platform_label}-x64.zip",
                "browser_download_url": "https://example.com/platform.zip",
            },
        ]
    }
    asset = _select_release_asset(release)
    assert asset is not None
    assert "source code" not in asset["name"].lower()


def test_resolve_managed_febio_executable_uses_metadata(tmp_path: Path) -> None:
    repo_root = tmp_path
    managed_root = managed_febio_root(repo_root)
    executable = managed_root / "custom" / ("febio4.exe" if os.name == "nt" else "febio4")
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.write_text("binary", encoding="utf-8")
    metadata_path = managed_febio_metadata_path(repo_root)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps({"executable_path": str(executable)}), encoding="utf-8")

    resolved = resolve_managed_febio_executable(repo_root)
    assert resolved == executable
