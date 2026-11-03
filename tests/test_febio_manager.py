from __future__ import annotations

import io
import json
import os
import platform
import tarfile
from pathlib import Path

import pytest

from cpt_predictor.utils.febio_manager import (
    _extract_archive,
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


def test_extract_archive_rejects_tar_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "payload.tar.gz"
    destination = tmp_path / "out"
    outside = tmp_path / "pwned.txt"
    with tarfile.open(archive_path, "w:gz") as archive:
        data = b"traversed"
        info = tarfile.TarInfo(name="../pwned.txt")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))

    with pytest.raises(RuntimeError, match="escapes destination"):
        _extract_archive(archive_path, destination)
    assert not outside.exists()


def test_extract_archive_allows_nested_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "ok.tar.gz"
    destination = tmp_path / "out"
    with tarfile.open(archive_path, "w:gz") as archive:
        data = b"febio"
        info = tarfile.TarInfo(name="pkg/febio4")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))

    extracted = _extract_archive(archive_path, destination)
    assert (extracted / "febio4").read_bytes() == b"febio" or (destination / "pkg" / "febio4").read_bytes() == b"febio"
