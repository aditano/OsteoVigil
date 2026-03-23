from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
MANAGED_FEBIO_ROOT_REL = Path(".third_party") / "febio"
MANAGED_METADATA_NAME = "install_metadata.json"
GITHUB_API_LATEST_RELEASE = "https://api.github.com/repos/febiosoftware/FEBio/releases/latest"
DEFAULT_CMAKE_PACKAGES = ("cmake>=3.28", "ninja>=1.11")
FEBIO_EXECUTABLE_NAMES = ("febio4", "febio4.exe", "febio3", "febio3.exe")


def _repo_root(repo_root: Optional[Path] = None) -> Path:
    return Path(repo_root) if repo_root else DEFAULT_REPO_ROOT


def managed_febio_root(repo_root: Optional[Path] = None) -> Path:
    return _repo_root(repo_root) / MANAGED_FEBIO_ROOT_REL


def managed_febio_metadata_path(repo_root: Optional[Path] = None) -> Path:
    return managed_febio_root(repo_root) / MANAGED_METADATA_NAME


def _log(message: str, verbose: bool) -> None:
    if verbose:
        print(f"[febio] {message}")


def _read_metadata(repo_root: Optional[Path] = None) -> dict[str, Any]:
    metadata_path = managed_febio_metadata_path(repo_root)
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_metadata(data: dict[str, Any], repo_root: Optional[Path] = None) -> None:
    metadata_path = managed_febio_metadata_path(repo_root)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _is_candidate_executable(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name.lower() not in FEBIO_EXECUTABLE_NAMES:
        return False
    if os.name == "nt":
        return path.suffix.lower() == ".exe"
    return os.access(path, os.X_OK)


def _find_febio_executable(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    for candidate_name in FEBIO_EXECUTABLE_NAMES:
        direct = root / candidate_name
        if _is_candidate_executable(direct):
            return direct
    for candidate in root.rglob("*"):
        if _is_candidate_executable(candidate):
            return candidate
    return None


def resolve_managed_febio_executable(repo_root: Optional[Path] = None) -> Optional[Path]:
    metadata = _read_metadata(repo_root)
    configured_path = metadata.get("executable_path")
    if configured_path:
        candidate = Path(configured_path)
        if candidate.exists():
            return candidate
    return _find_febio_executable(managed_febio_root(repo_root))


def _request_json(url: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "OsteoVigil-FEBio-Bootstrap",
        },
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        url,
        headers={
            "Accept": "application/octet-stream",
            "User-Agent": "OsteoVigil-FEBio-Bootstrap",
        },
    )
    with urlopen(request, timeout=300) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return destination


def _extract_archive(archive_path: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if archive_path.name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
    elif archive_path.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(destination)
    else:
        raise RuntimeError(f"Unsupported archive format: {archive_path.name}")

    entries = [entry for entry in destination.iterdir()]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return destination


def _current_platform_tokens() -> tuple[list[str], list[str]]:
    system = platform.system().lower()
    machine = platform.machine().lower()

    platform_tokens = {
        "windows": ["windows", "win"],
        "darwin": ["mac", "macos", "osx", "darwin"],
        "linux": ["linux", "ubuntu"],
    }.get(system, [system])

    arch_tokens = {
        "x86_64": ["x64", "amd64", "x86_64"],
        "amd64": ["x64", "amd64", "x86_64"],
        "arm64": ["arm64", "aarch64", "apple-silicon"],
        "aarch64": ["arm64", "aarch64", "apple-silicon"],
    }.get(machine, [machine])

    return platform_tokens, arch_tokens


def _score_release_asset(asset: dict[str, Any]) -> int:
    name = str(asset.get("name", "")).lower()
    if not name or "source code" in name:
        return -1
    if not name.endswith((".zip", ".tar.gz", ".tgz")):
        return -1

    platform_tokens, arch_tokens = _current_platform_tokens()
    score = 0
    if any(token in name for token in platform_tokens):
        score += 10
    else:
        return -1
    if any(token in name for token in arch_tokens):
        score += 5
    if "sdk" in name:
        score += 1
    if "release" in name:
        score += 1
    return score


def _select_release_asset(release: dict[str, Any]) -> Optional[dict[str, Any]]:
    best_asset: Optional[dict[str, Any]] = None
    best_score = -1
    for asset in release.get("assets", []):
        score = _score_release_asset(asset)
        if score > best_score:
            best_score = score
            best_asset = asset
    return best_asset


def _venv_bin_dir(python_executable: Path) -> Path:
    return python_executable.resolve().parent


def _find_tool(tool_name: str, python_executable: Path) -> Optional[str]:
    search_paths = [str(_venv_bin_dir(python_executable))]
    path_env = os.environ.get("PATH", "")
    if path_env:
        search_paths.extend(path_env.split(os.pathsep))
    return shutil.which(tool_name, path=os.pathsep.join(search_paths))


def _run_command(command: list[str], cwd: Path, verbose: bool) -> None:
    _log(f"Running: {' '.join(command)}", verbose)
    subprocess.check_call(command, cwd=cwd)


def _ensure_build_tools(python_executable: Path, verbose: bool) -> tuple[str, Optional[str]]:
    cmake_path = _find_tool("cmake", python_executable)
    ninja_path = _find_tool("ninja", python_executable)
    if cmake_path and ninja_path:
        return cmake_path, ninja_path

    _log("Installing local build helpers for FEBio (cmake, ninja)...", verbose)
    _run_command(
        [str(python_executable), "-m", "pip", "install", *DEFAULT_CMAKE_PACKAGES],
        cwd=_repo_root(),
        verbose=verbose,
    )

    cmake_path = _find_tool("cmake", python_executable)
    ninja_path = _find_tool("ninja", python_executable)
    if not cmake_path:
        raise RuntimeError("Unable to locate cmake after installing it into the local environment.")
    return cmake_path, ninja_path


def _record_install(
    executable_path: Path,
    *,
    install_method: str,
    release: dict[str, Any],
    repo_root: Optional[Path] = None,
) -> None:
    metadata = {
        "version": release.get("tag_name") or release.get("name") or "unknown",
        "release_name": release.get("name") or release.get("tag_name") or "unknown",
        "install_method": install_method,
        "executable_path": str(executable_path),
        "installed_at_utc": datetime.now(timezone.utc).isoformat(),
        "managed_root": str(managed_febio_root(repo_root)),
    }
    _write_metadata(metadata, repo_root=repo_root)


def _install_from_release_asset(
    release: dict[str, Any],
    *,
    repo_root: Optional[Path],
    verbose: bool,
) -> Optional[Path]:
    asset = _select_release_asset(release)
    if not asset:
        return None

    install_root = managed_febio_root(repo_root)
    downloads_dir = install_root / "downloads"
    extracts_dir = install_root / "assets" / str(release.get("tag_name", "latest"))
    archive_path = downloads_dir / str(asset["name"])
    browser_download_url = str(asset.get("browser_download_url") or "")
    if not browser_download_url:
        return None

    _log(f"Downloading FEBio release asset {asset['name']}...", verbose)
    _download_file(browser_download_url, archive_path)
    extracted_root = _extract_archive(archive_path, extracts_dir)
    executable = _find_febio_executable(extracted_root)
    if not executable:
        _log("No FEBio executable was found inside the downloaded release asset.", verbose)
        return None

    _record_install(executable, install_method="release_asset", release=release, repo_root=repo_root)
    return executable


def _download_source_archive(release: dict[str, Any], destination: Path, verbose: bool) -> Path:
    tag = str(release.get("tag_name") or "").strip()
    if not tag:
        raise RuntimeError("Latest FEBio release did not include a tag name.")
    source_url = f"https://github.com/febiosoftware/FEBio/archive/refs/tags/{tag}.tar.gz"
    _log(f"Downloading FEBio source archive for {tag}...", verbose)
    return _download_file(source_url, destination)


def _install_from_source(
    release: dict[str, Any],
    *,
    python_executable: Path,
    repo_root: Optional[Path],
    verbose: bool,
) -> Path:
    install_root = managed_febio_root(repo_root)
    tag = str(release.get("tag_name") or "latest")
    downloads_dir = install_root / "downloads"
    source_archive = downloads_dir / f"{tag}.tar.gz"
    source_parent = install_root / "source" / tag
    build_dir = install_root / "build" / tag / f"{platform.system().lower()}-{platform.machine().lower()}"

    _download_source_archive(release, source_archive, verbose)
    extracted_source_root = _extract_archive(source_archive, source_parent)
    if build_dir.exists():
        shutil.rmtree(build_dir)

    cmake_path, ninja_path = _ensure_build_tools(python_executable, verbose)

    configure_cmd = [
        cmake_path,
        "-S",
        str(extracted_source_root),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DUSE_MKL=OFF",
    ]
    if ninja_path:
        configure_cmd.extend(["-G", "Ninja"])

    _run_command(configure_cmd, cwd=_repo_root(repo_root), verbose=verbose)
    _run_command(
        [cmake_path, "--build", str(build_dir), "--config", "Release", "--parallel"],
        cwd=_repo_root(repo_root),
        verbose=verbose,
    )

    executable = _find_febio_executable(build_dir)
    if not executable:
        raise RuntimeError("Built FEBio successfully, but could not locate the FEBio executable in the build tree.")

    _record_install(executable, install_method="source_build", release=release, repo_root=repo_root)
    return executable


def ensure_febio_available(
    *,
    python_executable: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    force_reinstall: bool = False,
    prefer_release_assets: bool = True,
    verbose: bool = True,
) -> Path:
    python_executable = Path(python_executable or os.sys.executable)
    existing = resolve_managed_febio_executable(repo_root)
    if existing and not force_reinstall:
        _log(f"Using managed FEBio at {existing}", verbose)
        return existing

    release = _request_json(GITHUB_API_LATEST_RELEASE)
    install_errors: list[str] = []

    if prefer_release_assets:
        try:
            installed = _install_from_release_asset(release, repo_root=repo_root, verbose=verbose)
            if installed:
                _log(f"Installed FEBio from release asset at {installed}", verbose)
                return installed
        except (HTTPError, URLError, RuntimeError, zipfile.BadZipFile, tarfile.TarError) as exc:
            install_errors.append(f"release asset install failed: {exc}")

    try:
        installed = _install_from_source(
            release,
            python_executable=python_executable,
            repo_root=repo_root,
            verbose=verbose,
        )
        _log(f"Installed FEBio from source at {installed}", verbose)
        return installed
    except Exception as exc:
        install_errors.append(f"source build failed: {exc}")

    raise RuntimeError("; ".join(install_errors) or "Unable to install FEBio automatically.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download or build FEBio into the repo-local .third_party/febio directory."
    )
    parser.add_argument("--force", action="store_true", help="Force a fresh FEBio install attempt.")
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Skip release-asset detection and build FEBio from source.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce installer logging.")
    return parser


def cli(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    executable = ensure_febio_available(
        python_executable=Path(os.sys.executable),
        repo_root=DEFAULT_REPO_ROOT,
        force_reinstall=bool(args.force),
        prefer_release_assets=not bool(args.source_only),
        verbose=not bool(args.quiet),
    )
    print(executable)
    return 0
