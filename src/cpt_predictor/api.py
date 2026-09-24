"""Local website API for tib/fib vertical-load analysis."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .errors import StrengthAnalysisError
from .strength_pipeline import analyze_dicom_path


REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIST = REPO_ROOT / "web" / "dist"


def create_app(dist_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="OsteoVigil tib/fib strength")
    web_root = Path(dist_dir) if dist_dir is not None else WEB_DIST

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "analysis": "voxel_hexahedral_linear_elastic"}

    @app.post("/api/analyze")
    async def analyze(
        files: list[UploadFile] = File(...),
        body_mass_kg: float = Form(70.0),
        use_dicom_weight: str = Form("1"),
    ) -> JSONResponse:
        if not files:
            return JSONResponse({"error": "Choose a DICOM series or zip.", "modality": "unknown"}, status_code=422)
        prefer_header = str(use_dicom_weight).lower() in {"1", "true", "on", "yes"}
        workspace = Path(tempfile.mkdtemp(prefix="osteovigil-"))
        try:
            saved = []
            for upload in files:
                name = Path(upload.filename or "slice.dcm").name
                if not name or name in {".", ".."}:
                    continue
                target = workspace / name
                content = await upload.read()
                if not content:
                    continue
                target.write_bytes(content)
                saved.append(target)
            if not saved:
                return JSONResponse({"error": "The upload did not contain any DICOM bytes.", "modality": "unknown"}, status_code=422)
            source: Path = saved[0] if len(saved) == 1 else workspace
            try:
                report = analyze_dicom_path(source, body_mass_kg=body_mass_kg, prefer_dicom_weight=prefer_header)
            except StrengthAnalysisError as exc:
                return JSONResponse({"error": str(exc), "modality": exc.modality}, status_code=422)
            return JSONResponse(report.public_dict())
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    if (web_root / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=web_root / "assets"), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        page = web_root / "index.html"
        if not page.is_file():
            return FileResponse(path=_fallback_page())
        return FileResponse(page)

    return app


def _fallback_page() -> Path:
    path = Path(tempfile.gettempdir()) / "osteovigil-web-missing.html"
    path.write_text(
        "<!doctype html><title>OsteoVigil</title><p>Build the website with npm run build in the web directory.</p>",
        encoding="utf-8",
    )
    return path


app = create_app()


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
