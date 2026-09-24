import { analyzeInBrowser } from "./browserSolver";
import {
  decodeVolume,
  modalityLabel,
  paintProjection,
  projectWeakness,
  projectWithWebGPU,
  projectionsAgree,
  type PaintMode,
  type Projection,
} from "./project";

type ScanKind = "ct" | "mri" | "radiograph" | "unknown";

type StrengthResponse = {
  modality: ScanKind;
  method: string;
  body_mass_kg: number;
  failure_load_n: number;
  failure_load_bodyweights: number;
  percent_vs_normal: number;
  comparison: "weaker" | "stronger" | "matched";
  walking_text: string;
  field_note: string;
  assumptions: string[];
  hotspot_z_mm: number | null;
  solver: string;
  views: {
    ap: { acquired: boolean };
    lateral: { acquired: boolean };
  };
  weakness?: {
    shape: number[];
    spacing_mm: number[];
    b64: string;
  };
  rasters?: Record<string, { shape: number[]; b64: string }>;
  error?: string;
};

function requiredElement<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) {
    throw new Error(`Missing #${id}`);
  }
  return node as T;
}

const form = requiredElement<HTMLFormElement>("analyze-form");
const fileInput = requiredElement<HTMLInputElement>("files");
const drop = requiredElement<HTMLLabelElement>("drop");
const dropLabel = requiredElement<HTMLSpanElement>("drop-label");
const massInput = requiredElement<HTMLInputElement>("mass");
const weightToggle = requiredElement<HTMLInputElement>("use-dicom-weight");
const runButton = requiredElement<HTMLButtonElement>("run");
const statusLine = requiredElement<HTMLParagraphElement>("status");
const results = requiredElement<HTMLElement>("results");
const modalityLine = requiredElement<HTMLParagraphElement>("modality");
const percentLine = requiredElement<HTMLParagraphElement>("percent");
const metrics = requiredElement<HTMLDListElement>("metrics");
const walkingLine = requiredElement<HTMLParagraphElement>("walking");
const fieldLine = requiredElement<HTMLParagraphElement>("field");
const assumptions = requiredElement<HTMLUListElement>("assumptions");
const apCanvas = requiredElement<HTMLCanvasElement>("ap");
const lateralCanvas = requiredElement<HTMLCanvasElement>("lateral");
const apState = requiredElement<HTMLSpanElement>("ap-state");
const lateralState = requiredElement<HTMLSpanElement>("lateral-state");
const rendererLine = requiredElement<HTMLParagraphElement>("renderer");
const engineNote = requiredElement<HTMLParagraphElement>("engine-note");

const BROWSER_SOLVER_NOTE = "The solver runs in this browser. The DICOM stays on this computer.";
let useBrowserSolver = location.hostname.endsWith("github.io");

function setStatus(text: string): void {
  statusLine.textContent = text;
}

function comparisonSentence(report: StrengthResponse): string {
  const percent = report.percent_vs_normal.toFixed(1);
  switch (report.comparison) {
    case "weaker":
      return `${percent}% weaker than a normal leg`;
    case "stronger":
      return `${percent}% stronger than a normal leg`;
    case "matched":
      return "0.0% difference from a normal leg under the same vertical load";
    default: {
      const neverWord: never = report.comparison;
      return neverWord;
    }
  }
}

function rasterProjection(shape: number[], b64: string): Projection | null {
  if (shape.length < 2 || shape[0] < 1 || shape[1] < 1) {
    return null;
  }
  const decoded = decodeVolume([1, shape[0], shape[1]], [1, 1, 1], b64);
  if (decoded.values.length < shape[0] * shape[1]) {
    return null;
  }
  return {
    values: decoded.values,
    width: shape[1],
    height: shape[0],
    spacing: [1, 1],
  };
}

function rasterPair(report: StrengthResponse): { ap: Projection | null; lateral: Projection | null; renderer: string; gpuMatch: string } {
  const apRaster = report.rasters?.ap;
  const lateralRaster = report.rasters?.lateral;
  return {
    ap: apRaster ? rasterProjection(apRaster.shape, apRaster.b64) : null,
    lateral: lateralRaster ? rasterProjection(lateralRaster.shape, lateralRaster.b64) : null,
    renderer: "cpu",
    gpuMatch: "n/a",
  };
}

async function projectionsFor(report: StrengthResponse): Promise<{ ap: Projection | null; lateral: Projection | null; renderer: string; gpuMatch: string }> {
  if (report.modality === "radiograph") {
    return rasterPair(report);
  }
  if (report.weakness) {
    const volume = decodeVolume(report.weakness.shape, report.weakness.spacing_mm, report.weakness.b64);
    const cpu = projectWeakness(volume);
    const gpu = await projectWithWebGPU(volume);
    if (gpu && projectionsAgree(cpu.ap.values, gpu.ap.values) && projectionsAgree(cpu.lateral.values, gpu.lateral.values)) {
      return { ap: gpu.ap, lateral: gpu.lateral, renderer: "webgpu", gpuMatch: "true" };
    }
    return { ap: cpu.ap, lateral: cpu.lateral, renderer: "cpu", gpuMatch: gpu ? "false" : "n/a" };
  }
  return rasterPair(report);
}

function paintMode(report: StrengthResponse): PaintMode {
  switch (report.modality) {
    case "radiograph":
      return "radiograph";
    case "ct":
    case "mri":
    case "unknown":
      return "weakness";
    default: {
      const neverKind: never = report.modality;
      return neverKind;
    }
  }
}

function paintPair(report: StrengthResponse, ap: Projection | null, lateral: Projection | null, distalAtBottom: boolean): void {
  const apAcquired = report.views.ap.acquired && ap !== null;
  const lateralAcquired = report.views.lateral.acquired && lateral !== null;
  const mode = paintMode(report);
  apState.textContent = apAcquired ? "" : "not in this study";
  lateralState.textContent = lateralAcquired ? "" : "not in this study";
  paintProjection(apCanvas, ap, apAcquired, "No AP view in this study.", distalAtBottom, mode);
  paintProjection(lateralCanvas, lateral, lateralAcquired, "No lateral view in this study.", distalAtBottom, mode);
}

function showReport(report: StrengthResponse, ap: Projection | null, lateral: Projection | null, renderer: string, gpuMatch: string): void {
  results.hidden = false;
  modalityLine.textContent = `${modalityLabel(report.modality)} · ${report.method.replaceAll("_", " ")}`;
  percentLine.textContent = comparisonSentence(report);
  const hotspot = report.hotspot_z_mm === null ? "none localized" : `${report.hotspot_z_mm.toFixed(1)} mm from the distal end`;
  const rows: Array<[string, string]> = [
    ["Failure load", `${Math.round(report.failure_load_n).toLocaleString()} N`],
    ["Body weights", `${report.failure_load_bodyweights.toFixed(2)}×`],
    ["Body mass", `${report.body_mass_kg.toFixed(1)} kg`],
    ["Hotspot", hotspot],
  ];
  metrics.replaceChildren();
  for (const [label, value] of rows) {
    const term = document.createElement("dt");
    term.textContent = label;
    const detail = document.createElement("dd");
    detail.textContent = value;
    metrics.append(term, detail);
  }
  walkingLine.textContent = report.walking_text;
  fieldLine.textContent = report.field_note;
  assumptions.replaceChildren();
  for (const line of report.assumptions) {
    const item = document.createElement("li");
    item.textContent = line;
    assumptions.append(item);
  }
  rendererLine.dataset.renderer = renderer;
  rendererLine.dataset.gpuMatch = gpuMatch;
  rendererLine.textContent = renderer === "webgpu" ? "Projection renderer: WebGPU" : "Projection renderer: CPU";
  paintPair(report, ap, lateral, report.modality !== "radiograph" && Boolean(report.weakness));
  setStatus("Analysis finished.");
}

function showFailure(message: string): void {
  results.hidden = true;
  setStatus(message);
  paintProjection(apCanvas, null, false, "No AP view yet.", true);
  paintProjection(lateralCanvas, null, false, "No lateral view yet.", true);
}

async function serverAvailable(): Promise<boolean> {
  if (useBrowserSolver) {
    return false;
  }
  try {
    const response = await fetch("/api/health", { signal: AbortSignal.timeout(800) });
    if (!response.ok) {
      return false;
    }
    const payload = (await response.json()) as { analysis?: string };
    return payload.analysis === "voxel_hexahedral_linear_elastic";
  } catch {
    return false;
  }
}

async function analyzeOnServer(files: File[]): Promise<StrengthResponse> {
  const body = new FormData();
  for (const file of files) {
    body.append("files", file, file.name);
  }
  body.append("body_mass_kg", massInput.value);
  body.append("use_dicom_weight", weightToggle.checked ? "1" : "0");
  const response = await fetch("/api/analyze", { method: "POST", body });
  const payload = (await response.json()) as StrengthResponse;
  if (!response.ok || payload.error) {
    throw new Error(payload.error || "The analysis did not finish.");
  }
  return payload;
}

async function analyze(files: File[]): Promise<void> {
  runButton.disabled = true;
  try {
    let payload: StrengthResponse;
    if (await serverAvailable()) {
      engineNote.textContent = "The solver runs on this computer.";
      setStatus("Solving the vertical load case…");
      payload = await analyzeOnServer(files);
    } else {
      useBrowserSolver = true;
      engineNote.textContent = BROWSER_SOLVER_NOTE;
      const browserPayload = await analyzeInBrowser(
        files,
        Number(massInput.value),
        weightToggle.checked,
        setStatus,
      );
      if (browserPayload.error) {
        throw new Error(browserPayload.error);
      }
      payload = browserPayload as StrengthResponse;
    }
    const projected = await projectionsFor(payload);
    showReport(payload, projected.ap, projected.lateral, projected.renderer, projected.gpuMatch);
  } catch (error) {
    const message = error instanceof Error ? error.message : "The analysis request failed.";
    showFailure(message);
  } finally {
    runButton.disabled = false;
  }
}

function selectedFiles(): File[] {
  return fileInput.files ? Array.from(fileInput.files) : [];
}

function describeFiles(files: File[]): void {
  if (files.length === 0) {
    dropLabel.textContent = "Drop a DICOM series, or a zip of one.";
    return;
  }
  dropLabel.textContent = files.length === 1 ? files[0].name : `${files.length} files selected`;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const files = selectedFiles();
  if (files.length === 0) {
    showFailure("Choose a DICOM series or zip.");
    return;
  }
  void analyze(files);
});

fileInput.addEventListener("change", () => {
  describeFiles(selectedFiles());
});

drop.addEventListener("dragover", (event) => {
  event.preventDefault();
});

drop.addEventListener("drop", (event) => {
  event.preventDefault();
  const dropped = event.dataTransfer?.files;
  if (!dropped || dropped.length === 0) {
    return;
  }
  fileInput.files = dropped;
  describeFiles(Array.from(dropped));
});

engineNote.textContent = useBrowserSolver
  ? BROWSER_SOLVER_NOTE
  : "A local server runs the solver when it is available. Otherwise the solver runs in this browser.";
paintProjection(apCanvas, null, false, "No AP view yet.", true);
paintProjection(lateralCanvas, null, false, "No lateral view yet.", true);
