import { analyzeInBrowser, browserSolverReady, SOLVER_THREAD, type BrowserProgress } from "./browserSolver";
import {
  formatComputeBadge,
  probeGraphics,
  viewCanvasIs2d,
  type DisplayPlace,
  type GraphicsProbe,
  type SolverPlace,
} from "./graphicsProbe";
import { createRunProgress, type RunProgress, type StageId } from "./runProgress";
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

type RadiographViewMeasure = {
  outer_diameter_mm: number;
  inner_diameter_mm: number;
  inner_estimated: boolean;
  canal_resolved: boolean;
  spacing_mm: number[];
};

type RadiographMeasures = {
  cortical_area_mm2: number;
  reference_cortical_area_mm2: number;
  canal_resolved: boolean;
  spacing_source: string;
  spacing_assumed: boolean;
  patient_failure_load_n: number;
  reference_failure_load_n: number;
  views: Record<string, RadiographViewMeasure>;
};

type StrengthResponse = {
  modality: ScanKind;
  method: string;
  body_mass_kg: number;
  failure_load_n: number;
  failure_load_bodyweights: number;
  reference_failure_load_n?: number;
  percent_vs_normal: number;
  comparison: "weaker" | "stronger" | "matched" | "unreliable";
  walking_text: string;
  field_note: string;
  assumptions: string[];
  hotspot_z_mm: number | null;
  solver: string;
  measurement_reliable?: boolean;
  measurement_note?: string;
  radiograph_measures?: RadiographMeasures;
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
const progressPercent = requiredElement<HTMLSpanElement>("progress-percent");
const progressTrack = requiredElement<HTMLDivElement>("progress-track");
const progressFill = requiredElement<HTMLDivElement>("progress-fill");
const computeBadge = requiredElement<HTMLParagraphElement>("compute-badge");
const results = requiredElement<HTMLElement>("results");
const modalityLine = requiredElement<HTMLParagraphElement>("modality");
const percentLine = requiredElement<HTMLParagraphElement>("percent");
const metrics = requiredElement<HTMLDListElement>("metrics");
const walkingLine = requiredElement<HTMLParagraphElement>("walking");
const fieldLine = requiredElement<HTMLParagraphElement>("field");
const assumptions = requiredElement<HTMLUListElement>("assumptions");
const measurementWarning = requiredElement<HTMLParagraphElement>("measurement-warning");
const apCanvas = requiredElement<HTMLCanvasElement>("ap");
const lateralCanvas = requiredElement<HTMLCanvasElement>("lateral");
const apState = requiredElement<HTMLSpanElement>("ap-state");
const lateralState = requiredElement<HTMLSpanElement>("lateral-state");
const rendererLine = requiredElement<HTMLParagraphElement>("renderer");
const engineNote = requiredElement<HTMLParagraphElement>("engine-note");

const BROWSER_SOLVER_NOTE = "The solver runs in this browser. The DICOM stays on this computer.";
let useBrowserSolver = location.hostname.endsWith("github.io");

const COLD_BROWSER_STAGES: readonly StageId[] = [
  "solver",
  "numpy",
  "scipy",
  "codecs",
  "engine",
  "read",
  "decode",
  "solve",
  "render",
];
const WARM_BROWSER_STAGES: readonly StageId[] = ["read", "decode", "solve", "render"];
const SERVER_STAGES: readonly StageId[] = ["upload", "solve", "render"];

function setStatus(text: string): void {
  statusLine.textContent = text;
}

function showProgress(label: string, percent: number): void {
  progressPercent.hidden = false;
  progressTrack.hidden = false;
  setStatus(label);
  progressPercent.textContent = `${percent}%`;
  progressFill.style.width = `${percent}%`;
  progressTrack.setAttribute("aria-valuenow", String(percent));
  progressTrack.setAttribute("aria-valuetext", `${label} ${percent}%`);
}

function resetProgress(): void {
  progressFill.style.transition = "none";
  progressFill.style.width = "0%";
  progressTrack.setAttribute("aria-valuenow", "0");
  progressTrack.setAttribute("aria-valuetext", "");
  progressPercent.textContent = "0%";
  progressFill.getBoundingClientRect();
  progressFill.style.transition = "";
}

function showComputeBadge(solver: SolverPlace, probe: GraphicsProbe): void {
  const display: DisplayPlace = "canvas-2d";
  const canvas2d = viewCanvasIs2d(apCanvas) && viewCanvasIs2d(lateralCanvas);
  computeBadge.hidden = false;
  computeBadge.textContent = formatComputeBadge(probe, solver, display, SOLVER_THREAD);
  computeBadge.dataset.compute = solver === "pyodide-main" ? "pyodide-wasm" : "local-python";
  computeBadge.dataset.display = display;
  computeBadge.dataset.canvas2d = canvas2d ? "true" : "false";
  computeBadge.dataset.gpu = probe.webgl || probe.webgpu ? "available" : "unavailable";
  computeBadge.dataset.solverThread = solver === "pyodide-main" ? "main" : "server";
  computeBadge.dataset.webgl = probe.webgl ? "true" : "false";
  computeBadge.dataset.webgpu = probe.webgpu ? "true" : "false";
  computeBadge.dataset.offscreenWebgl = probe.offscreenWebgl ? "true" : "false";
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
    case "unreliable":
      return report.measurement_note || "Measurement unreliable. Cannot claim stronger than normal from this radiograph.";
    default: {
      const neverWord: never = report.comparison;
      return neverWord;
    }
  }
}

function viewLabel(name: string): string {
  switch (name) {
    case "ap":
      return "AP";
    case "lateral":
      return "Lateral";
    default:
      return name;
  }
}

function diameterSummary(measures: RadiographMeasures, field: "outer_diameter_mm" | "inner_diameter_mm"): string {
  return Object.entries(measures.views)
    .map(([name, view]) => {
      const estimated = field === "inner_diameter_mm" && view.inner_estimated ? " estimated" : "";
      return `${viewLabel(name)} ${view[field].toFixed(1)} mm${estimated}`;
    })
    .join(", ");
}

function spacingSummary(measures: RadiographMeasures): string {
  const views = Object.entries(measures.views)
    .map(([name, view]) => {
      const row = view.spacing_mm[0] ?? 0;
      const col = view.spacing_mm[1] ?? row;
      return `${viewLabel(name)} ${row.toFixed(3)} × ${col.toFixed(3)} mm`;
    })
    .join(", ");
  const assumed = measures.spacing_assumed ? "assumed " : "";
  return `${assumed}${measures.spacing_source}: ${views}`;
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

function showMeasurementWarning(report: StrengthResponse): void {
  percentLine.classList.toggle("unreliable", report.comparison === "unreliable");
  percentLine.dataset.comparison = report.comparison;
  if (report.comparison === "unreliable") {
    measurementWarning.hidden = false;
    measurementWarning.textContent =
      "The raw millimetre measures below are for audit. They are not a stronger-than-normal result.";
    return;
  }
  if (report.measurement_note) {
    measurementWarning.hidden = false;
    measurementWarning.textContent = report.measurement_note;
    return;
  }
  measurementWarning.hidden = true;
  measurementWarning.textContent = "";
}

function showReport(report: StrengthResponse, ap: Projection | null, lateral: Projection | null, renderer: string, gpuMatch: string): void {
  results.hidden = false;
  modalityLine.textContent = `${modalityLabel(report.modality)} · ${report.method.replaceAll("_", " ")}`;
  percentLine.textContent = comparisonSentence(report);
  showMeasurementWarning(report);
  const hotspot = report.hotspot_z_mm === null ? "none localized" : `${report.hotspot_z_mm.toFixed(1)} mm from the distal end`;
  const untrusted = report.comparison === "unreliable" ? " (untrusted)" : "";
  const rows: Array<[string, string]> = [
    ["Failure load", `${Math.round(report.failure_load_n).toLocaleString()} N${untrusted}`],
    ["Body weights", `${report.failure_load_bodyweights.toFixed(2)}×${untrusted}`],
    ["Body mass", `${report.body_mass_kg.toFixed(1)} kg`],
    ["Hotspot", hotspot],
  ];
  const measures = report.radiograph_measures;
  if (measures) {
    rows.push(
      ["Midshaft outer diameter", diameterSummary(measures, "outer_diameter_mm")],
      ["Midshaft inner diameter", diameterSummary(measures, "inner_diameter_mm")],
      ["Cortical area", `${Math.round(measures.cortical_area_mm2).toLocaleString()} mm²`],
      ["Spacing", spacingSummary(measures)],
      ["Reference failure load", `${Math.round(measures.reference_failure_load_n).toLocaleString()} N`],
    );
  }
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
}

function showFailure(message: string): void {
  results.hidden = true;
  percentLine.classList.remove("unreliable");
  delete percentLine.dataset.comparison;
  measurementWarning.hidden = true;
  measurementWarning.textContent = "";
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

function analyzeOnServer(files: File[], onUpload: (fraction: number) => void): Promise<StrengthResponse> {
  const body = new FormData();
  for (const file of files) {
    body.append("files", file, file.name);
  }
  body.append("body_mass_kg", massInput.value);
  body.append("use_dicom_weight", weightToggle.checked ? "1" : "0");
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/analyze");
    xhr.responseType = "text";
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && event.total > 0) {
        onUpload(event.loaded / event.total);
      }
    };
    xhr.upload.onload = () => onUpload(1);
    xhr.onerror = () => reject(new Error("The analysis request failed."));
    xhr.onload = () => {
      let payload: StrengthResponse;
      try {
        payload = JSON.parse(xhr.responseText) as StrengthResponse;
      } catch {
        reject(new Error("The analysis request failed."));
        return;
      }
      if (xhr.status < 200 || xhr.status >= 300 || payload.error) {
        reject(new Error(payload.error || "The analysis did not finish."));
        return;
      }
      resolve(payload);
    };
    xhr.send(body);
  });
}

function yieldToBrowser(): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, 0);
  });
}

async function analyze(files: File[]): Promise<void> {
  runButton.disabled = true;
  resetProgress();
  showProgress("Starting analysis…", 0);
  const graphics = await probeGraphics();
  let tracker: RunProgress | null = null;
  try {
    let payload: StrengthResponse;
    if (await serverAvailable()) {
      engineNote.textContent = "The solver runs on this computer.";
      showComputeBadge("local-python", graphics);
      tracker = createRunProgress(SERVER_STAGES, showProgress);
      tracker.advance("upload", "Uploading the study…", 0);
      payload = await analyzeOnServer(files, (fraction) => {
        if (fraction >= 1) {
          tracker?.advance("solve", "Solving the vertical load case…");
          return;
        }
        tracker?.advance("upload", "Uploading the study…", fraction);
      });
    } else {
      useBrowserSolver = true;
      engineNote.textContent = BROWSER_SOLVER_NOTE;
      showComputeBadge("pyodide-main", graphics);
      const stages = browserSolverReady() ? WARM_BROWSER_STAGES : COLD_BROWSER_STAGES;
      tracker = createRunProgress(stages, showProgress);
      const browserPayload = await analyzeInBrowser(
        files,
        Number(massInput.value),
        weightToggle.checked,
        (event: BrowserProgress) => tracker?.advance(event.stage, event.label, event.fraction),
      );
      if (browserPayload.error) {
        throw new Error(browserPayload.error);
      }
      payload = browserPayload as StrengthResponse;
    }
    tracker.advance("render", "Rendering views…");
    await yieldToBrowser();
    const projected = await projectionsFor(payload);
    showReport(payload, projected.ap, projected.lateral, projected.renderer, projected.gpuMatch);
    tracker.finish();
  } catch (error) {
    tracker?.stop();
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
