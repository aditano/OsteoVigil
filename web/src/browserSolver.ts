import { decodeDicomFrame, prepareDicomDecoders } from "./dicomDecode";

type StrengthPayload = {
  error?: string;
  modality?: string;
};

type PyodideFs = {
  mkdir: (path: string) => void;
  writeFile: (path: string, data: string | Uint8Array) => void;
  readdir: (path: string) => string[];
  unlink: (path: string) => void;
  rmdir: (path: string) => void;
};

type PyodideApi = {
  FS: PyodideFs;
  loadPackage: (names: string | string[]) => Promise<void>;
  runPythonAsync: (code: string) => Promise<unknown>;
  globals: { set: (name: string, value: unknown) => void };
};

type LoadPyodide = (options: { indexURL: string }) => Promise<PyodideApi>;

const PYODIDE_INDEX = "https://cdn.jsdelivr.net/pyodide/v314.0.6/full/";
const PYODIDE_SCRIPT = `${PYODIDE_INDEX}pyodide.js`;

export const PYODIDE_BASE_PACKAGES = ["numpy", "scipy", "micropip"] as const;
export const DICOM_CODEC_PYODIDE_PACKAGES = ["pillow"] as const;
export const DICOM_CODEC_REQUIRED_PACKAGES = ["pydicom==2.4.4", "pylibjpeg==2.1.0"] as const;
// Native pylibjpeg plugins have no wasm wheels. The imagecodecs wasm wheel
// aborts Pyodide when a codec is called, so JPEG 2000 and lossless JPEG are
// decoded in the browser and passed into pydicom. Pillow covers baseline JPEG.
export const DICOM_CODEC_OPTIONAL_PACKAGES = [
  "pylibjpeg-libjpeg==2.4.0",
  "pylibjpeg-openjpeg==2.6.0",
] as const;
export const DICOM_CODEC_STATUS = "Loading DICOM image codecs…";
export const COMPRESSED_PIXEL_ERROR =
  "This DICOM is compressed and the image codecs could not decode it.";

let runtime: Promise<PyodideApi> | null = null;

function pyodideLoader(): LoadPyodide {
  const candidate = (globalThis as { loadPyodide?: LoadPyodide }).loadPyodide;
  if (!candidate) {
    throw new Error("The browser solver did not start.");
  }
  return candidate;
}

function loadScript(url: string): Promise<void> {
  if (document.querySelector(`script[src="${url}"]`)) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = url;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("The browser solver could not be downloaded."));
    document.head.append(script);
  });
}

function mkdirTree(fs: PyodideFs, path: string): void {
  const parts = path.split("/").filter((part) => part.length > 0);
  let current = "";
  for (const part of parts) {
    current += `/${part}`;
    try {
      fs.mkdir(current);
    } catch {
      // The directory already exists.
    }
  }
}

function resetDirectory(fs: PyodideFs, path: string): void {
  mkdirTree(fs, path);
  for (const name of fs.readdir(path)) {
    if (name === "." || name === "..") {
      continue;
    }
    const child = `${path}/${name}`;
    try {
      fs.unlink(child);
    } catch {
      resetDirectory(fs, child);
      fs.rmdir(child);
    }
  }
}

async function writeEngine(pyodide: PyodideApi): Promise<void> {
  const base = import.meta.env.BASE_URL;
  const manifestResponse = await fetch(`${base}engine/manifest.json`);
  if (!manifestResponse.ok) {
    throw new Error("The strength solver files are missing from this site.");
  }
  const files = (await manifestResponse.json()) as string[];
  mkdirTree(pyodide.FS, "/pkg/cpt_predictor");
  for (const relative of files) {
    const response = await fetch(`${base}engine/${relative}`);
    if (!response.ok) {
      throw new Error(`The strength solver is missing ${relative}.`);
    }
    const text = await response.text();
    const slash = relative.lastIndexOf("/");
    if (slash > 0) {
      mkdirTree(pyodide.FS, `/pkg/${relative.slice(0, slash)}`);
    }
    pyodide.FS.writeFile(`/pkg/${relative}`, text);
  }
}

function pythonPackageList(packages: readonly string[]): string {
  return JSON.stringify(packages);
}

async function installMicropipPackages(
  pyodide: PyodideApi,
  packages: readonly string[],
  optional: boolean,
): Promise<void> {
  const names = pythonPackageList(packages);
  const body = optional
    ? `for name in ${names}:
    try:
        await micropip.install(name)
    except Exception:
        pass`
    : `missing = []
for name in ${names}:
    try:
        await micropip.install(name)
    except Exception:
        missing.append(name)
if missing:
    raise RuntimeError(",".join(missing))`;
  await pyodide.runPythonAsync(`import micropip\n${body}`);
}

function userFacingSolverError(message: string): string {
  const codecFailure = /handlers are available to decode|missing required dependencies|could not be read because Pillow|Unable to decode the pixel data|JPEG 2000 plugin|pylibjpeg|imagecodecs/i.test(
    message,
  );
  if (codecFailure) {
    return COMPRESSED_PIXEL_ERROR;
  }
  if (!message.includes("Traceback")) {
    return message;
  }
  const lines = message
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
  const last = lines[lines.length - 1] ?? "The analysis request failed.";
  return last.replace(/^[A-Za-z_][\w.]*:\s*/, "");
}

async function startRuntime(onStatus: (text: string) => void): Promise<PyodideApi> {
  onStatus("Loading the browser solver…");
  await loadScript(PYODIDE_SCRIPT);
  const pyodide = await pyodideLoader()({ indexURL: PYODIDE_INDEX });
  onStatus("Loading NumPy and SciPy…");
  await pyodide.loadPackage([...PYODIDE_BASE_PACKAGES]);
  onStatus(DICOM_CODEC_STATUS);
  try {
    await pyodide.loadPackage([...DICOM_CODEC_PYODIDE_PACKAGES]);
    await installMicropipPackages(pyodide, DICOM_CODEC_REQUIRED_PACKAGES, false);
    await prepareDicomDecoders();
    const codecHost = globalThis as typeof globalThis & {
      osteovigilDecodeDicomFrame?: typeof decodeDicomFrame;
    };
    codecHost.osteovigilDecodeDicomFrame = decodeDicomFrame;
  } catch {
    throw new Error("The DICOM image codecs could not be installed in this browser.");
  }
  await installMicropipPackages(pyodide, DICOM_CODEC_OPTIONAL_PACKAGES, true);
  await writeEngine(pyodide);
  await pyodide.runPythonAsync("import sys\nsys.path.insert(0, '/pkg')\nimport cpt_predictor.browser_api");
  return pyodide;
}

function safeFileName(name: string, index: number): string {
  const base = name.split(/[/\\]/).pop() || `slice_${index}.dcm`;
  const cleaned = base.replace(/[^A-Za-z0-9._-]/g, "_");
  return `${String(index).padStart(4, "0")}_${cleaned}`;
}

export async function analyzeInBrowser(
  files: File[],
  bodyMassKg: number,
  preferDicomWeight: boolean,
  onStatus: (text: string) => void,
): Promise<StrengthPayload> {
  if (!runtime) {
    runtime = startRuntime(onStatus).catch((error: unknown) => {
      runtime = null;
      throw error;
    });
  }
  const pyodide = await runtime;
  onStatus("Solving the vertical load case…");
  const studyDir = "/studies/current";
  resetDirectory(pyodide.FS, studyDir);
  const written: string[] = [];
  for (let index = 0; index < files.length; index += 1) {
    const file = files[index];
    const filename = safeFileName(file.name, index);
    const bytes = new Uint8Array(await file.arrayBuffer());
    pyodide.FS.writeFile(`${studyDir}/${filename}`, bytes);
    written.push(filename);
  }
  const source = written.length === 1 && written[0].toLowerCase().endsWith(".zip")
    ? `${studyDir}/${written[0]}`
    : studyDir;
  pyodide.globals.set("study_dir", source);
  pyodide.globals.set("body_mass_kg", bodyMassKg);
  pyodide.globals.set("prefer_weight", preferDicomWeight);
  let raw: unknown;
  try {
    raw = await pyodide.runPythonAsync(`
import json
from cpt_predictor.browser_api import analyze_upload
json.dumps(analyze_upload(study_dir, float(body_mass_kg), bool(prefer_weight)))
`);
  } catch (error) {
    const message = error instanceof Error ? error.message : "The analysis request failed.";
    throw new Error(userFacingSolverError(message));
  }
  if (typeof raw !== "string") {
    throw new Error("The browser solver did not return a result.");
  }
  const payload = JSON.parse(raw) as StrengthPayload;
  if (payload.error) {
    payload.error = userFacingSolverError(payload.error);
  }
  return payload;
}
