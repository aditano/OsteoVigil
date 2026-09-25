export type GraphicsProbe = {
  webgl: boolean;
  webgpu: boolean;
  offscreenWebgl: boolean;
};

export type SolverPlace = "pyodide-main" | "local-python";
export type DisplayPlace = "canvas-2d" | "webgl";

function isWebglContext(
  context: RenderingContext | OffscreenRenderingContext | null,
): context is WebGLRenderingContext | WebGL2RenderingContext {
  return context !== null && "drawArrays" in context;
}

function releaseWebgl(context: WebGLRenderingContext | WebGL2RenderingContext): void {
  const lose = context.getExtension("WEBGL_lose_context");
  if (lose) {
    lose.loseContext();
  }
}

function canvasHasWebgl(canvas: HTMLCanvasElement | OffscreenCanvas): boolean {
  try {
    const context = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
    if (!isWebglContext(context)) {
      return false;
    }
    releaseWebgl(context);
    return true;
  } catch {
    return false;
  }
}

function withTimeout<T>(promise: Promise<T>, milliseconds: number, fallback: T): Promise<T> {
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(fallback), milliseconds);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      () => {
        clearTimeout(timer);
        resolve(fallback);
      },
    );
  });
}

export async function probeGraphics(): Promise<GraphicsProbe> {
  const offscreenWebgl = typeof OffscreenCanvas === "function" && canvasHasWebgl(new OffscreenCanvas(8, 8));
  const webgl = offscreenWebgl || canvasHasWebgl(document.createElement("canvas"));
  let webgpu = false;
  if (navigator.gpu) {
    webgpu = await withTimeout(
      navigator.gpu.requestAdapter().then((adapter) => adapter !== null),
      1500,
      false,
    );
  }
  return { webgl, webgpu, offscreenWebgl };
}

function computeLabel(solver: SolverPlace, thread: "main" | "worker"): string {
  switch (solver) {
    case "pyodide-main":
      return `CPU (Pyodide/WASM, ${thread} thread)`;
    case "local-python":
      return "CPU (local Python)";
    default: {
      const neverSolver: never = solver;
      return neverSolver;
    }
  }
}

function displayLabel(display: DisplayPlace): string {
  switch (display) {
    case "webgl":
      return "GPU (WebGL)";
    case "canvas-2d":
      return "CPU (2D canvas)";
    default: {
      const neverDisplay: never = display;
      return neverDisplay;
    }
  }
}

export function formatComputeBadge(
  probe: GraphicsProbe,
  solver: SolverPlace,
  display: DisplayPlace,
  thread: "main" | "worker",
): string {
  if (!probe.webgl && !probe.webgpu) {
    return "GPU unavailable · CPU only";
  }
  return `Compute: ${computeLabel(solver, thread)} · Display: ${displayLabel(display)}`;
}

export function viewCanvasIs2d(canvas: HTMLCanvasElement): boolean {
  return canvas.getContext("2d") !== null;
}
