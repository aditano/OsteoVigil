export type WeaknessVolume = {
  shape: [number, number, number];
  spacing: [number, number, number];
  values: Float32Array;
};

export type Projection = {
  values: Float32Array;
  width: number;
  height: number;
  spacing: [number, number];
};

export function decodeVolume(shape: number[], spacing: number[], b64: string): WeaknessVolume {
  const bytes = Uint8Array.from(atob(b64), (char) => char.charCodeAt(0));
  const values = new Float32Array(bytes.buffer);
  return {
    shape: [shape[0], shape[1], shape[2]],
    spacing: [spacing[0], spacing[1], spacing[2]],
    values,
  };
}

export function projectWeakness(volume: WeaknessVolume): { ap: Projection; lateral: Projection } {
  const [sizeZ, sizeY, sizeX] = volume.shape;
  const ap = new Float32Array(sizeZ * sizeX);
  const lateral = new Float32Array(sizeZ * sizeY);
  ap.fill(Number.NEGATIVE_INFINITY);
  lateral.fill(Number.NEGATIVE_INFINITY);
  for (let z = 0; z < sizeZ; z += 1) {
    for (let y = 0; y < sizeY; y += 1) {
      for (let x = 0; x < sizeX; x += 1) {
        const value = volume.values[(z * sizeY + y) * sizeX + x];
        const apIndex = z * sizeX + x;
        const lateralIndex = z * sizeY + y;
        if (value > ap[apIndex]) {
          ap[apIndex] = value;
        }
        if (value > lateral[lateralIndex]) {
          lateral[lateralIndex] = value;
        }
      }
    }
  }
  replaceInfinite(ap);
  replaceInfinite(lateral);
  return {
    ap: { values: ap, width: sizeX, height: sizeZ, spacing: [volume.spacing[2], volume.spacing[0]] },
    lateral: { values: lateral, width: sizeY, height: sizeZ, spacing: [volume.spacing[1], volume.spacing[0]] },
  };
}

function replaceInfinite(values: Float32Array): void {
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) {
      values[index] = 0;
    }
  }
}

const SHADER = `
struct Params {
  sizeZ: u32,
  sizeY: u32,
  sizeX: u32,
  mode: u32,
}
@group(0) @binding(0) var<storage, read> volume: array<f32>;
@group(0) @binding(1) var<storage, read_write> image: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (params.mode == 0u) {
    let count = params.sizeZ * params.sizeX;
    if (gid.x >= count) { return; }
    let x = gid.x % params.sizeX;
    let z = gid.x / params.sizeX;
    var best = -1e30;
    for (var y: u32 = 0u; y < params.sizeY; y = y + 1u) {
      let value = volume[(z * params.sizeY + y) * params.sizeX + x];
      if (value > best) { best = value; }
    }
    image[gid.x] = best;
  } else {
    let count = params.sizeZ * params.sizeY;
    if (gid.x >= count) { return; }
    let y = gid.x % params.sizeY;
    let z = gid.x / params.sizeY;
    var best = -1e30;
    for (var x: u32 = 0u; x < params.sizeX; x = x + 1u) {
      let value = volume[(z * params.sizeY + y) * params.sizeX + x];
      if (value > best) { best = value; }
    }
    image[gid.x] = best;
  }
}
`;

export async function projectWithWebGPU(volume: WeaknessVolume): Promise<{ ap: Projection; lateral: Projection } | null> {
  const gpu = navigator.gpu;
  if (!gpu) {
    return null;
  }
  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    return null;
  }
  const device = await adapter.requestDevice();
  const module = device.createShaderModule({ code: SHADER });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module, entryPoint: "main" },
  });
  const [sizeZ, sizeY, sizeX] = volume.shape;
  const volumeBuffer = device.createBuffer({
    size: volume.values.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(volumeBuffer, 0, volume.values);
  const ap = await reduce(device, pipeline, bindGroupLayout, volumeBuffer, sizeZ, sizeY, sizeX, 0);
  const lateral = await reduce(device, pipeline, bindGroupLayout, volumeBuffer, sizeZ, sizeY, sizeX, 1);
  replaceInfinite(ap);
  replaceInfinite(lateral);
  return {
    ap: { values: ap, width: sizeX, height: sizeZ, spacing: [volume.spacing[2], volume.spacing[0]] },
    lateral: { values: lateral, width: sizeY, height: sizeZ, spacing: [volume.spacing[1], volume.spacing[0]] },
  };
}

async function reduce(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  layout: GPUBindGroupLayout,
  volumeBuffer: GPUBuffer,
  sizeZ: number,
  sizeY: number,
  sizeX: number,
  mode: number,
): Promise<Float32Array> {
  const count = mode === 0 ? sizeZ * sizeX : sizeZ * sizeY;
  const output = device.createBuffer({
    size: Math.max(16, count * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const params = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(params, 0, new Uint32Array([sizeZ, sizeY, sizeX, mode]));
  const bind = device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: { buffer: volumeBuffer } },
      { binding: 1, resource: { buffer: output } },
      { binding: 2, resource: { buffer: params } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  const readback = device.createBuffer({
    size: Math.max(16, count * 4),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(output, 0, readback, 0, Math.max(16, count * 4));
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const copy = new Float32Array(readback.getMappedRange().slice(0, count * 4));
  readback.unmap();
  return copy;
}

export function projectionsAgree(cpu: Float32Array, gpu: Float32Array): boolean {
  if (cpu.length !== gpu.length || cpu.length === 0) {
    return false;
  }
  let worst = 0;
  for (let index = 0; index < cpu.length; index += 1) {
    worst = Math.max(worst, Math.abs(cpu[index] - gpu[index]));
  }
  return worst < 1e-3;
}

export type PaintMode = "weakness" | "radiograph";

export function paintProjection(
  canvas: HTMLCanvasElement,
  projection: Projection | null,
  acquired: boolean,
  missingText: string,
  distalAtBottom = true,
  mode: PaintMode = "weakness",
): void {
  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#141210";
  context.fillRect(0, 0, width, height);
  if (!acquired || !projection) {
    context.fillStyle = "#d9cbb8";
    context.font = "18px Liberation Sans, sans-serif";
    context.fillText(missingText, 24, height / 2);
    return;
  }
  const image = context.createImageData(projection.width, projection.height);
  let peak = 0.05;
  if (mode === "weakness") {
    for (let index = 0; index < projection.values.length; index += 1) {
      peak = Math.max(peak, projection.values[index]);
    }
  }
  for (let z = 0; z < projection.height; z += 1) {
    const displayRow = distalAtBottom ? projection.height - 1 - z : z;
    for (let x = 0; x < projection.width; x += 1) {
      const value = projection.values[z * projection.width + x];
      const color = pixelColor(value, peak, mode);
      const offset = (displayRow * projection.width + x) * 4;
      image.data[offset] = color[0];
      image.data[offset + 1] = color[1];
      image.data[offset + 2] = color[2];
      image.data[offset + 3] = 255;
    }
  }
  const scale = Math.min(width / projection.width, height / projection.height);
  const drawWidth = Math.max(1, Math.round(projection.width * scale));
  const drawHeight = Math.max(1, Math.round(projection.height * scale));
  const temp = document.createElement("canvas");
  temp.width = projection.width;
  temp.height = projection.height;
  const tempContext = temp.getContext("2d");
  if (!tempContext) {
    return;
  }
  tempContext.putImageData(image, 0, 0);
  context.imageSmoothingEnabled = false;
  context.drawImage(temp, 0, 0, drawWidth, drawHeight);
}

function pixelColor(value: number, peak: number, mode: PaintMode): [number, number, number] {
  switch (mode) {
    case "weakness":
      return weaknessColor(value, peak);
    case "radiograph":
      return radiographColor(value);
    default: {
      const neverMode: never = mode;
      return neverMode;
    }
  }
}

function weaknessColor(value: number, peak: number): [number, number, number] {
  if (value <= 0) {
    return [28, 48, 58];
  }
  const scaled = Math.max(0, Math.min(1, value / peak));
  const red = Math.round(230 + (154 - 230) * scaled);
  const green = Math.round(177 + (32 - 177) * scaled);
  const blue = Math.round(90 + (18 - 90) * scaled);
  return [red, green, blue];
}

function radiographColor(value: number): [number, number, number] {
  if (!(value >= 2)) {
    const gray = Math.round(Math.max(0, Math.min(1, value)) * 255);
    return [gray, gray, gray];
  }
  const shifted = value - 2;
  const step = Math.max(0, Math.min(10, Math.floor(shifted + 1e-3)));
  const gray = Math.max(0, Math.min(1, shifted - step));
  const weakness = (step / 10) * 2 - 1;
  const level = Math.round(gray * 255);
  const tint = boneTint(weakness);
  const mix = 0.58;
  return [
    Math.round(level * (1 - mix) + tint[0] * mix),
    Math.round(level * (1 - mix) + tint[1] * mix),
    Math.round(level * (1 - mix) + tint[2] * mix),
  ];
}

function boneTint(weakness: number): [number, number, number] {
  if (weakness <= 0) {
    const towardStronger = -weakness;
    return [
      Math.round(232 + (96 - 232) * towardStronger),
      Math.round(214 + (168 - 214) * towardStronger),
      Math.round(176 + (142 - 176) * towardStronger),
    ];
  }
  return [
    Math.round(232 + (154 - 232) * weakness),
    Math.round(180 + (32 - 180) * weakness),
    Math.round(90 + (18 - 90) * weakness),
  ];
}

type Modality = "ct" | "mri" | "radiograph" | "unknown";

export function modalityLabel(kind: Modality): string {
  switch (kind) {
    case "ct":
      return "CT";
    case "mri":
      return "MRI";
    case "radiograph":
      return "Radiograph";
    case "unknown":
      return "Unknown";
    default: {
      const neverKind: never = kind;
      return neverKind;
    }
  }
}
