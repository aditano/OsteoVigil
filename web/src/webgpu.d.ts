interface Navigator {
  gpu?: GPU;
}

interface GPU {
  requestAdapter(): Promise<GPUAdapter | null>;
}

interface GPUAdapter {
  requestDevice(): Promise<GPUDevice>;
}

interface GPUDevice {
  createShaderModule(descriptor: { code: string }): GPUShaderModule;
  createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout;
  createPipelineLayout(descriptor: { bindGroupLayouts: GPUBindGroupLayout[] }): GPUPipelineLayout;
  createComputePipeline(descriptor: {
    layout: GPUPipelineLayout;
    compute: { module: GPUShaderModule; entryPoint: string };
  }): GPUComputePipeline;
  createBuffer(descriptor: { size: number; usage: number }): GPUBuffer;
  createBindGroup(descriptor: { layout: GPUBindGroupLayout; entries: GPUBindGroupEntry[] }): GPUBindGroup;
  createCommandEncoder(): GPUCommandEncoder;
  queue: GPUQueue;
}

interface GPUShaderModule {}

interface GPUBindGroupLayout {}

interface GPUPipelineLayout {}

interface GPUComputePipeline {}

interface GPUBindGroup {}

interface GPUCommandBuffer {}

interface GPUBuffer {
  mapAsync(mode: number): Promise<void>;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
}

interface GPUQueue {
  writeBuffer(buffer: GPUBuffer, offset: number, data: BufferSource): void;
  submit(commands: GPUCommandBuffer[]): void;
}

interface GPUCommandEncoder {
  beginComputePass(): GPUComputePassEncoder;
  copyBufferToBuffer(
    source: GPUBuffer,
    sourceOffset: number,
    destination: GPUBuffer,
    destinationOffset: number,
    size: number,
  ): void;
  finish(): GPUCommandBuffer;
}

interface GPUComputePassEncoder {
  setPipeline(pipeline: GPUComputePipeline): void;
  setBindGroup(index: number, bindGroup: GPUBindGroup): void;
  dispatchWorkgroups(count: number): void;
  end(): void;
}

interface GPUBindGroupLayoutDescriptor {
  entries: Array<{
    binding: number;
    visibility: number;
    buffer: { type: "read-only-storage" | "storage" | "uniform" };
  }>;
}

interface GPUBindGroupEntry {
  binding: number;
  resource: { buffer: GPUBuffer };
}

declare const GPUShaderStage: { COMPUTE: number };
declare const GPUBufferUsage: {
  STORAGE: number;
  COPY_DST: number;
  COPY_SRC: number;
  UNIFORM: number;
  MAP_READ: number;
};
declare const GPUMapMode: { READ: number };
