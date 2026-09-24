import { Decoder } from "jpeg-lossless-decoder-js";

const JPEG_LOSSLESS_P14 = "1.2.840.10008.1.2.4.57";
const JPEG_LOSSLESS_SV1 = "1.2.840.10008.1.2.4.70";
const JPEG2000_LOSSLESS = "1.2.840.10008.1.2.4.90";
const JPEG2000 = "1.2.840.10008.1.2.4.91";

type CodecKind = "jpeg-lossless" | "jpeg2000";

interface J2KDecoder {
  getEncodedBuffer(length: number): Uint8Array;
  decode(): void;
  getDecodedBuffer(): Uint8Array;
  getFrameInfo(): {
    width: number;
    height: number;
    bitsPerSample: number;
    componentCount: number;
    isSigned: boolean;
  };
}

interface OpenJpegModule {
  J2KDecoder: new () => J2KDecoder;
}

type OpenJpegFactory = (options?: {
  print?: (text: string) => void;
  printErr?: (text: string) => void;
}) => Promise<OpenJpegModule>;

let openjpeg: OpenJpegModule | null = null;

function codecKind(transferSyntaxUid: string): CodecKind | null {
  switch (transferSyntaxUid) {
    case JPEG_LOSSLESS_P14:
    case JPEG_LOSSLESS_SV1:
      return "jpeg-lossless";
    case JPEG2000_LOSSLESS:
    case JPEG2000:
      return "jpeg2000";
    default:
      return null;
  }
}

function copiedBytes(view: ArrayBufferView): Uint8Array {
  return new Uint8Array(view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength));
}

function decodeLossless(frame: Uint8Array, bitsAllocated: number): Uint8Array {
  const copy = frame.slice();
  const bytesPerSample = bitsAllocated <= 8 ? 1 : 2;
  const decoded = new Decoder().decode(copy.buffer, 0, copy.byteLength, bytesPerSample);
  return copiedBytes(decoded);
}

function decodeJpeg2000(frame: Uint8Array): Uint8Array {
  if (!openjpeg) {
    throw new Error("The JPEG 2000 codec is not ready.");
  }
  const decoder = new openjpeg.J2KDecoder();
  const encoded = decoder.getEncodedBuffer(frame.length);
  encoded.set(frame);
  decoder.decode();
  return copiedBytes(decoder.getDecodedBuffer());
}

export async function prepareDicomDecoders(): Promise<void> {
  if (openjpeg) {
    return;
  }
  const imported = (await import("@cornerstonejs/codec-openjpeg/decode")) as { default?: OpenJpegFactory };
  const factory = imported.default;
  if (!factory) {
    throw new Error("The JPEG 2000 codec did not load.");
  }
  openjpeg = await factory({
    print: () => undefined,
    printErr: () => undefined,
  });
}

export function decodeDicomFrame(
  frame: Uint8Array,
  _rows: number,
  _columns: number,
  bitsAllocated: number,
  _samplesPerPixel: number,
  _signed: boolean,
  transferSyntaxUid: string,
): Uint8Array {
  const kind = codecKind(transferSyntaxUid);
  if (kind === "jpeg-lossless") {
    return decodeLossless(frame, bitsAllocated);
  }
  if (kind === "jpeg2000") {
    return decodeJpeg2000(frame);
  }
  throw new Error(`No browser codec for transfer syntax ${transferSyntaxUid}.`);
}
