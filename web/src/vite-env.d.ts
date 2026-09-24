/// <reference types="vite/client" />

declare module "jpeg-lossless-decoder-js" {
  export class Decoder {
    decode(buffer: ArrayBuffer, offset: number, length: number, numBytes: number): Uint8Array | Uint16Array | Int16Array;
  }
}

declare module "@cornerstonejs/codec-openjpeg/decode" {
  interface J2KDecoder {
    getEncodedBuffer(length: number): Uint8Array;
    decode(): void;
    getDecodedBuffer(): Uint8Array;
  }
  interface OpenJpegModule {
    J2KDecoder: new () => J2KDecoder;
  }
  const factory: (options?: {
    print?: (text: string) => void;
    printErr?: (text: string) => void;
  }) => Promise<OpenJpegModule>;
  export default factory;
}
