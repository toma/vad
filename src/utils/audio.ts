export type TypedArray = Uint8Array | Int16Array | Float32Array;

/** Reinterpret a Node Buffer as a typed array of the given bit depth. */
export function toTypedArray(
  buffer: Buffer,
  bits: 8,
  endianness?: "LE" | "BE",
): Uint8Array;
export function toTypedArray(
  buffer: Buffer,
  bits: 16,
  endianness?: "LE" | "BE",
): Int16Array;
export function toTypedArray(
  buffer: Buffer,
  bits: 32,
  endianness?: "LE" | "BE",
): Float32Array;
export function toTypedArray(
  buffer: Buffer,
  bits: 8 | 16 | 32 = 8,
  endianness: "LE" | "BE" = "LE",
): TypedArray {
  const byteWidth = bits / 8;
  const byteOffset = buffer.byteOffset + (buffer.byteOffset % byteWidth);
  const byteLength = buffer.byteLength - (buffer.byteLength % byteWidth);

  let typedArray: TypedArray;

  if (bits === 8) {
    typedArray = new Uint8Array(
      buffer.buffer,
      byteOffset,
      byteLength / byteWidth,
    );
  } else if (bits === 16) {
    typedArray = new Int16Array(
      buffer.buffer,
      byteOffset,
      byteLength / byteWidth,
    );

    if (endianness === "BE") {
      swap16(typedArray as Int16Array);
    }
  } else if (bits === 32) {
    typedArray = new Float32Array(
      buffer.buffer,
      byteOffset,
      byteLength / byteWidth,
    );

    if (endianness === "BE") {
      swap32(typedArray as Float32Array);
    }
  } else {
    throw new Error("Unsupported bit depth");
  }

  return typedArray;
}

function swap16(array: Int16Array): void {
  for (const [i, val] of array.entries()) {
    array[i] = ((val & 0xff) << 8) | ((val >> 8) & 0xff);
  }
}

function swap32(array: Float32Array): void {
  const uint32Array = new Uint32Array(
    array.buffer,
    array.byteOffset,
    array.length,
  );

  for (const [i, val] of uint32Array.entries()) {
    uint32Array[i] =
      ((val & 0xff) << 24) |
      ((val & 0xff00) << 8) |
      ((val >> 8) & 0xff00) |
      ((val >> 24) & 0xff);
  }
}

/** mu-law (G.711) codec. Encodes/decodes between 16-bit PCM and 8-bit mu-law. */
export class Mulaw {
  private static readonly SIGN_BIT = 0x80;
  private static readonly QUANT_MASK = 0xf;
  private static readonly SEG_SHIFT = 4;
  private static readonly SEG_MASK = 0x70;
  private static readonly BIAS = 0x84;

  /** Encode 16-bit PCM to 8-bit mu-law. Does NOT resample. */
  public static encode(samples: Int16Array, dither = false): Uint8Array {
    const encoded = new Uint8Array(samples.length);
    let i = 0;
    for (const sample of samples) {
      encoded[i++] = this.linearToMulaw(sample, dither);
    }
    return encoded;
  }

  /** Decode 8-bit mu-law to 16-bit PCM. Does NOT resample. */
  public static decode(samples: Uint8Array): Int16Array {
    const decoded = new Int16Array(samples.length);
    let i = 0;
    for (const byte of samples) {
      decoded[i++] = this.mulawToLinear(byte);
    }
    return decoded;
  }

  private static linearToMulaw(pcmVal: number, dither: boolean): number {
    let mask: number;

    if (pcmVal < 0) {
      pcmVal = this.BIAS - pcmVal;
      mask = 0x7f;
    } else {
      pcmVal += this.BIAS;
      mask = 0xff;
    }

    if (pcmVal > 0x7fff) {
      pcmVal = 0x7fff;
    }

    const seg = this.valSeg(pcmVal);

    if (dither) {
      const stepSize = 1 << (seg + 3);
      pcmVal += (Math.random() + Math.random() - 1) * stepSize;
      pcmVal = Math.max(this.BIAS, Math.min(0x7fff, Math.round(pcmVal)));
      const newSeg = this.valSeg(pcmVal);
      return (((newSeg << 4) | ((pcmVal >> (newSeg + 3)) & 0xf)) ^ mask) & 0xff;
    }

    return (((seg << 4) | ((pcmVal >> (seg + 3)) & 0xf)) ^ mask) & 0xff;
  }

  private static mulawToLinear(uVal: number): number {
    uVal = ~uVal & 0xff;

    let t = ((uVal & this.QUANT_MASK) << 3) + this.BIAS;
    t <<= (uVal & this.SEG_MASK) >> this.SEG_SHIFT;

    return uVal & this.SIGN_BIT ? this.BIAS - t : t - this.BIAS;
  }

  private static valSeg(val: number): number {
    let r = 0;

    val >>= 7;
    if ((val & 0xf0) !== 0) {
      val >>= 4;
      r += 4;
    }
    if ((val & 0x0c) !== 0) {
      val >>= 2;
      r += 2;
    }
    if ((val & 0x02) !== 0) {
      r += 1;
    }
    return r;
  }
}

/**
 * Upsample an Int16Array from inputSampleRate to outputSampleRate via linear interpolation.
 */
export function upsamplePCM(
  samples: Int16Array,
  inputSampleRate: number,
  outputSampleRate: number,
): Int16Array {
  if (outputSampleRate <= inputSampleRate) {
    throw new Error(
      "Output sample rate must be higher than input sample rate for upsampling.",
    );
  }

  const ratio = outputSampleRate / inputSampleRate;
  const newLength = Math.floor(samples.length * ratio);
  const upsampled = new Int16Array(newLength);

  for (let i = 0; i < newLength; i++) {
    const srcIndex = i / ratio;
    const indexLower = Math.floor(srcIndex);
    const indexUpper = Math.ceil(srcIndex);

    if (indexUpper >= samples.length) {
      upsampled[i] = samples[samples.length - 1];
      continue;
    }

    const sampleLower = samples[indexLower];
    const sampleUpper = samples[indexUpper];
    const weight = srcIndex - indexLower;

    upsampled[i] = Math.round(sampleLower * (1 - weight) + sampleUpper * weight);
  }

  return upsampled;
}
