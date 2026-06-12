import { Tensor } from "onnxruntime-node";
import { BaseOnnxModel } from "../utils/models/base.js";

const BATCH_SIZE = 1;

/** Divisor to normalize int16 PCM samples into the [-1, 1] float range. */
const INT16_NORM = 32768;

export interface SileroParams {
  /** Absolute path to a Silero VAD v5 ONNX `model.onnx`. */
  modelPath: string;
}

type SileroIO = {
  input: Tensor;
  state: Tensor;
  sr: Tensor;
};

type SileroOutput = {
  output: Tensor;
  stateN: Tensor;
};

export class Silero {
  private readonly model: BaseOnnxModel<SileroIO, SileroOutput>;
  private context: Int16Array | null;
  private state: Tensor | null;

  constructor(params: SileroParams) {
    this.model = new BaseOnnxModel<SileroIO, SileroOutput>(params.modelPath);
    this.context = null;
    this.state = new Tensor("float32", new Float32Array(2 * BATCH_SIZE * 128), [
      2,
      BATCH_SIZE,
      128,
    ]);
  }

  public async destroy(): Promise<void> {
    let acquired = false;
    try {
      // Acquire the semaphore in case the session is mid-inference on the state tensor.
      await this.model.getSessionSemaphore().acquire();
      acquired = true;

      if (this.state) {
        this.state.dispose();
        this.state = null;
      }
      this.context = null;
    } catch (error) {
      console.error("Error disposing state tensor", error);
    } finally {
      if (acquired) {
        this.model.getSessionSemaphore().release();
      }
    }

    // Release the ONNX session outside the per-instance semaphore so we don't
    // deadlock with anything that might re-acquire it. Skips quietly if the
    // session was never created.
    await this.model.releaseSession();
  }

  public async process(
    audioFrame: Int16Array,
    sr: number = 16000,
  ): Promise<number> {
    if (!this.state) {
      return 0;
    }

    if (!audioFrame || audioFrame.length === 0) {
      console.warn("Empty audio frame provided");
      return 0;
    }

    if (sr !== 16000 && sr !== 8000) {
      console.warn(`Unsupported sample rate: ${sr}, using 16000`);
      sr = 16000;
    }

    const contextSize = sr === 16000 ? 64 : 32;
    const numSamples = sr === 16000 ? 512 : 256;

    if (!this.context || this.context.length === 0) {
      this.context = new Int16Array(contextSize);
    }

    let processedAudioFrame = audioFrame;
    if (audioFrame.length !== numSamples) {
      if (audioFrame.length > numSamples) {
        processedAudioFrame = audioFrame.subarray(0, numSamples);
      } else {
        const paddedFrame = new Int16Array(numSamples);
        paddedFrame.set(audioFrame);
        processedAudioFrame = paddedFrame;
      }
    }

    const x = new Int16Array(this.context.length + processedAudioFrame.length);
    x.set(this.context);
    x.set(processedAudioFrame, this.context.length);

    let inputTensor: Tensor | null = null;
    let outputTensor: Tensor | null = null;
    let stateNTensor: Tensor | null = null;
    let srTensor: Tensor | null = null;
    let oldState: Tensor | null = null;
    let acquired = false;

    try {
      const floatX = new Float32Array(x.length);
      for (let i = 0; i < x.length; i++) {
        floatX[i] = x[i] / INT16_NORM;
      }

      const session = await this.model.getSession();
      if (!this.state || !this.context) {
        return 0;
      }

      await this.model.getSessionSemaphore().acquire();
      acquired = true;
      if (!this.state || !this.context) {
        return 0;
      }

      inputTensor = new Tensor("float32", floatX, [BATCH_SIZE, x.length]);
      srTensor = new Tensor("int64", [sr], [1]);

      const sessionResult = await session.run({
        input: inputTensor,
        state: this.state,
        sr: srTensor,
      });

      if (!sessionResult || !sessionResult.stateN || !sessionResult.output) {
        throw new Error("Invalid session result");
      }

      stateNTensor = sessionResult.stateN;
      outputTensor = sessionResult.output;

      if (
        !outputTensor.data ||
        typeof outputTensor.data !== "object" ||
        !("length" in outputTensor.data)
      ) {
        throw new Error("Invalid output tensor data");
      }

      oldState = this.state;

      this.state = stateNTensor;
      stateNTensor = null;

      this.context.set(x.subarray(x.length - contextSize));

      const outputData = outputTensor.data as unknown as number[];
      if (outputData.length === 0) {
        throw new Error("Empty output data");
      }

      const [speechProb] = outputData;

      if (typeof speechProb !== "number" || isNaN(speechProb)) {
        console.warn("Invalid speech probability, returning 0");
        return 0;
      }

      return Math.max(0, Math.min(1, speechProb));
    } catch (error) {
      console.error("Error processing frame", error);
      return 0;
    } finally {
      // Dispose tensors before releasing the semaphore so no other call can
      // observe a tensor we're about to free.
      try {
        oldState?.dispose();
      } catch (e) {
        console.error("Error disposing old state:", e);
      }

      if (stateNTensor && stateNTensor !== this.state) {
        try {
          stateNTensor.dispose();
        } catch (e) {
          console.error("Error disposing orphaned state tensor:", e);
        }
      }

      try {
        inputTensor?.dispose();
      } catch (e) {
        console.error("Error disposing input tensor:", e);
      }

      try {
        srTensor?.dispose();
      } catch (e) {
        console.error("Error disposing sr tensor:", e);
      }

      try {
        outputTensor?.dispose();
      } catch (e) {
        console.error("Error disposing output tensor:", e);
      }

      if (acquired) {
        this.model.getSessionSemaphore().release();
      }
    }
  }
}

export default Silero;
