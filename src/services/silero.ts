import { Tensor } from "onnxruntime-node";
import { SileroV5Model } from "~/utils/models/silero";

const BATCH_SIZE = 1;

/** Divisor to normalize int16 PCM samples into the [-1, 1] float range. */
const INT16_NORM = 32768;

class Silero {
  private context: Int16Array | null;
  private state: Tensor | null;

  constructor() {
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
      // We have to acquire the semaphore here in case the session
      // is currently calculating on the state tensor
      await SileroV5Model.getSessionSemaphore().acquire();
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
        SileroV5Model.getSessionSemaphore().release();
      }
    }
  }

  public async process(
    audioFrame: Int16Array,
    sr: number = 16000,
  ): Promise<number> {
    if (!this.state) {
      return 0;
    }

    // Validate input parameters
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

    // Ensure audioFrame is the correct length
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

    // Combine context and audioFrame
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
      // Normalize audio data
      const floatX = new Float32Array(x.length);
      for (let i = 0; i < x.length; i++) {
        floatX[i] = x[i] / INT16_NORM;
      }

      // Get session asynchronously
      const session = await SileroV5Model.getSession();
      if (!this.state || !this.context) {
        return 0;
      }

      // Acquire session semaphore
      await SileroV5Model.getSessionSemaphore().acquire();
      acquired = true;
      if (!this.state || !this.context) {
        return 0;
      }

      // Create input tensor
      inputTensor = new Tensor("float32", floatX, [BATCH_SIZE, x.length]);
      srTensor = new Tensor("int64", [sr], [1]);

      // Run inference
      const sessionResult = await session.run({
        input: inputTensor,
        state: this.state,
        sr: srTensor,
      });

      // Validate session result
      if (!sessionResult || !sessionResult.stateN || !sessionResult.output) {
        throw new Error("Invalid session result");
      }

      // Record the outputs
      stateNTensor = sessionResult.stateN;
      outputTensor = sessionResult.output;

      // Validate output tensor
      if (
        !outputTensor.data ||
        typeof outputTensor.data !== "object" ||
        !("length" in outputTensor.data)
      ) {
        throw new Error("Invalid output tensor data");
      }

      // Save the old state BEFORE updating
      oldState = this.state;

      // Update the state tensor
      this.state = stateNTensor;
      stateNTensor = null; // Clear the reference

      this.context.set(x.subarray(x.length - contextSize));

      const outputData = outputTensor.data as unknown as number[];
      if (outputData.length === 0) {
        throw new Error("Empty output data");
      }

      const [speechProb] = outputData;

      // Validate speech probability
      if (typeof speechProb !== "number" || isNaN(speechProb)) {
        console.warn("Invalid speech probability, returning 0");
        return 0;
      }

      return Math.max(0, Math.min(1, speechProb)); // Clamp between 0 and 1
    } catch (error) {
      console.error("Error processing frame", error);
      return 0;
    } finally {
      // Dispose tensors BEFORE releasing semaphore
      // This ensures no other thread can use the session while we're disposing tensors

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

      // Release semaphore LAST, after all tensor cleanup is complete
      if (acquired) {
        SileroV5Model.getSessionSemaphore().release();
      }
    }
  }
}

export default Silero;
