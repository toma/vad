import _ from "lodash";
import type { VADOutputType } from "~/models/vad";
import {
  type RealtimeVADOptions,
  type RealtimeVADOptionsInput,
  type RealtimeVADParams,
  defaultRealtimeVADInput,
  defaultRealtimeVADOptions,
} from "~/models/realtimeVadOptions";
import { Mulaw, toTypedArray, upsamplePCM } from "~/utils/audio";
import Silero from "./silero";

const MAX_SAMPLE_PAYLOADS = 50;
const MAX_PROBABILITIES = 1000;
const MIN_FRAME_DURATION_MS = 30;
const SAMPLES_PER_MS_16KHZ = 16;
const EASING_THRESHOLD_PADDING = 0.4;

interface SamplePayload {
  samples: Int16Array;
  originTimestamp: number;
}

interface LatestFrames {
  originTimestamp: number;
  frames: Int16Array[];
}

interface LabelSpeech {
  label: string;
  isSpeaking: boolean;
  originTimestamp: number;
}

export class RealtimeVAD {
  private readonly silero: Silero;
  private readonly options: RealtimeVADOptions;
  private readonly samplePayloads: SamplePayload[] = [];
  private readonly probabilities: number[] = [];
  
  private inputLabelToOptions: Record<string, RealtimeVADOptionsInput> = {};
  private inputLabelToSpeech: Record<string, boolean> = {};
  private destroyed = false;

  constructor(options: RealtimeVADParams = {}) {
    this.silero = new Silero();
    this.options = _.merge({}, defaultRealtimeVADOptions, options);
    this.validateOptions();
    if (options?.verbose) {
      console.log(this.options);
    }

    for (const input of this.options.inputs) {
      this.inputLabelToOptions[input.label] = input;
    }
  }

  private get frameSampleCount(): number {
    return this.options.frameDurationMs * SAMPLES_PER_MS_16KHZ;
  }

  public push(buffer: Buffer, originTimestamp: number): void {
    if (this.destroyed) {
      return;
    }

    let samples: Int16Array;
    if (this.options.format === "mulaw_8000") {
      samples = upsamplePCM(Mulaw.decode(toTypedArray(buffer, 8)), 8000, 16000);
    } else {
      samples = toTypedArray(buffer, 16);
    }

    this.samplePayloads.push({ samples, originTimestamp });
    this.reduceSamples();
  }

  public async run(): Promise<LabelSpeech[]> {
    if (this.destroyed) {
      return [];
    }

    const res = this.getLatestFrames();
    if (!res) {
      return [];
    }

    const { frames, originTimestamp } = res;
    for (const frame of frames) {
      const frameProb = await this.process(frame);
      if (this.destroyed) {
        return [];
      }

      this.probabilities.push(frameProb);
    }
    while (this.probabilities.length > MAX_PROBABILITIES) {
      this.probabilities.shift();
    }

    const results: LabelSpeech[] = [];
    for (const label of Object.keys(this.inputLabelToOptions)) {
      const option = this.inputLabelToOptions[label]!;
      const probabilities = _.cloneDeep(this.probabilities).map((prob) =>
        this.applyQuadraticEasing(label, prob),
      );

      while (probabilities.length > option.lookbackFrames) {
        const probs = probabilities.slice(0, option.lookbackFrames);
        const currentSpeech = this.getSpeechOrSilence(label, probs);
        this.inputLabelToSpeech[label] =
          currentSpeech ?? this.inputLabelToSpeech[label];
        probabilities.shift();
      }

      results.push({
        label,
        isSpeaking: this.inputLabelToSpeech[label] ?? false,
        originTimestamp,
      });
    }

    return results;
  }

  public async onMessage(buffer: Buffer): Promise<VADOutputType | undefined> {
    const originTimestamp = Date.now();

    try {
      this.push(buffer, originTimestamp);
    } catch {
      return;
    }

    const processTimestamp = Date.now();
    const outputs = await this.run();
    const procEnd = Date.now();
    if (_.isEmpty(outputs)) {
      return;
    }

    const latency = procEnd - processTimestamp;
    const procStart =
      _.minBy(outputs, "originTimestamp")?.originTimestamp ?? originTimestamp;

    return {
      start: procStart,
      end: procEnd,
      latency,
      outputs: outputs.map((output) => ({
        label: output.label,
        isSpeaking: output.isSpeaking,
      })),
    };
  }

  public async destroy(): Promise<void> {
    this.destroyed = true;
    this.samplePayloads.length = 0;
    this.probabilities.length = 0;
    this.inputLabelToOptions = {};
    this.inputLabelToSpeech = {};
    await this.silero?.destroy();
  }

  /** Clamp options into valid ranges, logging any corrections. */
  private validateOptions(): void {
    if (this.options.inputs.length === 0) {
      console.error("No inputs provided, adding default input");
      this.options.inputs.push(defaultRealtimeVADInput);
    }

    if (this.options.frameDurationMs < MIN_FRAME_DURATION_MS) {
      console.error(
        `frameDurationMs should be at least ${MIN_FRAME_DURATION_MS}, changing to ${MIN_FRAME_DURATION_MS}`,
      );
      this.options.frameDurationMs = MIN_FRAME_DURATION_MS;
    }

    for (const input of this.options.inputs) {
      if (
        input.positiveSpeechThreshold < 0 ||
        input.positiveSpeechThreshold > 1
      ) {
        console.error(
          "positiveSpeechThreshold should be a number between 0 and 1",
        );
      }
      if (
        input.negativeSpeechThreshold < 0 ||
        input.negativeSpeechThreshold > input.positiveSpeechThreshold
      ) {
        console.error(
          "negativeSpeechThreshold should be between 0 and positiveSpeechThreshold",
        );
      }
      if (input.lookbackFrames < 0) {
        input.lookbackFrames = Math.abs(input.lookbackFrames);
        console.error(
          `lookbackFrames should be non-negative, changing to ${input.lookbackFrames}`,
        );
      }
    }
  }

  /** Smooth a raw probability toward 0/1 within a label's threshold band. */
  private applyQuadraticEasing(label: string, prob: number): number {
    const option = this.inputLabelToOptions[label]!;

    const min = Math.max(0, option.negativeSpeechThreshold - EASING_THRESHOLD_PADDING);
    const max = Math.min(1, option.positiveSpeechThreshold + EASING_THRESHOLD_PADDING);

    const normalizedN = (prob - min) / (max - min);
    const clampedN = Math.max(0, Math.min(1, normalizedN));
    return 1 - Math.pow(1 - clampedN, 2);
  }

  private async process(frame: Int16Array): Promise<number> {
    if (this.destroyed) {
      return 0;
    }

    return await this.silero.process(frame);
  }

  /** Drain buffered samples into as many full-size frames as are available. */
  private getLatestFrames(): LatestFrames | null {
    this.reduceSamples();
    if (this.samplePayloads.length === 0) {
      return null;
    }

    const originTimestamp = this.samplePayloads[0].originTimestamp;
    const frames: Int16Array[] = [];

    let totalSamples = _.sumBy(
      this.samplePayloads,
      (payload) => payload.samples.length,
    );

    let sampleIndex = 0;
    let sampleOffset = 0;

    while (totalSamples >= this.frameSampleCount) {
      const frameSamples = new Int16Array(this.frameSampleCount);
      let frameFilled = 0;

      while (frameFilled < this.frameSampleCount) {
        const currentPayload = this.samplePayloads[sampleIndex];
        const remainingSamplesInPayload =
          currentPayload.samples.length - sampleOffset;
        const samplesToCopy = Math.min(
          this.frameSampleCount - frameFilled,
          remainingSamplesInPayload,
        );

        frameSamples.set(
          currentPayload.samples.subarray(
            sampleOffset,
            sampleOffset + samplesToCopy,
          ),
          frameFilled,
        );

        frameFilled += samplesToCopy;
        sampleOffset += samplesToCopy;
        totalSamples -= samplesToCopy;

        if (sampleOffset >= currentPayload.samples.length) {
          sampleIndex += 1;
          sampleOffset = 0;
        }
      }

      frames.push(frameSamples);
    }

    // Drop fully-consumed payloads, then trim the partially-consumed head.
    if (sampleIndex > 0) {
      this.samplePayloads.splice(0, sampleIndex);
    }
    if (this.samplePayloads.length > 0 && sampleOffset > 0) {
      this.samplePayloads[0].samples =
        this.samplePayloads[0].samples.subarray(sampleOffset);
    }

    if (frames.length === 0) {
      return null;
    }

    return { originTimestamp, frames };
  }

  /**
   * Decide speech vs silence over a window of eased probabilities.
   * Returns null when the window is dominated by ambiguous frames.
   */
  private getSpeechOrSilence(
    label: string,
    probabilities: number[],
  ): boolean | null {
    const option = this.inputLabelToOptions[label]!;

    const numSpeechFrames = probabilities.filter(
      (prob) => prob >= option.positiveSpeechThreshold,
    ).length;
    const numSilenceFrames = probabilities.filter(
      (prob) => prob <= option.negativeSpeechThreshold,
    ).length;
    const numUnknownFrames =
      probabilities.length - numSpeechFrames - numSilenceFrames;

    return numUnknownFrames > numSpeechFrames &&
      numUnknownFrames > numSilenceFrames
      ? null
      : numSpeechFrames > numSilenceFrames;
  }

  private reduceSamples(): void {
    const excessLength = this.samplePayloads.length - MAX_SAMPLE_PAYLOADS;
    if (excessLength > 0) {
      _.pullAt(this.samplePayloads, _.range(0, excessLength));
    }
  }
}
