import type { AudioFormatType } from "./audio.js";

export interface RealtimeVADOptions {
  format: AudioFormatType;
  frameDurationMs: number;
  inputs: RealtimeVADOptionsInput[];
}

export interface RealtimeVADParams {
  /** Absolute path to a Silero VAD v5 ONNX `model.onnx`. */
  modelPath: string;
  format?: AudioFormatType;
  frameDurationMs?: number;
  verbose?: boolean;
  inputs?: Partial<RealtimeVADOptionsInput>[];
}

export interface RealtimeVADOptionsInput {
  label: string;
  positiveSpeechThreshold: number;
  negativeSpeechThreshold: number;
  lookbackFrames: number;
}

export const defaultRealtimeVADInput: RealtimeVADOptionsInput = {
  label: "default",
  positiveSpeechThreshold: 0.5,
  negativeSpeechThreshold: 0.2,
  lookbackFrames: 5,
};

export const defaultRealtimeVADOptions: RealtimeVADOptions = {
  format: "mulaw_8000",
  frameDurationMs: 80,
  inputs: [defaultRealtimeVADInput],
};
