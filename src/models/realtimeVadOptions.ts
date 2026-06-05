import type { AudioFormatType } from "~/models/audio";

export interface RealtimeVADOptions {
  format: AudioFormatType;
  frameDurationMs: number;
  inputs: RealtimeVADOptionsInput[];
}

export interface RealtimeVADParams {
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
