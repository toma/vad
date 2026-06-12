import type { AudioFormatType } from "./audio.js";
import type { ChatMessage } from "./chat.js";

export interface VADInputItem {
  label: string;
  positiveSpeechThreshold?: number;
  negativeSpeechThreshold?: number;
  lookbackFrames?: number;
}

export interface VADInputType {
  format?: AudioFormatType;
  frameDurationMs?: number;
  inputs: VADInputItem[];
}

export interface VADOutputItem {
  label: string;
  isSpeaking: boolean;
}

export interface VADOutputType {
  latency: number;
  start: number;
  end: number;
  outputs: VADOutputItem[];
}

export interface VADEndOfTurnInputType {
  context: ChatMessage[];
}

export interface VADEndOfTurnOutputType {
  endOfTurnProbability: number;
}
