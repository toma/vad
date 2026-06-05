import type { Tensor } from "onnxruntime-node";
import { BaseOnnxModel, type BaseOnnxModelSession } from "./base";

/** Silero VAD v5 ONNX model (MIT). https://huggingface.co/onnx-community/silero-vad */
export const SileroV5Model = new BaseOnnxModel<
  {
    input: Tensor;
    state: Tensor;
    sr: Tensor;
  },
  {
    output: Tensor;
    stateN: Tensor;
  }
>({ repo: "onnx-community/silero-vad", file: "onnx/model.onnx" });

export function initializeSilero(): Promise<BaseOnnxModelSession> {
  return SileroV5Model.getSession();
}
