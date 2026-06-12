import { AutoTokenizer, type PreTrainedTokenizer } from "@huggingface/transformers";
import _ from "lodash";
import { Tensor } from "onnxruntime-node";
import { foldContext, type ChatMessage } from "../models/chat.js";
import type {
  VADEndOfTurnInputType,
  VADEndOfTurnOutputType,
} from "../models/vad.js";
import { softmax } from "../utils/math.js";
import { BaseOnnxModel } from "../utils/models/base.js";

const PUNCS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~';
const MAX_MESSAGES = 4;
const EOT_TOKEN = "<|im_end|>";

const EOT_ABORTED_DEFAULT: VADEndOfTurnOutputType = {
  endOfTurnProbability: 0.5,
};

export interface EndOfTurnDetectorParams {
  /** Absolute path to the LiveKit turn-detector ONNX model. */
  modelPath: string;
  /**
   * Absolute path to the tokenizer. Either a directory containing
   * `tokenizer.json` / `tokenizer_config.json` or a HuggingFace repo id.
   */
  tokenizerPath: string;
  /**
   * Maximum number of requests that can be queued behind the inference
   * semaphore before new requests are shed (returning a neutral 0.5).
   */
  queueDepth?: number;
}

type EotIO = { input_ids: Tensor };
type EotOutput = { logits: Tensor };

class EotTokenizer {
  private constructor(private readonly tokenizer: PreTrainedTokenizer) {}

  public static async load(tokenizerPath: string): Promise<EotTokenizer> {
    const tokenizer = await AutoTokenizer.from_pretrained(tokenizerPath);
    return new EotTokenizer(tokenizer);
  }

  public get eotToken(): string {
    return EOT_TOKEN;
  }

  public get eotIndex(): number {
    return this.encode(EOT_TOKEN)[0];
  }

  public encode(text: string): number[] {
    return this.tokenizer.encode(text);
  }

  public decode(tokens: number[]): string {
    return this.tokenizer.decode(tokens);
  }

  public formatContext(context: ChatMessage[]): string {
    const folded = foldContext(context).filter((msg) =>
      ["user", "assistant"].includes(msg.role),
    );

    const lastMessage = _.last(folded);
    if (!lastMessage || lastMessage.role === "assistant") {
      folded.push({
        role: "user",
        content: "",
        timestamp: Date.now(),
      });
    }

    const reduced = folded.slice(-MAX_MESSAGES);

    const normalized = reduced.map((msg) => ({
      role: msg.role,
      content: this.normalize(msg.content),
    }));

    const templateContext = this.tokenizer.apply_chat_template(normalized, {
      add_generation_prompt: true,
      tokenize: false,
    }) as string;

    const lastEotIndex = templateContext.lastIndexOf(EOT_TOKEN);
    return lastEotIndex >= 0
      ? templateContext.slice(0, lastEotIndex)
      : templateContext;
  }

  private normalize(text: string): string {
    const stripped = text
      .split("")
      .filter((char) => !PUNCS.includes(char))
      .join("");
    return stripped.toLowerCase().split(" ").join(" ");
  }
}

/**
 * End-of-turn detector backed by the LiveKit turn-detector ONNX model.
 *
 * Given a short chat transcript, returns the probability that the user has
 * finished their conversational turn.
 */
export class EndOfTurnDetector {
  private readonly model: BaseOnnxModel<EotIO, EotOutput>;
  private readonly tokenizerPath: string;
  private tokenizerPromise: Promise<EotTokenizer> | null = null;

  constructor(params: EndOfTurnDetectorParams) {
    this.model = new BaseOnnxModel<EotIO, EotOutput>(params.modelPath, {
      queueDepth: params.queueDepth,
    });
    this.tokenizerPath = params.tokenizerPath;
  }

  private getTokenizer(): Promise<EotTokenizer> {
    if (!this.tokenizerPromise) {
      this.tokenizerPromise = EotTokenizer.load(this.tokenizerPath);
    }
    return this.tokenizerPromise;
  }

  public async detect(
    input: VADEndOfTurnInputType,
    signal?: AbortSignal,
  ): Promise<VADEndOfTurnOutputType> {
    let inputTensor: Tensor | undefined;
    let outputTensor: Tensor | undefined;
    let acquired = false;

    try {
      const semaphore = this.model.getSessionSemaphore();

      // Shed load instead of queuing unboundedly when the model is saturated.
      if (semaphore.isQueueFull()) {
        return EOT_ABORTED_DEFAULT;
      }

      await semaphore.acquire();
      acquired = true;

      if (signal?.aborted) {
        return EOT_ABORTED_DEFAULT;
      }

      const session = await this.model.getSession();
      const tokenizer = await this.getTokenizer();

      const context = tokenizer.formatContext(input.context);
      const tokens = tokenizer.encode(context);

      if (!tokens || tokens.length === 0) {
        throw new Error("Invalid tokens generated from context");
      }

      inputTensor = new Tensor("int64", tokens, [1, tokens.length]);

      if (signal?.aborted) {
        return EOT_ABORTED_DEFAULT;
      }

      ({ logits: outputTensor } = await session.run({ input_ids: inputTensor }));

      if (!outputTensor || !outputTensor.dims || outputTensor.dims.length < 2) {
        throw new Error("Invalid output tensor shape from model");
      }

      const shape = outputTensor.dims;
      const vocabSize = shape[shape.length - 1];
      const logitsData = outputTensor.data as unknown as number[];

      if (!logitsData || logitsData.length === 0) {
        throw new Error("Invalid logits data from model");
      }

      const lastTokenIndex = shape[1] - 1;
      const lastTokenLogitsStart = lastTokenIndex * vocabSize;

      if (
        lastTokenLogitsStart < 0 ||
        lastTokenLogitsStart + vocabSize > logitsData.length
      ) {
        throw new Error("Invalid logits array bounds");
      }

      const lastTokenLogits = logitsData.slice(
        lastTokenLogitsStart,
        lastTokenLogitsStart + vocabSize,
      );

      const probs = softmax(lastTokenLogits);

      if (tokenizer.eotIndex < 0 || tokenizer.eotIndex >= probs.length) {
        throw new Error("Invalid EOT index");
      }

      return { endOfTurnProbability: probs[tokenizer.eotIndex] };
    } catch (error) {
      console.error("Error in EndOfTurnDetector.detect:", error);
      return EOT_ABORTED_DEFAULT;
    } finally {
      try {
        inputTensor?.dispose();
      } catch (disposeError) {
        console.error("Error disposing input tensor:", disposeError);
      }

      try {
        outputTensor?.dispose();
      } catch (disposeError) {
        console.error("Error disposing output tensor:", disposeError);
      }

      if (acquired) {
        try {
          this.model.getSessionSemaphore().release();
        } catch (releaseError) {
          console.error("Error releasing semaphore:", releaseError);
        }
      }
    }
  }

  public async destroy(): Promise<void> {
    await this.model.releaseSession();
  }
}

export default EndOfTurnDetector;
