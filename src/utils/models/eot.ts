import { AutoTokenizer, type PreTrainedTokenizer } from "@huggingface/transformers";
import _ from "lodash";
import type { Tensor } from "onnxruntime-node";
import { foldContext, type ChatMessage } from "~/models/chat";
import env from "~/utils/env";
import { BaseOnnxModel } from "./base";

const PUNCS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~';
const MAX_MESSAGES = 4;
const TURN_DETECTOR_REPO = "livekit/turn-detector";

/** 
 * LiveKit turn-detector ONNX model (Apache-2.0)
 * @see https://huggingface.co/livekit/turn-detector
 */
export const EndOfTurnModel = new BaseOnnxModel<
  {
    input_ids: Tensor;
  },
  {
    logits: Tensor;
  }
>(
  { repo: TURN_DETECTOR_REPO, file: "model_quantized.onnx" },
  { queueDepth: env.EOT_QUEUE_DEPTH },
);

export class EndOfTurnTokenizer {
  private static instance: EndOfTurnTokenizer;
  private tokenizer!: PreTrainedTokenizer;

  private constructor() {}

  public static async getInstance(): Promise<EndOfTurnTokenizer> {
    if (!EndOfTurnTokenizer.instance) {
      EndOfTurnTokenizer.instance = new EndOfTurnTokenizer();
      EndOfTurnTokenizer.instance.tokenizer =
        await AutoTokenizer.from_pretrained(TURN_DETECTOR_REPO);
    }
    return EndOfTurnTokenizer.instance;
  }

  public get eotToken(): string {
    return "<|im_end|>";
  }

  public get eotIndex(): number {
    return this.encode(this.eotToken)[0];
  }

  public encode(text: string): number[] {
    return this.tokenizer.encode(text);
  }

  public decode(tokens: number[]): string {
    return this.tokenizer.decode(tokens);
  }

  public formatContext(context: ChatMessage[]): string {
    // Fold the context and reduce to assistant and user messages
    const folded = foldContext(context).filter((msg) =>
      ["user", "assistant"].includes(msg.role),
    );

    // If the last message is an assistant message, add a new user message
    const lastMessage = _.last(folded);
    if (!lastMessage || lastMessage.role === "assistant") {
      folded.push({
        role: "user",
        content: "",
        timestamp: Date.now(),
      });
    }

    // Reduce to the last MAX_MESSAGES messages
    const reduced = folded.slice(-MAX_MESSAGES);

    // Normalize the message content
    const normalized = reduced.map((msg) => ({
      role: msg.role,
      content: this.normalize(msg.content),
    }));

    // Apply the chat template
    const templateContext = this.tokenizer.apply_chat_template(normalized, {
      add_generation_prompt: true,
      tokenize: false,
    }) as string;

    // Slice to the last EOT token
    const lastEotIndex = templateContext.lastIndexOf(this.eotToken);
    const sliced =
      lastEotIndex >= 0
        ? templateContext.slice(0, lastEotIndex)
        : templateContext;

    return sliced;
  }

  private normalize(text: string): string {
    const stripped = text
      .split("")
      .filter((char) => !PUNCS.includes(char))
      .join("");
    return stripped.toLowerCase().split(" ").join(" ");
  }
}

export async function initializeEndOfTurn(): Promise<void> {
  await Promise.all([
    EndOfTurnModel.getSession(),
    EndOfTurnTokenizer.getInstance(),
  ]);
}
