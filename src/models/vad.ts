import { z } from "zod";
import { AudioFormatSchema } from "./audio";
import { ChatMessageSchema } from "./chat";

const StringOrNumberSchema = z
  .union([z.string(), z.number()])
  .transform((v) => {
    if (typeof v === "string") {
      return Number(v);
    }
    return v;
  })
  .optional();

export const VADInputSchema = z.object({
  format: AudioFormatSchema.optional(),
  frameDurationMs: StringOrNumberSchema,
  inputs: z.array(
    z.object({
      label: z.string(),
      positiveSpeechThreshold: StringOrNumberSchema,
      negativeSpeechThreshold: StringOrNumberSchema,
      lookbackFrames: StringOrNumberSchema,
    }),
  ),
});
export type VADInputType = z.infer<typeof VADInputSchema>;

export const VADOutputSchema = z.object({
  latency: z.number(),
  start: z.number(),
  end: z.number(),
  outputs: z.array(
    z.object({
      label: z.string(),
      isSpeaking: z.boolean(),
    }),
  ),
});
export type VADOutputType = z.infer<typeof VADOutputSchema>;

export const VADEndOfTurnInputSchema = z.object({
  context: z.array(ChatMessageSchema),
});
export type VADEndOfTurnInputType = z.infer<typeof VADEndOfTurnInputSchema>;

export const VADEndOfTurnOutputSchema = z.object({
  endOfTurnProbability: z.number(),
});
export type VADEndOfTurnOutputType = z.infer<typeof VADEndOfTurnOutputSchema>;

export function queryStringFromVADInput(input: VADInputType): string {
  const params = new URLSearchParams();

  if (input.format) {
    params.append("format", input.format);
  }
  if (input.frameDurationMs) {
    params.append("frameDurationMs", input.frameDurationMs.toString());
  }

  input.inputs.forEach((inputItem, index) => {
    Object.entries(inputItem).forEach(([key, value]) => {
      params.append(`inputs[${index}][${key}]`, value?.toString() ?? "");
    });
  });

  return params.toString();
}

export function vadInputFromQueryString(queryString: string): VADInputType {
  const params = new URLSearchParams(queryString);
  const result: Partial<VADInputType> = {};
  const formatParse = AudioFormatSchema.safeParse(params.get("format"));

  result.format = formatParse.success ? formatParse.data : "mulaw_8000";
  result.frameDurationMs = params.get("frameDurationMs")
    ? Number(params.get("frameDurationMs"))
    : undefined;

  result.inputs = [];
  for (const [key, value] of params.entries()) {
    const match = key.match(/inputs\[(\d+)\]\[(\w+)\]/);
    if (match) {
      const [, index, field] = match;
      if (!index || !field) {
        continue;
      }
      const i = Number(index);
      if (!result.inputs[i]) {
        result.inputs[i] = {} as VADInputType["inputs"][0];
      }
      (result.inputs[i] as Record<string, unknown>)[field] =
        field === "label" ? value : Number(value);
    }
  }

  return VADInputSchema.parse(result);
}
