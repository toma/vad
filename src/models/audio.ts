import { z } from "zod";

export const AudioFormatSchema = z.enum(["mulaw_8000", "pcm_16000"]);
export type AudioFormatType = z.infer<typeof AudioFormatSchema>;
