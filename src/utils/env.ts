import { z } from "zod";

const EnvSchema = z.object({
  ENV: z.enum(["local", "staging", "production"]).default("local"),
  DEBUG: z
    .string()
    .optional()
    .default("false")
    .transform((val) => val.toLowerCase() === "true" || val === "1"),
  PORT: z.string().optional().default("8086"),
  EOT_QUEUE_DEPTH: z.coerce.number().optional().default(3),
});

export default EnvSchema.parse(process.env);
