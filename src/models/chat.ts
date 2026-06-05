import { z } from "zod";

export const ChatRoleSchema = z.enum(["user", "assistant", "system"]);
export type ChatRole = z.infer<typeof ChatRoleSchema>;

/**
 * A single message in a conversation.
 * @property role - The role of the message sender (user, assistant, or system)
 * @property content - The text content of the message
 * @property timestamp - Unix timestamp (ms) when the message was created
 */
export const ChatMessageSchema = z.object({
  role: ChatRoleSchema,
  content: z.string(),
  timestamp: z.number(),
});
export type ChatMessage = z.infer<typeof ChatMessageSchema>;

/**
 * Collapse consecutive messages that share the same role into one.
 * - Joins folded content with a single space and trims it.
 * - Drops messages that end up empty after folding.
 */
export function foldContext(context: ChatMessage[]): ChatMessage[] {
  if (context.length === 0) {
    return [];
  }

  const folded: ChatMessage[] = [];
  for (const message of context) {
    const lastMessage = folded[folded.length - 1];
    if (lastMessage && lastMessage.role === message.role) {
      lastMessage.content = [lastMessage.content, message.content]
        .filter(Boolean)
        .join(" ")
        .trim();
    } else {
      folded.push({ ...message });
    }
  }

  return folded.filter((message) => message.content.trim() !== "");
}
