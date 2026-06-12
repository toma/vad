export type ChatRole = "user" | "assistant" | "system";

/**
 * A single message in a conversation.
 * @property role - The role of the message sender (user, assistant, or system)
 * @property content - The text content of the message
 * @property timestamp - Unix timestamp (ms) when the message was created
 */
export interface ChatMessage {
  role: ChatRole;
  content: string;
  timestamp: number;
}

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
