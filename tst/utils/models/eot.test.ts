import { afterAll, beforeAll, describe, expect, it, mock } from "bun:test";
import type { ChatMessage } from "~/models/chat";

// Mock dependencies
const mockGetModelPath = mock(() => Promise.resolve("/tmp/fake-model.onnx"));
const mockFromPretrained = mock();
const mockInferenceSession = mock();
const mockTensor = mock();

mock.module("~/utils/hf", () => ({
  getModelPath: mockGetModelPath,
}));

mock.module("@huggingface/transformers", () => ({
  AutoTokenizer: { from_pretrained: mockFromPretrained },
}));

mock.module("onnxruntime-node", () => ({
  InferenceSession: {
    create: mockInferenceSession,
  },
  Tensor: mockTensor,
}));

// Import after mocks
import {
  EndOfTurnModel,
  EndOfTurnTokenizer,
  initializeEndOfTurn,
} from "../../../src/utils/models/eot";

describe("End of Turn (EOT) Model", () => {
  let eotTokenizer: EndOfTurnTokenizer;

  beforeAll(async () => {
    const mockTokenizerInstance = {
      encode: mock().mockImplementation((text: string) => {
        // Simple mock encoding
        return text.split(" ").map((word, i) => i + 1);
      }),
      decode: mock().mockImplementation((tokens: number[]) => {
        // Simple mock decoding
        return tokens.map((t) => `token_${t}`).join(" ");
      }),
      apply_chat_template: mock().mockImplementation(
        (messages: any[], _options: any) => {
          // Mock chat template application
          return messages.map((m) => `${m.role}: ${m.content}`).join(" ");
        },
      ),
    };

    mockFromPretrained.mockResolvedValue(mockTokenizerInstance);

    // Set up ONNX mocks
    const mockSession = {
      run: mock().mockResolvedValue({
        logits: {
          dims: [1, 10, 1000],
          data: new Array(10000).fill(0.1),
          dispose: mock(),
        },
      }),
      release: mock(),
    };

    mockInferenceSession.mockResolvedValue(mockSession);
    mockTensor.mockImplementation((type, data, shape) => ({
      type,
      data,
      shape,
      dispose: mock(),
    }));

    // Initialize the EOT model and tokenizer
    await initializeEndOfTurn();
    eotTokenizer = await EndOfTurnTokenizer.getInstance();
  });

  afterAll(async () => {
    // Clean up resources
    const session = await EndOfTurnModel.getSession();
    session.release();
  });

  describe("EndOfTurnTokenizer", () => {
    it("should be a singleton instance", async () => {
      const instance1 = await EndOfTurnTokenizer.getInstance();
      const instance2 = await EndOfTurnTokenizer.getInstance();
      expect(instance1).toBe(instance2);
    });

    it("should have correct EOT token", () => {
      expect(eotTokenizer.eotToken).toBe("<|im_end|>");
    });

    it("should have valid EOT index", () => {
      const eotIndex = eotTokenizer.eotIndex;
      expect(eotIndex).toBeGreaterThanOrEqual(0);
      expect(typeof eotIndex).toBe("number");
    });

    it("should encode and decode text correctly", () => {
      const testText = "Hello world";
      const encoded = eotTokenizer.encode(testText);
      const decoded = eotTokenizer.decode(encoded);

      expect(Array.isArray(encoded)).toBe(true);
      expect(encoded.length).toBeGreaterThan(0);
      expect(typeof decoded).toBe("string");
    });

    it("should normalize text correctly", () => {
      const testContext: ChatMessage[] = [
        {
          role: "user",
          content: "Hello, World! How are you?",
          timestamp: 1234567890,
        },
      ];

      const formatted = eotTokenizer.formatContext(testContext);
      expect(typeof formatted).toBe("string");
      expect(formatted.length).toBeGreaterThan(0);
    });

    it("should format conversation context correctly", () => {
      const context: ChatMessage[] = [
        {
          role: "user",
          content: "Hello",
          timestamp: 1234567890,
        },
        {
          role: "assistant",
          content: "Hi there! How can I help you?",
          timestamp: 1234567891,
        },
        {
          role: "user",
          content: "I have a question",
          timestamp: 1234567892,
        },
      ];

      const formatted = eotTokenizer.formatContext(context);
      expect(typeof formatted).toBe("string");
      expect(formatted.length).toBeGreaterThan(0);
      expect(formatted).toContain("hello");
      expect(formatted).toContain("hi there how can i help you");
      expect(formatted).toContain("i have a question");
    });
  });

  describe("EndOfTurnModel", () => {
    it("should load the model successfully", async () => {
      const session = await EndOfTurnModel.getSession();
      expect(session).toBeDefined();
      expect(typeof session.run).toBe("function");
    });

    it("should have a semaphore for thread safety", () => {
      const semaphore = EndOfTurnModel.getSessionSemaphore();
      expect(semaphore).toBeDefined();
      expect(typeof semaphore.acquire).toBe("function");
      expect(typeof semaphore.release).toBe("function");
    });
  });

  describe("EOT Integration", () => {
    it("should process a simple conversation and return probability", async () => {
      const context: ChatMessage[] = [
        {
          role: "user",
          content: "Hello, how are you today?",
          timestamp: 1234567890,
        },
      ];

      // Get the session
      const session = await EndOfTurnModel.getSession();

      // Format and tokenize the context
      const formattedContext = eotTokenizer.formatContext(context);
      const tokens = eotTokenizer.encode(formattedContext);

      expect(tokens.length).toBeGreaterThan(0);

      // Create input tensor
      const inputTensor = new (await import("onnxruntime-node")).Tensor(
        "int64",
        tokens,
        [1, tokens.length],
      );

      // Run inference
      const result = await session.run({ input_ids: inputTensor });
      expect(result.logits).toBeDefined();
      expect(result.logits.dims).toBeDefined();
      expect(result.logits.dims.length).toBeGreaterThanOrEqual(2);

      // Clean up
      inputTensor.dispose();
      result.logits.dispose();
    });

    it("should handle different conversation lengths", async () => {
      const shortContext: ChatMessage[] = [
        {
          role: "user",
          content: "Hi",
          timestamp: 1234567890,
        },
      ];

      const longContext: ChatMessage[] = [
        {
          role: "user",
          content:
            "Hello, I have a very long question that goes on and on about various topics including technology, science, and philosophy. I want to understand how this system works and whether it can properly detect when I'm done speaking.",
          timestamp: 1234567890,
        },
        {
          role: "assistant",
          content:
            "I understand you have a complex question. Please go ahead and ask it.",
          timestamp: 1234567891,
        },
        {
          role: "user",
          content:
            "Thank you for your patience. Let me explain my situation in detail...",
          timestamp: 1234567892,
        },
      ];

      const shortFormatted = eotTokenizer.formatContext(shortContext);
      const longFormatted = eotTokenizer.formatContext(longContext);

      expect(shortFormatted.length).toBeLessThan(longFormatted.length);
      expect(shortFormatted).toContain("hi");
      expect(longFormatted).toContain("hello i have a very long question");
    });

    it("should handle edge cases", () => {
      // Empty context
      const emptyContext: ChatMessage[] = [];
      const emptyFormatted = eotTokenizer.formatContext(emptyContext);
      expect(typeof emptyFormatted).toBe("string");

      // Context with special characters
      const specialContext: ChatMessage[] = [
        {
          role: "user",
          content: "Test with @#$%^&*() symbols and numbers 12345!",
          timestamp: 1234567890,
        },
      ];
      const specialFormatted = eotTokenizer.formatContext(specialContext);
      expect(typeof specialFormatted).toBe("string");
      expect(specialFormatted).toContain(
        "test with  symbols and numbers 12345",
      );
    });

    it("should reduce extremely large contexts to MAX_MESSAGES limit", () => {
      // Create a context with many more than 4 messages (MAX_MESSAGES = 4)
      const largeContext: ChatMessage[] = [];

      // Add 10 alternating user/assistant messages
      for (let i = 0; i < 9; i++) {
        largeContext.push({
          role: i % 2 === 0 ? "user" : "assistant",
          content: `Message ${i + 1}: This is message number ${i + 1} in the conversation.`,
          timestamp: 1234567890 + i,
        });
      }

      expect(largeContext.length).toBe(9);

      // Format the large context
      const formattedLarge = eotTokenizer.formatContext(largeContext);

      // Create a smaller context with just the last 4 messages
      const smallContext = largeContext.slice(-4);
      const formattedSmall = eotTokenizer.formatContext(smallContext);

      // The formatted large context should be reduced and similar to the small context
      expect(typeof formattedLarge).toBe("string");
      expect(typeof formattedSmall).toBe("string");

      // The formatted large context should contain the last 4 messages
      expect(formattedLarge).toContain("message 8");
      expect(formattedLarge).toContain("message 9");

      // But should NOT contain the earliest messages
      expect(formattedLarge).not.toContain("message 1");
      expect(formattedLarge).not.toContain("message 2");
      expect(formattedLarge).not.toContain("message 3");
      expect(formattedLarge).not.toContain("message 4");
      expect(formattedLarge).not.toContain("message 5");
    });
  });
});
