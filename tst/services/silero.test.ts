import { describe, expect, mock, spyOn, test } from "bun:test";
import { Tensor } from "onnxruntime-node";
import Silero from "../../src/services/silero";

// Mock onnxruntime-node
mock.module("onnxruntime-node", () => ({
  Tensor: mock().mockImplementation((type, data, dims) => ({
    data,
    dims,
    dispose: mock(),
    location: "cpu",
    type,
  })),
  InferenceSession: {
    create: mock().mockResolvedValue({
      run: mock().mockResolvedValue({}),
      release: mock().mockResolvedValue(undefined),
    }),
  },
}));

// Mock SileroV5Model
mock.module("../../src/utils/models/silero", () => ({
  SileroV5Model: {
    getSessionSemaphore: mock().mockReturnValue({
      acquire: mock().mockResolvedValue(undefined),
      release: mock(),
    }),
    getSession: mock().mockResolvedValue({
      run: mock().mockResolvedValue({
        stateN: new Tensor("float32", new Float32Array(256), [2, 1, 128]),
        output: new Tensor("float32", new Float32Array([0.7]), [1]),
      }),
      release: mock().mockResolvedValue(undefined),
    }),
  },
}));

describe("Silero", () => {
  describe("constructor", () => {
    test("should initialize with correct default values", () => {
      const silero = new Silero();
      expect(silero["context"]).toBeNull();
      expect(silero["state"]).toBeDefined();
      expect(silero["state"]!.dims).toEqual([2, 1, 128]);
    });
  });

  describe("destroy", () => {
    test("should dispose of state and reset context", async () => {
      const silero = new Silero();
      const disposeSpy = spyOn(silero["state"]!, "dispose");

      await silero.destroy();

      expect(disposeSpy).toHaveBeenCalled();
      expect(silero["state"]).toBeNull();
      expect(silero["context"]).toBeNull();
    });
  });

  describe("process", () => {
    test("should return 0 if state is null", async () => {
      const silero = new Silero();
      silero["state"] = null;

      const result = await silero.process(new Int16Array(512));

      expect(result).toBe(0);
    });

    test("should process audio frame correctly", async () => {
      const silero = new Silero();
      const audioFrame = new Int16Array(512).fill(1);

      const result = await silero.process(audioFrame);

      expect(result).toBeCloseTo(0.7, 6);
    });

    test("should handle different sample rates", async () => {
      const silero = new Silero();
      const audioFrame = new Int16Array(256).fill(1);

      const result = await silero.process(audioFrame, 8000);

      expect(result).toBeCloseTo(0.7, 6);
    });

    test("should pad short audio frames", async () => {
      const silero = new Silero();
      const audioFrame = new Int16Array(100).fill(1);

      const result = await silero.process(audioFrame);

      expect(result).toBeCloseTo(0.7, 6);
    });

    test("should truncate long audio frames", async () => {
      const silero = new Silero();
      const audioFrame = new Int16Array(1000).fill(1);

      const result = await silero.process(audioFrame);

      expect(result).toBeCloseTo(0.7, 6);
    });
  });
});
