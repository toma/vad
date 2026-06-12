import { describe, expect, mock, spyOn, test } from "bun:test";
import Silero from "../../src/services/silero";

const FAKE_MODEL_PATH = "/tmp/fake-silero.onnx";

interface FakeSession {
  run: ReturnType<typeof mock>;
  released: boolean;
  release: ReturnType<typeof mock>;
}

interface FakeSemaphore {
  acquire: ReturnType<typeof mock>;
  release: ReturnType<typeof mock>;
  isQueueFull: () => boolean;
}

/**
 * Replace the internal BaseOnnxModel on a Silero instance with an in-memory
 * stub that returns the given speech probability and never touches the real
 * onnxruntime. Returns the stub so tests can assert against it.
 */
function stubModel(
  silero: Silero,
  speechProb = 0.7,
): { session: FakeSession; semaphore: FakeSemaphore } {
  const session: FakeSession = {
    run: mock().mockResolvedValue({
      stateN: { dispose: mock() },
      output: {
        data: [speechProb],
        dims: [1],
        dispose: mock(),
      },
    }),
    released: false,
    release: mock(async () => {
      session.released = true;
    }),
  };
  const semaphore: FakeSemaphore = {
    acquire: mock().mockResolvedValue(undefined),
    release: mock(),
    isQueueFull: () => false,
  };
  (silero as unknown as { model: unknown }).model = {
    getSession: () => Promise.resolve(session),
    getSessionSemaphore: () => semaphore,
    releaseSession: async () => {
      if (!session.released) {
        await session.release();
      }
    },
  };
  return { session, semaphore };
}

describe("Silero", () => {
  describe("constructor", () => {
    test("initializes with default state and null context", () => {
      const silero = new Silero({ modelPath: FAKE_MODEL_PATH });
      expect(silero["context"]).toBeNull();
      expect(silero["state"]).toBeDefined();
      expect(silero["state"]!.dims).toEqual([2, 1, 128]);
    });
  });

  describe("destroy", () => {
    test("disposes state, clears context, and releases the session", async () => {
      const silero = new Silero({ modelPath: FAKE_MODEL_PATH });
      const { session } = stubModel(silero);
      const disposeSpy = spyOn(silero["state"]!, "dispose");

      await silero.destroy();

      expect(disposeSpy).toHaveBeenCalled();
      expect(silero["state"]).toBeNull();
      expect(silero["context"]).toBeNull();
      expect(session.release).toHaveBeenCalled();
    });
  });

  describe("process", () => {
    test("returns 0 if state is null", async () => {
      const silero = new Silero({ modelPath: FAKE_MODEL_PATH });
      stubModel(silero);
      silero["state"] = null;

      const result = await silero.process(new Int16Array(512));

      expect(result).toBe(0);
    });

    test("returns the probability from the underlying session", async () => {
      const silero = new Silero({ modelPath: FAKE_MODEL_PATH });
      stubModel(silero, 0.7);
      const audioFrame = new Int16Array(512).fill(1);

      const result = await silero.process(audioFrame);

      expect(result).toBeCloseTo(0.7, 6);
    });

    test("handles 8kHz sample rate", async () => {
      const silero = new Silero({ modelPath: FAKE_MODEL_PATH });
      stubModel(silero, 0.7);
      const audioFrame = new Int16Array(256).fill(1);

      const result = await silero.process(audioFrame, 8000);

      expect(result).toBeCloseTo(0.7, 6);
    });

    test("pads short audio frames", async () => {
      const silero = new Silero({ modelPath: FAKE_MODEL_PATH });
      stubModel(silero, 0.7);
      const audioFrame = new Int16Array(100).fill(1);

      const result = await silero.process(audioFrame);

      expect(result).toBeCloseTo(0.7, 6);
    });

    test("truncates long audio frames", async () => {
      const silero = new Silero({ modelPath: FAKE_MODEL_PATH });
      stubModel(silero, 0.7);
      const audioFrame = new Int16Array(1000).fill(1);

      const result = await silero.process(audioFrame);

      expect(result).toBeCloseTo(0.7, 6);
    });
  });
});
