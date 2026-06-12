import {
  afterAll,
  beforeAll,
  describe,
  expect,
  mock,
  spyOn,
  test,
} from "bun:test";

const sharedMockInferenceSessionCreate = mock(() =>
  Promise.resolve({ run: mock(), release: mock() }),
);
const sharedMockTensor = mock();

mock.module("onnxruntime-node", () => ({
  InferenceSession: { create: sharedMockInferenceSessionCreate },
  Tensor: sharedMockTensor,
}));

mock.module("@huggingface/transformers", () => ({
  AutoTokenizer: { from_pretrained: mock(() => Promise.resolve({})) },
}));

import { EndOfTurnDetector } from "../../src/services/endOfTurn";

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Minimal BoundedSemaphore fake. Returned by stubbing `getSessionSemaphore`
 * on the detector's internal model so we can drive queue-depth behavior
 * deterministically.
 */
class FakeBoundedSemaphore {
  private _value: number;
  private _waiters: ((value: void) => void)[] = [];
  private _maxQueueDepth: number | null;
  private _pending = 0;

  constructor(value: number, maxQueueDepth?: number) {
    this._value = value;
    this._maxQueueDepth = maxQueueDepth ?? null;
  }

  public isQueueFull(): boolean {
    return this._maxQueueDepth !== null && this._pending >= this._maxQueueDepth;
  }

  public async acquire(): Promise<void> {
    this._pending++;
    try {
      if (this._value > 0) {
        this._value--;
        return;
      }
      await new Promise<void>((resolve) => {
        this._waiters.push(resolve);
      });
    } finally {
      this._pending--;
    }
  }

  public release(): void {
    if (this._waiters.length > 0) {
      const resolve = this._waiters.shift()!;
      resolve();
    } else {
      this._value++;
    }
  }
}

function buildDetector(opts?: {
  queueDepth?: number;
  semaphore?: FakeBoundedSemaphore;
}): { detector: EndOfTurnDetector; semaphore: FakeBoundedSemaphore } {
  const detector = new EndOfTurnDetector({
    modelPath: "/tmp/fake-eot.onnx",
    tokenizerPath: "/tmp/fake-tokenizer",
    queueDepth: opts?.queueDepth,
  });
  const semaphore =
    opts?.semaphore ?? new FakeBoundedSemaphore(1, opts?.queueDepth);
  const model = (detector as unknown as { model: unknown }).model as {
    getSessionSemaphore: () => FakeBoundedSemaphore;
    getSession: () => Promise<unknown>;
  };
  model.getSessionSemaphore = () => semaphore;
  // Force session lookups to fail so detect() exits the try via catch
  // without performing real inference.
  model.getSession = () => Promise.reject(new Error("test mock"));
  return { detector, semaphore };
}

describe("EndOfTurnDetector queue depth limiting", () => {
  // detect() catches the forced rejection and logs via console.error. That
  // log is expected; silence it to keep test output readable.
  let consoleErrorSpy: ReturnType<typeof spyOn>;
  beforeAll(() => {
    consoleErrorSpy = spyOn(console, "error").mockImplementation(() => {});
  });
  afterAll(() => {
    consoleErrorSpy.mockRestore();
  });

  test("sheds load when queue depth exceeds threshold", async () => {
    const semaphore = new FakeBoundedSemaphore(1, 3);
    await semaphore.acquire();

    const { detector } = buildDetector({ queueDepth: 3, semaphore });

    const input = {
      context: [
        { role: "user" as const, content: "hello" },
      ],
    };
    const promises: Promise<{ endOfTurnProbability: number }>[] = [];

    for (let i = 0; i < 3; i++) {
      promises.push(detector.detect(input));
    }

    await sleep(10);

    const shedResult = await detector.detect(input);
    expect(shedResult.endOfTurnProbability).toBe(0.5);

    semaphore.release();
    await Promise.all(promises);
  });

  test("does not shed when queueDepth is not configured", async () => {
    const semaphore = new FakeBoundedSemaphore(1);
    await semaphore.acquire();

    const { detector } = buildDetector({ semaphore });

    const input = {
      context: [
        { role: "user" as const, content: "hello" },
      ],
    };
    const promises: Promise<{ endOfTurnProbability: number }>[] = [];

    for (let i = 0; i < 5; i++) {
      promises.push(detector.detect(input));
    }

    await sleep(10);
    expect(semaphore.isQueueFull()).toBe(false);

    semaphore.release();
    await Promise.all(promises);
  });

  test("returns default when signal is already aborted after acquire", async () => {
    const { detector } = buildDetector();

    const ac = new AbortController();
    ac.abort();

    const input = {
      context: [
        { role: "user" as const, content: "hello" },
      ],
    };
    const result = await detector.detect(input, ac.signal);
    expect(result.endOfTurnProbability).toBe(0.5);
  });
});
