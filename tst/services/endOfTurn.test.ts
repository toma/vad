import { afterAll, afterEach, describe, expect, mock, test } from "bun:test";
import { sleep } from "~/utils/async";

const sharedMockInferenceSessionCreate = mock(() =>
  Promise.resolve({ run: mock(), release: mock() }),
);
const sharedMockTensor = mock();

mock.module("onnxruntime-node", () => ({
  InferenceSession: { create: sharedMockInferenceSessionCreate },
  Tensor: sharedMockTensor,
}));

mock.module("~/utils/hf", () => ({
  getModelPath: mock(() => Promise.resolve("/tmp/fake-model.onnx")),
}));

mock.module("@huggingface/transformers", () => ({
  AutoTokenizer: { from_pretrained: mock(() => Promise.resolve({})) },
}));

mock.module("~/utils/math", () => ({
  softmax: mock(),
}));

import { endOfTurnHandler } from "../../src/services/endOfTurn";
import { EndOfTurnModel, EndOfTurnTokenizer } from "../../src/utils/models/eot";

const originalGetSession = EndOfTurnModel.getSession.bind(EndOfTurnModel);
const originalGetInstance =
  EndOfTurnTokenizer.getInstance.bind(EndOfTurnTokenizer);
const originalGetSemaphore =
  EndOfTurnModel.getSessionSemaphore.bind(EndOfTurnModel);

/**
 * Minimal BoundedSemaphore fake. Avoids importing async-mutex directly
 * because other test files mock that module globally.
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

function createSemaphore(
  value = 1,
  maxQueueDepth?: number,
): FakeBoundedSemaphore {
  return new FakeBoundedSemaphore(value, maxQueueDepth);
}

function restoreOriginals(): void {
  EndOfTurnModel.getSession = originalGetSession;
  EndOfTurnModel.getSessionSemaphore = originalGetSemaphore;
  EndOfTurnTokenizer.getInstance = originalGetInstance;
}

describe("endOfTurnHandler queue depth limiting", () => {
  afterEach(() => {
    restoreOriginals();
  });

  afterAll(() => {
    restoreOriginals();
  });

  test("sheds load when queue depth exceeds threshold", async () => {
    const semaphore = createSemaphore(1, 3);
    await semaphore.acquire();

    EndOfTurnModel.getSessionSemaphore = mock(() => semaphore) as never;
    EndOfTurnModel.getSession = mock(() =>
      Promise.reject(new Error("test mock")),
    ) as never;
    EndOfTurnTokenizer.getInstance = mock(() =>
      Promise.reject(new Error("test mock")),
    ) as never;

    const input = {
      context: [
        { role: "user" as const, content: "hello", timestamp: Date.now() },
      ],
    };
    const promises: Promise<{ endOfTurnProbability: number }>[] = [];

    for (let i = 0; i < 3; i++) {
      promises.push(endOfTurnHandler(input));
    }

    await sleep(10);

    const shedResult = await endOfTurnHandler(input);
    expect(shedResult.endOfTurnProbability).toBe(0.5);

    semaphore.release();
    await Promise.all(promises);
  });

  test("does not shed when queueDepth is not configured", async () => {
    const semaphore = createSemaphore(1);
    await semaphore.acquire();

    EndOfTurnModel.getSessionSemaphore = mock(() => semaphore) as never;
    EndOfTurnModel.getSession = mock(() =>
      Promise.reject(new Error("test mock")),
    ) as never;
    EndOfTurnTokenizer.getInstance = mock(() =>
      Promise.reject(new Error("test mock")),
    ) as never;

    const input = {
      context: [
        { role: "user" as const, content: "hello", timestamp: Date.now() },
      ],
    };
    const promises: Promise<{ endOfTurnProbability: number }>[] = [];

    for (let i = 0; i < 5; i++) {
      promises.push(endOfTurnHandler(input));
    }

    await sleep(10);
    expect(semaphore.isQueueFull()).toBe(false);

    semaphore.release();
    await Promise.all(promises);
  });

  test("returns default when signal is already aborted after acquire", async () => {
    const semaphore = createSemaphore(1);

    EndOfTurnModel.getSessionSemaphore = mock(() => semaphore) as never;
    EndOfTurnModel.getSession = mock(() =>
      Promise.reject(new Error("test mock")),
    ) as never;
    EndOfTurnTokenizer.getInstance = mock(() =>
      Promise.reject(new Error("test mock")),
    ) as never;

    const ac = new AbortController();
    ac.abort();

    const input = {
      context: [
        { role: "user" as const, content: "hello", timestamp: Date.now() },
      ],
    };
    const result = await endOfTurnHandler(input, ac.signal);
    expect(result.endOfTurnProbability).toBe(0.5);
  });
});
