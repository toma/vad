import {
  afterEach,
  beforeEach,
  describe,
  expect,
  mock,
  spyOn,
  test,
} from "bun:test";
import { InferenceSession } from "onnxruntime-node";
import { sleep } from "~/utils/async";

// Mock dependencies
const mockGetModelPath = mock();
const mockOnTerminate = mock();
const mockSessionRun = mock();
const mockSessionRelease = mock();
const mockSemaphoreAcquire = mock();
const mockSemaphoreRelease = mock();

mock.module("~/utils/hf", () => ({
  getModelPath: mockGetModelPath,
}));
mock.module("~/utils/common", () => ({
  onTerminate: mockOnTerminate,
}));
mock.module("onnxruntime-node", () => ({
  InferenceSession: {
    create: mock(),
  },
}));
mock.module("async-mutex", () => ({
  Semaphore: mock().mockImplementation(() => ({
    acquire: mockSemaphoreAcquire,
    release: mockSemaphoreRelease,
  })),
}));

// Import after mocks
import {
  BaseOnnxModel,
  BaseOnnxModelSession,
  BoundedSemaphore,
} from "../../../src/utils/models/base";

// Helper to create a fake session
function createFakeSession() {
  return {
    run: mockSessionRun,
    release: mockSessionRelease,
  } as unknown as InferenceSession;
}

describe("BaseOnnxModel", () => {
  beforeEach(() => {
    mockGetModelPath.mockReset();
    mockOnTerminate.mockReset();
    mockSessionRun.mockReset();
    mockSessionRelease.mockReset();
    mockSemaphoreAcquire.mockReset();
    mockSemaphoreRelease.mockReset();
    (InferenceSession.create as any).mockReset();
    // Don't actually call the onTerminate callback during tests
    mockOnTerminate.mockImplementation(() => {});
  });

  test("should initialize and return a session", async () => {
    const fakeSession = createFakeSession();
    mockGetModelPath.mockResolvedValue("/tmp/model.onnx");
    (InferenceSession.create as any).mockResolvedValue(fakeSession);

    const model = new BaseOnnxModel({ repo: "r", file: "model.onnx" });
    const session = await model.getSession();
    expect(session).toBeInstanceOf(BaseOnnxModelSession);
    expect(mockGetModelPath).toHaveBeenCalledWith({
      repo: "r",
      file: "model.onnx",
    });
    expect(InferenceSession.create).toHaveBeenCalledWith("/tmp/model.onnx", {
      graphOptimizationLevel: "all",
      enableCpuMemArena: false,
      enableMemPattern: false,
      executionMode: "sequential",
    });
    expect(mockOnTerminate).toHaveBeenCalled();
  });

  test("should return the same session if not released", async () => {
    const fakeModel = { fake: true };
    const fakeSession = createFakeSession();
    mockGetModelPath.mockResolvedValue(fakeModel);
    (InferenceSession.create as any).mockResolvedValue(fakeSession);

    const model = new BaseOnnxModel({ repo: "r", file: "model.onnx" });
    const session1 = await model.getSession();
    const session2 = await model.getSession();
    expect(session1).toBe(session2);
    expect(mockGetModelPath).toHaveBeenCalledTimes(1);
    expect(InferenceSession.create).toHaveBeenCalledTimes(1);
  });

  test("should re-initialize if session is released", async () => {
    const fakeModel = { fake: true };
    const fakeSession1 = createFakeSession();
    const fakeSession2 = createFakeSession();
    mockGetModelPath.mockResolvedValue(fakeModel);
    (InferenceSession.create as any)
      .mockResolvedValueOnce(fakeSession1)
      .mockResolvedValueOnce(fakeSession2);

    const model = new BaseOnnxModel({ repo: "r", file: "model.onnx" });
    const session1 = await model.getSession();
    // Simulate release
    (session1 as any)._released = true;
    const session2 = await model.getSession();
    expect(session2).not.toBe(session1);
    expect(InferenceSession.create).toHaveBeenCalledTimes(2);
  });

  test("should wait for initialization if already in progress", async () => {
    const fakeModel = { fake: true };
    const fakeSession = createFakeSession();
    mockGetModelPath.mockResolvedValue(fakeModel);
    (InferenceSession.create as any).mockImplementation(async () => {
      await sleep(10);
      return fakeSession;
    });

    const model = new BaseOnnxModel({ repo: "r", file: "model.onnx" });
    const [session1, session2] = await Promise.all([
      model.getSession(),
      model.getSession(),
    ]);
    expect(session1).toBe(session2);
    expect(InferenceSession.create).toHaveBeenCalledTimes(1);
  });

  test("should throw and log error if initialization fails", async () => {
    const error = new Error("Failed to load model");
    mockGetModelPath.mockRejectedValue(error);
    const model = new BaseOnnxModel({ repo: "r", file: "bad.onnx" });
    const consoleSpy = spyOn(console, "error").mockImplementation(() => {});

    await expect(model.getSession()).rejects.toThrow("Failed to load model");
    expect(consoleSpy).toHaveBeenCalledWith(
      "Failed to initialize model r/bad.onnx:",
      error,
    );

    consoleSpy.mockRestore();
  });

  test("getSessionSemaphore returns a singleton semaphore", () => {
    const model = new BaseOnnxModel({ repo: "r", file: "model.onnx" });
    const sem1 = model.getSessionSemaphore();
    const sem2 = model.getSessionSemaphore();
    expect(sem1).toBeDefined();
    expect(sem1).toBe(sem2);
    expect(sem1.acquire).toBeDefined();
    expect(sem1.release).toBeDefined();
  });
});

describe("BoundedSemaphore", () => {
  beforeEach(() => {
    mockSemaphoreAcquire.mockReset();
    mockSemaphoreRelease.mockReset();
  });

  test("isQueueFull returns false when no queueDepth configured", () => {
    const sem = new BoundedSemaphore(1);
    expect(sem.isQueueFull()).toBe(false);
  });

  test("isQueueFull tracks pending acquires", async () => {
    const resolvers: (() => void)[] = [];
    mockSemaphoreAcquire.mockImplementation(
      () => new Promise<void>((r) => resolvers.push(r)),
    );

    const sem = new BoundedSemaphore(1, 2);

    const p1 = sem.acquire();
    expect(sem.isQueueFull()).toBe(false);

    const p2 = sem.acquire();
    expect(sem.isQueueFull()).toBe(true);

    for (const r of resolvers) {
      r();
    }
    await p1;
    await p2;

    expect(sem.isQueueFull()).toBe(false);
  });

  test("acquire decrements pending even when inner semaphore rejects", async () => {
    mockSemaphoreAcquire.mockRejectedValue(new Error("fail"));

    const sem = new BoundedSemaphore(1, 2);
    await sem.acquire().catch(() => {});

    expect(sem.isQueueFull()).toBe(false);
  });

  test("release delegates to inner semaphore", () => {
    const sem = new BoundedSemaphore(1);
    sem.release();
    expect(mockSemaphoreRelease).toHaveBeenCalledTimes(1);
  });
});

describe("BaseOnnxModel with queueDepth", () => {
  afterEach(() => {
    mockSemaphoreAcquire.mockReset();
    mockSemaphoreRelease.mockReset();
  });

  test("getSessionSemaphore respects queueDepth from constructor", async () => {
    const resolvers: (() => void)[] = [];
    mockSemaphoreAcquire.mockImplementation(
      () => new Promise<void>((r) => resolvers.push(r)),
    );

    const model = new BaseOnnxModel({ repo: "r", file: "model.onnx" }, { queueDepth: 1 });
    const sem = model.getSessionSemaphore();

    const p1 = sem.acquire();
    expect(sem.isQueueFull()).toBe(true);

    for (const r of resolvers) {
      r();
    }
    await p1;
  });
});

describe("BaseOnnxModelSession", () => {
  let fakeSession: any;
  beforeEach(() => {
    fakeSession = {
      run: mockSessionRun,
      release: mockSessionRelease,
    };
    mockSessionRun.mockReset();
    mockSessionRelease.mockReset();
  });

  test("run should call session.run if not released", async () => {
    const session = new BaseOnnxModelSession(fakeSession);
    mockSessionRun.mockResolvedValue("output");
    const result = await session.run({ input: 1 });
    expect(result).toBe("output");
    expect(mockSessionRun).toHaveBeenCalledWith({ input: 1 });
  });

  test("run should throw if released", async () => {
    const session = new BaseOnnxModelSession(fakeSession);
    (session as any)._released = true;
    const consoleSpy = spyOn(console, "error").mockImplementation(() => {});

    let error: Error | null = null;
    try {
      await session.run({ input: 1 });
    } catch (err) {
      error = err as Error;
    }

    expect(error).toBeInstanceOf(Error);
    expect(error?.message).toBe("Session has been released");

    consoleSpy.mockRestore();
  });

  test("release should call session.release and set released", async () => {
    const session = new BaseOnnxModelSession(fakeSession);
    mockSessionRelease.mockResolvedValue(undefined);
    await session.release();
    expect(mockSessionRelease).toHaveBeenCalled();
    expect(session.released).toBe(true);
  });

  test("release should not call session.release if already released", async () => {
    const session = new BaseOnnxModelSession(fakeSession);
    (session as any)._released = true;
    await session.release();
    expect(mockSessionRelease).not.toHaveBeenCalled();
  });

  test("release should log error if session.release throws", async () => {
    const session = new BaseOnnxModelSession(fakeSession);
    const error = new Error("fail");
    mockSessionRelease.mockRejectedValue(error);
    const consoleSpy = spyOn(console, "error").mockImplementation(() => {});

    await session.release();
    expect(consoleSpy).toHaveBeenCalledWith("Error releasing session:", error);
    expect(session.released).toBe(true);

    consoleSpy.mockRestore();
  });
});
