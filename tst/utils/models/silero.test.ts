import type { BunFile } from "bun";
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  mock,
  spyOn,
  test,
} from "bun:test";

// Mock the onnxruntime-node module
const mockRun = mock().mockResolvedValue({});
const mockCreate = mock().mockResolvedValue({
  run: mockRun,
});

mock.module("onnxruntime-node", () => ({
  default: {
    InferenceSession: {
      create: mockCreate,
    },
  },
}));

const Bun = global.Bun;

// Mock the Bun.file function
const mockArrayBuffer = new ArrayBuffer(8);
spyOn(Bun, "file").mockReturnValue({
  arrayBuffer: mock().mockResolvedValue(mockArrayBuffer),
} as unknown as BunFile);

// Import SileroV5Model after setting up the mocks
const SileroV5Model = require("../../../src/utils/models/silero").SileroV5Model;

describe("SileroV5Model", () => {
  beforeEach(() => {
    // Reset all mocks before each test
    mock.restore();
    // Reset the session and semaphore
    SileroV5Model["session"] = null;
    SileroV5Model["semaphore"] = null;
    // Reset the mockCreate function
    mockCreate.mockClear();
    // Reset the Bun.file mock
    mock(Bun.file).mockReturnValue({
      arrayBuffer: mock().mockResolvedValue(new ArrayBuffer(8)),
    } as unknown as BunFile);
  });

  afterEach(() => {
    mock.restore();
  });

  describe("getSession", () => {
    test("should return the existing session if it exists", async () => {
      const mockSession = {
        run: mockRun,
      };
      mockCreate.mockResolvedValueOnce(mockSession);

      const session1 = await SileroV5Model.getSession();
      mockCreate.mockClear(); // Clear the mock to ensure it's not called again
      const session2 = await SileroV5Model.getSession();

      expect(session1).toEqual(session2);
      expect(mockCreate).toHaveBeenCalledTimes(0);
    });
  });

  describe("getSessionSemaphore", () => {
    test("should create a new semaphore if none exists", () => {
      // Reset the semaphore
      SileroV5Model["semaphore"] = null;

      const semaphore = SileroV5Model.getSessionSemaphore();

      expect(semaphore).toBeDefined();
      expect(typeof semaphore.acquire).toBe("function");
      expect(typeof semaphore.release).toBe("function");
    });

    test("should return the existing semaphore if it exists", () => {
      const semaphore1 = SileroV5Model.getSessionSemaphore();
      const semaphore2 = SileroV5Model.getSessionSemaphore();

      expect(semaphore1).toBe(semaphore2);
    });
  });
});
