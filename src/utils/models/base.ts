import { Semaphore } from "async-mutex";
import { InferenceSession } from "onnxruntime-node";
import { onTerminate } from "~/utils/common";
import { getModelPath, type HuggingFaceFile } from "~/utils/hf";

interface BaseOnnxModelParams {
  queueDepth?: number;
}

/**
 * Wraps an async-mutex Semaphore with optional bounded queue depth.
 * When queueDepth is set, callers can check isQueueFull() before acquiring
 * to shed load instead of queuing unboundedly.
 */
export class BoundedSemaphore {
  private readonly _semaphore: Semaphore;
  private readonly _maxQueueDepth: number | null;
  private _pending = 0;

  constructor(permits: number, maxQueueDepth?: number) {
    this._semaphore = new Semaphore(permits);
    this._maxQueueDepth = maxQueueDepth ?? null;
  }

  public isQueueFull(): boolean {
    return this._maxQueueDepth !== null && this._pending >= this._maxQueueDepth;
  }

  public async acquire(): Promise<void> {
    this._pending++;
    try {
      await this._semaphore.acquire();
    } finally {
      this._pending--;
    }
  }

  public release(): void {
    this._semaphore.release();
  }
}

/**
 * Base class for ONNX models whose weights are fetched from HuggingFace.
 * Lazily loads a single shared InferenceSession behind a semaphore.
 */
export class BaseOnnxModel<TInput = any, TOutput = any> {
  private static globalSemaphore: Semaphore = new Semaphore(1);

  private _session: BaseOnnxModelSession<TInput, TOutput> | null;
  private _semaphore: BoundedSemaphore | null;
  private _queueDepth?: number;
  private _initializing: Promise<
    BaseOnnxModelSession<TInput, TOutput>
  > | null;

  constructor(
    private readonly modelRef: HuggingFaceFile,
    params?: BaseOnnxModelParams,
  ) {
    this._session = null;
    this._semaphore = null;
    this._queueDepth = params?.queueDepth;
    this._initializing = null;
  }

  /**
   * Get a singleton instance of the model.
   * @returns A singleton instance of the model.
   */
  public async getSession(): Promise<
    BaseOnnxModelSession<TInput, TOutput>
  > {
    // If session already exists and is not released, return it
    if (this._session && !this._session.released) {
      return this._session;
    }

    // Use double-checked locking to prevent race conditions
    // Acquire global semaphore first to ensure thread-safe initialization check
    await BaseOnnxModel.globalSemaphore.acquire();
    try {
      // Check again after acquiring the global lock
      if (this._session && !this._session.released) {
        return this._session;
      }

      // If initialization is already in progress, wait for it
      if (this._initializing) {
        return await this._initializing;
      }

      // Start initialization
      this._initializing = this._initializeSession();
      try {
        this._session = await this._initializing;
        return this._session;
      } finally {
        this._initializing = null;
      }
    } finally {
      BaseOnnxModel.globalSemaphore.release();
    }
  }

  public getSessionSemaphore(): BoundedSemaphore {
    if (!this._semaphore) {
      this._semaphore = new BoundedSemaphore(1, this._queueDepth);
    }
    return this._semaphore;
  }

  private async _initializeSession(): Promise<
    BaseOnnxModelSession<TInput, TOutput>
  > {
    // Acquire instance-specific semaphore for the actual initialization work
    await this.getSessionSemaphore().acquire();
    try {
      const modelPath = await getModelPath(this.modelRef);
      const session = await InferenceSession.create(modelPath, {
        graphOptimizationLevel: "all",
        enableCpuMemArena: false,
        enableMemPattern: false,
        executionMode: "sequential", // ML models have to be run sequentially
      });
      const modelSession = new BaseOnnxModelSession(session);

      onTerminate(async () => {
        if (!modelSession.released) {
          await modelSession.release();
        }
      });

      return modelSession;
    } catch (error) {
      console.error(
        `Failed to initialize model ${this.modelRef.repo}/${this.modelRef.file}:`,
        error,
      );
      throw error;
    } finally {
      this.getSessionSemaphore().release();
    }
  }
}

export class BaseOnnxModelSession<TInput = any, TOutput = any> {
  private _released: boolean;

  constructor(private readonly session: InferenceSession) {
    this._released = false;
  }

  public get released(): boolean {
    return this._released;
  }

  public run(input: TInput): Promise<TOutput> {
    if (this.released) {
      throw new Error("Session has been released");
    }
    return this.session.run(input as any) as Promise<TOutput>;
  }

  public async release(): Promise<void> {
    if (this.released) {
      return;
    }
    try {
      await this.session.release();
    } catch (error) {
      console.error("Error releasing session:", error);
    } finally {
      this._released = true;
    }
  }
}
