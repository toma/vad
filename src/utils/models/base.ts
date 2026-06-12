import { Semaphore } from "async-mutex";
import { InferenceSession } from "onnxruntime-node";

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
 * Wraps an onnxruntime InferenceSession loaded from a local file path.
 * Lazily creates a single session per instance, guarded by a semaphore
 * because onnxruntime sessions are not safe to run concurrently.
 */
export class BaseOnnxModel<TInput = any, TOutput = any> {
  private _session: BaseOnnxModelSession<TInput, TOutput> | null;
  private _semaphore: BoundedSemaphore | null;
  private _queueDepth?: number;
  private _initializing: Promise<
    BaseOnnxModelSession<TInput, TOutput>
  > | null;

  constructor(
    private readonly modelPath: string,
    params?: BaseOnnxModelParams,
  ) {
    this._session = null;
    this._semaphore = null;
    this._queueDepth = params?.queueDepth;
    this._initializing = null;
  }

  public async getSession(): Promise<
    BaseOnnxModelSession<TInput, TOutput>
  > {
    if (this._session && !this._session.released) {
      return this._session;
    }

    if (this._initializing) {
      return await this._initializing;
    }

    this._initializing = this._initializeSession();
    try {
      this._session = await this._initializing;
      return this._session;
    } finally {
      this._initializing = null;
    }
  }

  public getSessionSemaphore(): BoundedSemaphore {
    if (!this._semaphore) {
      this._semaphore = new BoundedSemaphore(1, this._queueDepth);
    }
    return this._semaphore;
  }

  /**
   * Release the underlying session if one was created. Safe to call before
   * `getSession()` was ever invoked — never triggers initialization.
   */
  public async releaseSession(): Promise<void> {
    if (this._session && !this._session.released) {
      await this._session.release();
    }
    this._session = null;
  }

  private async _initializeSession(): Promise<
    BaseOnnxModelSession<TInput, TOutput>
  > {
    try {
      const session = await InferenceSession.create(this.modelPath, {
        graphOptimizationLevel: "all",
        enableCpuMemArena: false,
        enableMemPattern: false,
        executionMode: "sequential",
      });
      return new BaseOnnxModelSession(session);
    } catch (error) {
      console.error(`Failed to initialize model ${this.modelPath}:`, error);
      throw error;
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
