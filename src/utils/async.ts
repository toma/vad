/** Resolve after `ms` milliseconds. */
export function sleep(ms?: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Map over an (async) iterable with bounded concurrency, preserving order.
 * @param iterable - The items to map over.
 * @param mapper - Async mapper invoked with (element, index).
 * @param concurrency - Maximum number of mappers running at once.
 */
export async function pmap<TInput, TOutput>(
  iterable: Iterable<TInput> | AsyncIterable<TInput>,
  mapper: (element: TInput, index: number) => TOutput | Promise<TOutput>,
  concurrency: number = Number.POSITIVE_INFINITY,
): Promise<TOutput[]> {
  return new Promise((resolve, reject_) => {
    if (!(Symbol.iterator in iterable) && !(Symbol.asyncIterator in iterable)) {
      throw new TypeError(
        `Expected \`input\` to be either an \`Iterable\` or \`AsyncIterable\`, got (${typeof iterable})`,
      );
    }

    if (typeof mapper !== "function") {
      throw new TypeError("Mapper function is required");
    }

    if (
      !(
        (Number.isSafeInteger(concurrency) && concurrency >= 1) ||
        concurrency === Number.POSITIVE_INFINITY
      )
    ) {
      throw new TypeError(
        `Expected \`concurrency\` to be an integer from 1 and up or \`Infinity\`, got \`${concurrency}\` (${typeof concurrency})`,
      );
    }

    const result: TOutput[] = [];
    let isRejected = false;
    let isResolved = false;
    let isIterableDone = false;
    let resolvingCount = 0;
    let currentIndex = 0;
    const iterator =
      Symbol.asyncIterator in iterable
        ? (iterable as AsyncIterable<TInput>)[Symbol.asyncIterator]()
        : (iterable as Iterable<TInput>)[Symbol.iterator]();

    const reject = (reason: Error): void => {
      isRejected = true;
      isResolved = true;
      reject_(reason);
    };

    const next = async (): Promise<void> => {
      if (isResolved) {
        return;
      }

      const nextItem = await iterator.next();

      const index = currentIndex;
      currentIndex++;

      if (nextItem.done) {
        isIterableDone = true;

        if (resolvingCount === 0 && !isResolved) {
          isResolved = true;
          resolve(result);
        }

        return;
      }

      resolvingCount++;

      (async () => {
        try {
          const element = await nextItem.value;

          if (isResolved) {
            return;
          }

          result[index] = await mapper(element, index);

          resolvingCount--;
          await next();
        } catch (error) {
          reject(error as Error);
        }
      })();
    };

    (async () => {
      for (let index = 0; index < concurrency; index++) {
        try {
          await next();
        } catch (error) {
          reject(error as Error);
          break;
        }

        if (isIterableDone || isRejected) {
          break;
        }
      }
    })();
  });
}
