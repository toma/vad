import { Tensor } from "onnxruntime-node";
import type {
  VADEndOfTurnInputType,
  VADEndOfTurnOutputType,
} from "~/models/vad";
import { softmax } from "~/utils/math";
import { EndOfTurnModel, EndOfTurnTokenizer } from "~/utils/models/eot";

const EOT_ABORTED_DEFAULT: VADEndOfTurnOutputType = {
  endOfTurnProbability: 0.5,
};

/**
 * Handles detection of whether a user has completed their turn in a conversation.
 *
 * @param input The input containing chat context for end-of-turn detection
 * @returns Object containing the probability that the user has finished their turn
 */
export async function endOfTurnHandler(
  input: VADEndOfTurnInputType,
  signal?: AbortSignal,
): Promise<VADEndOfTurnOutputType> {
  let inputTensor: Tensor | undefined;
  let outputTensor: Tensor | undefined;
  let acquired: boolean = false;

  try {
    const semaphore = EndOfTurnModel.getSessionSemaphore();

    // Shed load instead of queuing unboundedly when the model is saturated
    if (semaphore.isQueueFull()) {
      return EOT_ABORTED_DEFAULT;
    }

    await semaphore.acquire();
    acquired = true;

    if (signal?.aborted) {
      return EOT_ABORTED_DEFAULT;
    }

    const session = await EndOfTurnModel.getSession();
    const eotTokenizer = await EndOfTurnTokenizer.getInstance();

    // Format and tokenize the context
    const context = eotTokenizer.formatContext(input.context);
    const tokens = eotTokenizer.encode(context);

    // Validate tokens array
    if (!tokens || tokens.length === 0) {
      throw new Error("Invalid tokens generated from context");
    }

    // Create the input tensor
    inputTensor = new Tensor("int64", tokens, [1, tokens.length]);

    if (signal?.aborted) {
      return EOT_ABORTED_DEFAULT;
    }

    // Run the session
    ({ logits: outputTensor } = await session.run({ input_ids: inputTensor! }));

    // Validate output tensor
    if (!outputTensor || !outputTensor.dims || outputTensor.dims.length < 2) {
      throw new Error("Invalid output tensor shape from model");
    }

    // Get the shape of the logits tensor
    const shape = outputTensor.dims;
    const vocabSize = shape[shape.length - 1];
    const logitsData = outputTensor.data as unknown as number[];

    // Validate logits data
    if (!logitsData || logitsData.length === 0) {
      throw new Error("Invalid logits data from model");
    }

    // Extract the last token's logits
    const lastTokenIndex = shape[1] - 1;
    const lastTokenLogitsStart = lastTokenIndex * vocabSize;

    // Validate array bounds
    if (
      lastTokenLogitsStart < 0 ||
      lastTokenLogitsStart + vocabSize > logitsData.length
    ) {
      throw new Error("Invalid logits array bounds");
    }

    const lastTokenLogits = logitsData.slice(
      lastTokenLogitsStart,
      lastTokenLogitsStart + vocabSize,
    );

    // Apply softmax only to the last token's logits
    const probs = softmax(lastTokenLogits);

    // Validate eotIndex
    if (eotTokenizer.eotIndex < 0 || eotTokenizer.eotIndex >= probs.length) {
      throw new Error("Invalid EOT index");
    }

    return { endOfTurnProbability: probs[eotTokenizer.eotIndex] };
  } catch (error) {
    console.error("Error in endOfTurnHandler:", error);

    // Return a safe default value
    return { endOfTurnProbability: 0.5 };
  } finally {
    // Dispose tensors BEFORE releasing the semaphore
    // This ensures no other thread can use the session while we're disposing tensors

    try {
      inputTensor?.dispose();
    } catch (disposeError) {
      console.error("Error disposing input tensor:", disposeError);
    }

    // Dispose output tensor last
    try {
      outputTensor?.dispose();
    } catch (disposeError) {
      console.error("Error disposing output tensor:", disposeError);
    }

    // Release semaphore LAST, after all tensor cleanup is complete
    if (acquired) {
      try {
        EndOfTurnModel.getSessionSemaphore().release();
      } catch (releaseError) {
        console.error("Error releasing semaphore:", releaseError);
      }
    }
  }
}
