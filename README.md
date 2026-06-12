# vad

> Embeddable real-time voice activity detection and end-of-turn detection for Node and Bun.

[![npm](https://img.shields.io/npm/v/@toma.com/vad.svg)](https://www.npmjs.com/package/@toma.com/vad)
[![CI](https://github.com/toma-so/vad/actions/workflows/ci.yml/badge.svg)](https://github.com/toma-so/vad/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

A small, dependency-light TypeScript library that wraps two ONNX models:

- **VAD** — stream audio frames and get per-input speaking / not-speaking state. Powered by [Silero VAD v5](https://github.com/snakers4/silero-vad).
- **End-of-Turn (EOT)** — given a short chat transcript, get the probability the user has finished their turn. Powered by the [LiveKit turn-detector](https://huggingface.co/livekit/turn-detector).

The library **does not bundle model weights**. You point each detector at local ONNX files that you've downloaded ahead of time — no network access is required at runtime when the paths are local.

## Install

```bash
npm i @toma.com/vad
# or
bun add @toma.com/vad
```

Requires Node ≥ 18 or Bun ≥ 1.3.

## Get the model files

Download once from HuggingFace (or vendor them into your repo / container image):

```bash
# Silero VAD v5
curl -L -o silero_vad.onnx \
  https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx

# LiveKit turn-detector (model + tokenizer)
mkdir -p turn-detector
curl -L -o turn-detector/model_quantized.onnx \
  https://huggingface.co/livekit/turn-detector/resolve/main/model_quantized.onnx
curl -L -o turn-detector/tokenizer.json \
  https://huggingface.co/livekit/turn-detector/resolve/main/tokenizer.json
curl -L -o turn-detector/tokenizer_config.json \
  https://huggingface.co/livekit/turn-detector/resolve/main/tokenizer_config.json
curl -L -o turn-detector/special_tokens_map.json \
  https://huggingface.co/livekit/turn-detector/resolve/main/special_tokens_map.json
```

## Real-time VAD

```ts
import { RealtimeVAD } from "@toma.com/vad";

const vad = new RealtimeVAD({
  modelPath: "/abs/path/to/silero_vad.onnx",
  format: "pcm_16000",        // or "mulaw_8000"
  frameDurationMs: 80,
  inputs: [
    {
      label: "default",
      positiveSpeechThreshold: 0.5,
      negativeSpeechThreshold: 0.2,
      lookbackFrames: 5,
    },
  ],
});

// Feed raw PCM (or mu-law) frames as Buffers as they arrive.
const result = await vad.onMessage(audioBuffer);
if (result) {
  for (const out of result.outputs) {
    console.log(out.label, out.isSpeaking);
  }
}

await vad.destroy();
```

You can also drive the loop yourself with `push(buffer, timestamp)` followed by `await run()` if you want to decouple ingestion from inference.

Define multiple entries in `inputs` to get independent detectors over the same audio (e.g. a sensitive detector and a strict detector running in parallel).

### Output shape

```ts
{
  start: 1717000000000,   // earliest sample timestamp processed in this batch
  end:   1717000000123,   // wall-clock time when processing finished
  latency: 7,             // ms spent in inference
  outputs: [{ label: "default", isSpeaking: true }],
}
```

## End-of-turn detection

```ts
import { EndOfTurnDetector } from "@toma.com/vad";

const eot = new EndOfTurnDetector({
  modelPath: "/abs/path/to/turn-detector/model_quantized.onnx",
  tokenizerPath: "/abs/path/to/turn-detector", // directory with tokenizer.json
  queueDepth: 3,                                // optional load-shedding limit
});

const { endOfTurnProbability } = await eot.detect({
  context: [
    { role: "user",      content: "what time do you open", timestamp: 1717000000000 },
    { role: "assistant", content: "we open at 9am",        timestamp: 1717000001000 },
    { role: "user",      content: "and on weekends",       timestamp: 1717000002000 },
  ],
});

await eot.destroy();
```

A value near `1` means the user has likely finished their turn; near `0` means they are likely to continue. If `queueDepth` is set and the inference queue is saturated, `detect()` sheds load and returns a neutral `0.5` without blocking.

## API reference

### `new RealtimeVAD(params)`

| Param | Type | Default | Description |
| --- | --- | --- | --- |
| `modelPath` | `string` | — | **Required.** Absolute path to `silero_vad.onnx`. |
| `format` | `"mulaw_8000" \| "pcm_16000"` | `"mulaw_8000"` | Encoding of the buffers you'll feed in. |
| `frameDurationMs` | `number` | `80` | Analysis window in ms (min 30). |
| `inputs` | `RealtimeVADOptionsInput[]` | one default | One or more detectors with independent thresholds. |
| `verbose` | `boolean` | `false` | Log resolved options on startup. |

Methods:

- `onMessage(buffer: Buffer): Promise<VADOutputType | undefined>` — push audio + run inference in one call. Convenience entry-point. Resolves to `undefined` when there aren't enough buffered samples to fill a frame yet.
- `push(buffer: Buffer, originTimestamp: number): void` — append audio to the internal buffer without running inference.
- `run(): Promise<Array<{ label: string; isSpeaking: boolean; originTimestamp: number }>>` — slice all buffered audio into frames and emit per-label state.
- `destroy(): Promise<void>` — release the ONNX session and clear internal buffers.

### `new EndOfTurnDetector(params)`

| Param | Type | Description |
| --- | --- | --- |
| `modelPath` | `string` | Absolute path to the turn-detector ONNX file. |
| `tokenizerPath` | `string` | Path to a local directory containing `tokenizer.json` and `tokenizer_config.json`. A HuggingFace repo id also works but triggers a network fetch on first `detect()`. |
| `queueDepth` | `number?` | If set, calls past this depth return `{ endOfTurnProbability: 0.5 }` immediately. |

Methods:

- `detect(input: { context: ChatMessage[] }, signal?: AbortSignal): Promise<{ endOfTurnProbability: number }>` — returns the EOT probability. Returns a neutral `0.5` if `signal` aborts or if `queueDepth` is exceeded.
- `destroy(): Promise<void>` — release the ONNX session.

## How it works

- **`src/services/realtime.ts`** buffers incoming audio, slices it into fixed-duration frames, runs each frame through Silero, applies per-label thresholds with a lookback window, and emits speaking / not-speaking state.
- **`src/services/endOfTurn.ts`** folds and normalizes the chat context, applies the turn-detector chat template, runs the ONNX model, and softmaxes the final token's logits to read off the end-of-turn token probability.
- **`src/utils/models/base.ts`** lazily creates one shared `InferenceSession` per model behind a semaphore (onnxruntime sessions are not safe to run concurrently), with optional bounded-queue load shedding.

## Models & licenses

This library does not bundle or redistribute any model weights. You bring your own copy of each model and it remains under its original license. All credit for the models goes to their authors.

| Model | Used for | Source | License |
| --- | --- | --- | --- |
| Silero VAD v5 | Voice activity detection | [`onnx-community/silero-vad`](https://huggingface.co/onnx-community/silero-vad) ([upstream](https://github.com/snakers4/silero-vad)) | [MIT](https://github.com/snakers4/silero-vad/blob/master/LICENSE) |
| LiveKit turn-detector | End-of-turn detection | [`livekit/turn-detector`](https://huggingface.co/livekit/turn-detector) | [Apache-2.0](https://huggingface.co/livekit/turn-detector) |

## Contributing

```bash
bun install
bun run typecheck
bun run lint
bun test
bun run build
```

Maintainers: [@anthonykrivonos](https://github.com/anthonykrivonos) (lead), [@ray-cj-huang](https://github.com/ray-cj-huang).

## License

[MIT](./LICENSE) © Toma
