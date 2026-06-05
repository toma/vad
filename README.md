# vad

> Real-time voice activity detection and end-of-turn detection, on [Bun](https://bun.sh).

[![npm](https://img.shields.io/npm/v/@toma.com/vad.svg)](https://www.npmjs.com/package/@toma.com/vad)
[![CI](https://github.com/toma-so/vad/actions/workflows/ci.yml/badge.svg)](https://github.com/toma-so/vad/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

A small, self-contained service exposing two capabilities over HTTP and WebSocket:

- **VAD** — stream audio and get per-input speaking / not-speaking state. Powered by [Silero VAD v5](https://github.com/snakers4/silero-vad) (ONNX).
- **End-of-Turn (EOT)** — given a short chat transcript, get the probability the user has finished their turn. Powered by the [LiveKit turn-detector](https://huggingface.co/livekit/turn-detector) (ONNX).

Model weights are downloaded from HuggingFace on first start and cached locally.

## Install

```bash
bun add @toma.com/vad
```

Requires [Bun](https://bun.sh) ≥ 1.3.

## Usage

Start the server — it downloads the models on first run, then listens on `:8086`:

```bash
bunx @toma.com/vad
curl localhost:8086/health   # -> "local"
```

From a clone, use `bun install && bun run dev` instead.

## API

### `GET /health`

Returns the current environment string (`local` by default), `200 OK`.

### `POST /eot` — end-of-turn detection

Request body:

```jsonc
{
  "context": [
    { "role": "user", "content": "what time do you open", "timestamp": 1717000000000 },
    { "role": "assistant", "content": "we open at 9am", "timestamp": 1717000001000 },
    { "role": "user", "content": "and on weekends", "timestamp": 1717000002000 }
  ]
}
```

`role` is one of `user` | `assistant` | `system`. Response:

```json
{ "endOfTurnProbability": 0.87 }
```

A value near `1` means the user has likely finished their turn; near `0` means they are likely to continue. Under heavy load the service sheds work and returns a neutral `0.5` (see `EOT_QUEUE_DEPTH`).

### `GET /` — real-time VAD (WebSocket)

Open a WebSocket to `/` with the stream configuration encoded as query parameters, then send raw audio frames as binary messages. The server replies with a JSON message per processed window.

**Query parameters**

| Param | Type | Default | Description |
| --- | --- | --- | --- |
| `format` | `mulaw_8000` \| `pcm_16000` | `mulaw_8000` | Encoding of the audio you send. |
| `frameDurationMs` | number | `80` | Analysis window size in ms (min 30). |
| `inputs[i][label]` | string | `default` | Name of an independent detection channel. |
| `inputs[i][positiveSpeechThreshold]` | number | `0.5` | Probability ≥ this counts as speech. |
| `inputs[i][negativeSpeechThreshold]` | number | `0.2` | Probability ≤ this counts as silence. |
| `inputs[i][lookbackFrames]` | number | `5` | Hysteresis window used to smooth state. |

You can define multiple `inputs[i]` with different thresholds to get several detectors over the same audio stream.

**Output message** (`VADOutput`):

```json
{
  "start": 1717000000000,
  "end": 1717000000123,
  "latency": 7,
  "outputs": [{ "label": "default", "isSpeaking": true }]
}
```

Send binary audio frames matching the declared `format`. On socket close the per-connection state is disposed automatically.

## Configuration

| Env var | Default | Description |
| --- | --- | --- |
| `PORT` | `8086` | Port to listen on. |
| `EOT_QUEUE_DEPTH` | `3` | Max queued EOT requests before the service sheds load (returns `0.5`). |
| `MODEL_CACHE_DIR` | `~/.cache/vad-models` | Where downloaded ONNX weights are cached. |
| `DEBUG` | `false` | Verbose logging. |
| `ENV` | `local` | Environment label returned by `/health`. |

## Docker

```bash
docker build -t vad .
docker run -p 8086:8086 vad
```

## How it works

- **`src/services/realtime.ts`** buffers incoming audio, slices it into fixed-duration frames, runs each frame through Silero, applies per-label thresholds with a lookback window, and emits speaking/not-speaking state.
- **`src/services/endOfTurn.ts`** folds and normalizes the chat context, applies the turn-detector chat template, runs the ONNX model, and softmaxes the final token's logits to read off the end-of-turn token probability.
- **`src/utils/models/base.ts`** lazily loads a single shared ONNX `InferenceSession` per model behind a semaphore (ONNX runtime is not thread-safe), with bounded-queue load shedding.
- **`src/utils/hf.ts`** downloads and caches model files from HuggingFace.

## Models & licenses

This project is just the serving layer — it doesn't bundle or redistribute any model weights. Both models are downloaded from HuggingFace at runtime and remain under their original licenses. All credit for the models goes to their authors.

| Model | Used for | Source | License |
| --- | --- | --- | --- |
| Silero VAD v5 | Voice activity detection | [`onnx-community/silero-vad`](https://huggingface.co/onnx-community/silero-vad) ([upstream](https://github.com/snakers4/silero-vad)) | [MIT](https://github.com/snakers4/silero-vad/blob/master/LICENSE) |
| LiveKit turn-detector | End-of-turn detection | [`livekit/turn-detector`](https://huggingface.co/livekit/turn-detector) | [Apache-2.0](https://huggingface.co/livekit/turn-detector) |

## Contributing

Issues and pull requests are welcome. Before opening a PR, make sure the checks pass:

```bash
bun install
bun run typecheck
bun run lint
bun test
```

Maintainers: [@anthonykrivonos](https://github.com/anthonykrivonos) (lead), [@ray-cj-huang](https://github.com/ray-cj-huang).

## License

[MIT](./LICENSE) © Toma
