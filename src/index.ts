#!/usr/bin/env bun
import type { Server } from "bun";
import { ZodError } from "zod";
import { VADEndOfTurnInputSchema, vadInputFromQueryString } from "~/models/vad";
import { endOfTurnHandler } from "~/services/endOfTurn";
import { RealtimeVAD } from "~/services/realtime";
import env from "~/utils/env";
import Logger from "~/utils/logger";
import { initializeEndOfTurn } from "~/utils/models/eot";
import { initializeSilero } from "~/utils/models/silero";

const logger = new Logger({ label: "vad" });

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function matches(req: Request, method: string, path: string): boolean {
  return req.method === method && new URL(req.url).pathname === path;
}

// Models are loaded sequentially to avoid ONNX runtime init races.
try {
  logger.log("🗣️ Loading VAD model (downloading from HuggingFace if needed)...");
  await initializeSilero();
  logger.log("🗣️✅ VAD model loaded");
} catch (err) {
  logger.error("🗣️🚨 Failed to load VAD model", err);
  process.exit(1);
}

try {
  logger.log("🙊 Loading End of Turn model...");
  await initializeEndOfTurn();
  logger.log("🙊✅ End of Turn model loaded");
} catch (err) {
  logger.error("🙊🚨 Failed to load End of Turn model", err);
  process.exit(1);
}

async function fetchHandler(
  req: Request,
  server: Server<RealtimeVAD>,
): Promise<Response | undefined> {
  // Healthcheck
  if (matches(req, "GET", "/health")) {
    return new Response(env.ENV, { status: 200 });
  }

  // End-of-turn detection
  if (matches(req, "POST", "/eot")) {
    try {
      const input = VADEndOfTurnInputSchema.parse(await req.json());
      const result = await endOfTurnHandler(input, req.signal);
      return json(result);
    } catch (error) {
      if (error instanceof ZodError) {
        return json(
          { error: "Invalid request body", issues: error.issues },
          400,
        );
      }
      logger.error("Error in EOT handler", error);
      return json({ error: "Internal server error" }, 500);
    }
  }

  // Real-time VAD WebSocket
  if (matches(req, "GET", "/")) {
    const input = vadInputFromQueryString(new URL(req.url).search);
    logger.log(`🗣️ New VAD client: ${JSON.stringify(input)}`);
    const upgraded = server.upgrade(req, { data: new RealtimeVAD(input) });
    if (upgraded) {
      return undefined;
    }
    return new Response("Expected a WebSocket upgrade request", { status: 426 });
  }

  logger.log(`⚠️ Not found: ${req.method} ${req.url}`);
  return new Response("Not Found", { status: 404 });
}

const server = Bun.serve<RealtimeVAD>({
  port: Number(env.PORT),
  fetch: fetchHandler,
  websocket: {
    message: async (ws, message) => {
      try {
        const res = await ws.data.onMessage(message as Buffer);
        if (res && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify(res));
        }
      } catch (error) {
        logger.error("Error in WebSocket message handler", error);
      }
    },
    close: async (ws) => {
      try {
        await ws.data.destroy();
      } catch (error) {
        logger.error("Error in WebSocket close handler", error);
      }
    },
    backpressureLimit: 1024 * 1024,
    closeOnBackpressureLimit: true,
  },
});

process.on("uncaughtException", (err) => {
  logger.error("Uncaught exception", err);
});
process.on("unhandledRejection", (err) => {
  logger.error("Unhandled rejection", err);
});

logger.log(`💻 Server is listening on port ${server.port}`);
