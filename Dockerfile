# syntax=docker/dockerfile:1
FROM oven/bun:1.3.13-slim AS base
WORKDIR /app

COPY package.json bun.lock ./
RUN --mount=type=cache,target=/root/.bun/install/cache,sharing=locked \
    bun install --frozen-lockfile

COPY . .

ENV PORT=8086
EXPOSE 8086

CMD ["bun", "src/index.ts"]
