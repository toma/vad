import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const DEFAULT_CACHE_DIR = path.join(os.homedir(), ".cache", "vad-models");
const DEFAULT_REVISION = "main";

function cacheDir(): string {
  return process.env.MODEL_CACHE_DIR || DEFAULT_CACHE_DIR;
}

function cacheKey(repo: string, file: string, revision: string): string {
  return `${repo}/${revision}/${file}`.replace(/[^a-zA-Z0-9._-]/g, "_");
}

export interface HuggingFaceFile {
  /** HuggingFace repo id, e.g. "onnx-community/silero-vad". */
  repo: string;
  /** File path within the repo, e.g. "onnx/model.onnx". */
  file: string;
  /** Git revision/branch/tag. Defaults to "main". */
  revision?: string;
}

/**
 * Download a file from a HuggingFace repo into the local cache and return its
 * absolute path. Re-downloads only when the file is missing from the cache.
 */
export async function getModelPath(ref: HuggingFaceFile): Promise<string> {
  const revision = ref.revision ?? DEFAULT_REVISION;
  const dir = cacheDir();
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  const localPath = path.join(dir, cacheKey(ref.repo, ref.file, revision));
  if (fs.existsSync(localPath) && fs.statSync(localPath).size > 0) {
    return localPath;
  }

  const url = `https://huggingface.co/${ref.repo}/resolve/${revision}/${ref.file}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Failed to download ${ref.repo}/${ref.file} (${revision}): ${response.status} ${response.statusText}`,
    );
  }

  // Write to a temp file first, then rename, so an interrupted download never
  // leaves a corrupt file in the cache.
  const bytes = await response.arrayBuffer();
  const tmpPath = `${localPath}.${process.pid}.part`;
  await Bun.write(tmpPath, bytes);
  fs.renameSync(tmpPath, localPath);
  return localPath;
}
