/** Numerically simple softmax over an array of logits. */
export function softmax(logits: number[]): number[] {
  const exp = logits.map((x) => Math.exp(x));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((x) => x / sum);
}
