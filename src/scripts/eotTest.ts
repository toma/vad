import fs from "node:fs";
import path from "node:path";
import type { ChatMessage } from "~/models/chat";
import { endOfTurnHandler } from "~/services/endOfTurn";
import { pmap } from "~/utils/async";
import { initializeEndOfTurn } from "~/utils/models/eot";

const ITERATIONS = 50;

const examples = JSON.parse(
  fs.readFileSync(path.join(__dirname, "eot_test.json"), "utf8"),
) as {
  label: string;
  result: boolean;
  context: ChatMessage[];
}[];

interface TestResult {
  label: string;
  expected: boolean;
  actual: boolean;
  probability: number;
  duration: number;
  correct: boolean;
}

const testExample = async (
  example: (typeof examples)[number],
): Promise<TestResult> => {
  const start = performance.now();
  try {
    const prob = (await endOfTurnHandler({ context: example.context }))
      .endOfTurnProbability;
    const actual = prob > 0.5;
    const correct = example.result === actual;
    const duration = performance.now() - start;

    return {
      label: example.label,
      expected: example.result,
      actual,
      probability: prob,
      duration,
      correct,
    };
  } catch (error) {
    const duration = performance.now() - start;
    console.error(`Error testing ${example.label}:`, error);
    return {
      label: example.label,
      expected: example.result,
      actual: false,
      probability: 0,
      duration,
      correct: false,
    };
  }
};

const runTestIteration = async (iteration: number): Promise<TestResult[]> => {
  console.log(`\n🔄 Running iteration ${iteration + 1}/${ITERATIONS}...`);
  const results = await pmap(examples, testExample, 3);

  // Log results for this iteration
  const correctCount = results.filter((r) => r.correct).length;
  const accuracy = (correctCount / results.length) * 100;
  const avgDuration =
    results.reduce((sum, r) => sum + r.duration, 0) / results.length;

  console.log(
    `   Accuracy: ${correctCount}/${results.length} (${accuracy.toFixed(1)}%)`,
  );
  console.log(`   Avg Duration: ${avgDuration.toFixed(2)}ms`);

  return results;
};

const calculateStatistics = (allResults: TestResult[][]) => {
  const flatResults = allResults.flat();
  const totalTests = flatResults.length;
  const totalCorrect = flatResults.filter((r) => r.correct).length;
  const overallAccuracy = (totalCorrect / totalTests) * 100;

  // Calculate accuracy per example
  const exampleStats = new Map<
    string,
    { correct: number; total: number; avgProb: number; avgDuration: number }
  >();

  flatResults.forEach((result) => {
    const stats = exampleStats.get(result.label) || {
      correct: 0,
      total: 0,
      avgProb: 0,
      avgDuration: 0,
    };
    stats.correct += result.correct ? 1 : 0;
    stats.total += 1;
    stats.avgProb += result.probability;
    stats.avgDuration += result.duration;
    exampleStats.set(result.label, stats);
  });

  // Calculate averages
  exampleStats.forEach((stats) => {
    stats.avgProb /= stats.total;
    stats.avgDuration /= stats.total;
  });

  return {
    overallAccuracy,
    totalTests,
    totalCorrect,
    exampleStats,
    allResults,
  };
};

const main = async () => {
  console.log("🚀 Initializing End of Turn model...");
  await initializeEndOfTurn();

  console.log(
    `📊 Running ${examples.length} examples ${ITERATIONS} times each...`,
  );
  const startTime = performance.now();

  const allResults: TestResult[][] = [];

  for (let i = 0; i < ITERATIONS; i++) {
    const iterationResults = await runTestIteration(i);
    allResults.push(iterationResults);

    // Progress indicator every 100 iterations
    if ((i + 1) % 100 === 0) {
      const elapsed = performance.now() - startTime;
      const avgTimePerIteration = elapsed / (i + 1);
      const remainingIterations = ITERATIONS - (i + 1);
      const estimatedTimeRemaining = remainingIterations * avgTimePerIteration;
      console.log(
        `\n📈 Progress: ${i + 1}/${ITERATIONS} (${(((i + 1) / ITERATIONS) * 100).toFixed(1)}%)`,
      );
      console.log(
        `⏱️  Estimated time remaining: ${(estimatedTimeRemaining / 1000).toFixed(1)}s`,
      );
    }
  }

  const totalTime = performance.now() - startTime;
  const stats = calculateStatistics(allResults);

  // Final report
  console.log("\n" + "=".repeat(80));
  console.log("🎯 FINAL TEST RESULTS");
  console.log("=".repeat(80));
  console.log(
    `📊 Overall Accuracy: ${stats.totalCorrect}/${stats.totalTests} (${stats.overallAccuracy.toFixed(2)}%)`,
  );
  console.log(`⏱️  Total Time: ${(totalTime / 1000).toFixed(2)}s`);
  console.log(
    `🔄 Average Time per Iteration: ${(totalTime / ITERATIONS).toFixed(2)}ms`,
  );
  console.log(
    `📈 Average Time per Test: ${(totalTime / stats.totalTests).toFixed(2)}ms`,
  );

  console.log("\n📋 Per-Example Statistics:");
  console.log("-".repeat(80));

  const sortedExamples = Array.from(stats.exampleStats.entries()).sort(
    (a, b) => b[1].correct / b[1].total - a[1].correct / a[1].total,
  );

  sortedExamples.forEach(([label, stats]) => {
    const accuracy = (stats.correct / stats.total) * 100;
    const status = accuracy >= 90 ? "🟢" : accuracy >= 70 ? "🟡" : "🔴";
    console.log(
      `${status} ${label}: ${stats.correct}/${stats.total} (${accuracy.toFixed(1)}%) | Avg Prob: ${stats.avgProb.toFixed(3)} | Avg Time: ${stats.avgDuration.toFixed(2)}ms`,
    );
  });

  // Save detailed results to file
  const resultsFile = path.join(
    __dirname,
    `eot_test_results_${Date.now()}.json`,
  );
  fs.writeFileSync(
    resultsFile,
    JSON.stringify(
      {
        summary: {
          overallAccuracy: stats.overallAccuracy,
          totalTests: stats.totalTests,
          totalCorrect: stats.totalCorrect,
          totalTime: totalTime,
          iterations: ITERATIONS,
        },
        exampleStats: Object.fromEntries(stats.exampleStats),
        allResults: stats.allResults,
      },
      null,
      2,
    ),
  );

  console.log(`\n💾 Detailed results saved to: ${resultsFile}`);
};

main().catch(console.error);
