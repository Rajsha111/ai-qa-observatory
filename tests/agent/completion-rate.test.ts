import { describe, it, expect } from 'vitest';
import { runAgent } from '../../src/agent/runner';
import type { ToolResult } from '../../src/agent/tools';

const OK: ToolResult = { tool: 'mock', exitCode: 0, output: 'passed', passed:
  true };

describe('Completion Rate — agent succeeds ≥ 80% of the time', () => {

    it('same task run 5x produces valid AgentRun at least 4/5 times', async () =>
        {
        const RUNS = 5;
        const MIN_SUCCESSES = 4;
        const results = await Promise.all(
            Array.from({ length: RUNS }, () =>
            runAgent('Updated /chat endpoint response format', {
                toolOverrides: {
                run_e2e_tests:  () => OK,
                run_eval_tests: () => OK,
                run_rag_tests:  () => OK,
                },
            }).then(run => ({
                success: typeof run.finalAnswer === 'string' &&
    run.finalAnswer.length > 0,
                runId: run.runId,
            })).catch(() => ({ success: false, runId: 'error' }))
            )
        );

        const successes = results.filter(r => r.success).length;
        console.log(`\nCompletion rate: ${successes}/${RUNS}`);
        results.forEach((r, i) => console.log(`  Run ${i + 1}: ${r.success ? 'pass'
    : 'FAIL'} (${r.runId})`));

        expect(successes).toBeGreaterThanOrEqual(MIN_SUCCESSES);
        }, 120_000);  // 5 parallel API calls

  });