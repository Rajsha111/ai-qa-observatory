import { describe, it, expect, vi } from 'vitest';
import { runAgent } from '../../src/agent/runner';
import type { ToolResult } from '../../src/agent/tools';

describe('Loop Guard — agent halts at max iterations', () => {

    it('never exceeds 5 tool calls even with an ambiguous change', async () => {
      const callCount = { e2e: 0, eval: 0, rag: 0 };
      const mock = (name: keyof typeof callCount): ToolResult => {
        callCount[name]++;
        return { tool: name, exitCode: 1, output: 'tests failed, retry needed',
  passed: false };
      };

      const run = await runAgent(
        'Complete system overhaul — everything changed, all tests needed repeatedly',
        {
          toolOverrides: {
            run_e2e_tests:  () => mock('e2e'),
            run_eval_tests: () => mock('eval'),
            run_rag_tests:  () => mock('rag'),
          },
        }
      );

      const totalCalls = callCount.e2e + callCount.eval + callCount.rag;
      expect(totalCalls).toBeLessThanOrEqual(5);          // hard cap
      expect(run.steps.length).toBeLessThanOrEqual(6);    // steps include the final thought
      expect(run.finalAnswer).toBeDefined();               // partial report, not crash
    }, 60_000);

  });