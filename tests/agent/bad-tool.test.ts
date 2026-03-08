import { describe, it, expect, vi } from 'vitest';
import Anthropic from '@anthropic-ai/sdk';
import { runAgent } from '../../src/agent/runner';
import type { ToolResult } from '../../src/agent/tools';
import type { AgentStep } from '../../src/shared/types';

describe('Bad Tool — agent recovers from tool failure', () => {
  it('e2e tool throws → agent logs error and continues with eval/rag', async () => {
    const eval_ = vi.fn((): ToolResult => ({
      tool: 'run_eval_tests',
      exitCode: 0,
      output: 'passed',
      passed: true,
    }));
    const rag = vi.fn((): ToolResult => ({
      tool: 'run_rag_tests',
      exitCode: 0,
      output: 'passed',
      passed: true,
    }));
    let modelTurn = 0;
    const messageCreate = vi.fn(async (): Promise<Anthropic.Message> => {
      modelTurn++;
      if (modelTurn === 1) {
        return {
          id: 'msg_1',
          type: 'message',
          role: 'assistant',
          model: 'claude-haiku-4-5-20251001',
          content: [
            { type: 'text', text: 'Let me run both suites:' },
            { type: 'tool_use', id: 'toolu_1', name: 'run_e2e_tests', input: {} },
            { type: 'tool_use', id: 'toolu_2', name: 'run_eval_tests', input: {} },
          ],
          stop_reason: 'tool_use',
          stop_sequence: null,
          usage: { input_tokens: 10, output_tokens: 20 },
        };
      }
      return {
        id: 'msg_2',
        type: 'message',
        role: 'assistant',
        model: 'claude-haiku-4-5-20251001',
        content: [{ type: 'text', text: 'Recovered from tool failure and continued.' }],
        stop_reason: 'end_turn',
        stop_sequence: null,
        usage: { input_tokens: 15, output_tokens: 8 },
      };
    });

    const run = await runAgent(
      'Changed system prompt and updated KB documents',
      {
        messageCreate,
        toolOverrides: {
          run_e2e_tests: () => {
            throw new Error('E2E service unreachable: ECONNREFUSED');
          },
          run_eval_tests: eval_,
          run_rag_tests: rag,
        },
      }
    );

      // The step with the failing tool must exist and record the error
      const failedStep = run.steps.find((s: AgentStep) => s.toolCalled === 'run_e2e_tests');
      expect(failedStep).toBeDefined();
      expect(failedStep?.observation).toMatch(/error/i);

      // Agent must NOT crash — finalAnswer must exist
      expect(run.finalAnswer).toBeDefined();

      // Agent must still have tried other tools (no full cascade failure)
    const otherToolsCalled = run.steps.filter((s: AgentStep) =>
      s.toolCalled === 'run_eval_tests' ||
      s.toolCalled === 'run_rag_tests'
    );
    expect(otherToolsCalled.length).toBeGreaterThan(0);
    expect(messageCreate).toHaveBeenCalledTimes(2);
  }, 30_000);
});
