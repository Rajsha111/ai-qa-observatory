import { describe, it, expect, vi } from "vitest";
import Anthropic from "@anthropic-ai/sdk";
import { runAgent } from '../../src/agent/runner';
import type { ToolResult } from '../../src/agent/tools';

const OK: ToolResult = {
  tool: 'mock',
  exitCode: 0,
  output: 'all tests passed',
  passed: true,
};

function scriptedMessageCreate(toolNames: Array<'run_e2e_tests' | 'run_eval_tests' | 'run_rag_tests'>) {
  let turn = 0;
  return vi.fn(async (): Promise<Anthropic.Message> => {
    turn++;
    if (turn === 1) {
      return {
        id: 'msg_1',
        type: 'message',
        role: 'assistant',
        model: 'claude-haiku-4-5-20251001',
        content: [
          { type: 'text', text: 'Selecting relevant suites.' },
          ...toolNames.map((name, i) => ({ type: 'tool_use' as const, id: `toolu_${i + 1}`, name, input: {} })),
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
      content: [{ type: 'text', text: 'Done.' }],
      stop_reason: 'end_turn',
      stop_sequence: null,
      usage: { input_tokens: 12, output_tokens: 6 },
    };
  });
}

describe('API response format change -> calls e2e -> eval not rag', () => {
  it('api response format change -> calls', async () => {
    const e2e = vi.fn(() => OK);
    const eval_ = vi.fn(() => OK);
    const rag = vi.fn(() => OK);
    const messageCreate = scriptedMessageCreate(['run_e2e_tests', 'run_eval_tests']);

    await runAgent(
      'Changed /chat endpoint to wrap response in a data field: { data: {message } }',
      {
        messageCreate,
        toolOverrides: {
          run_e2e_tests: e2e,
          run_eval_tests: eval_,
          run_rag_tests: rag,
        },
      }
    );

    expect(e2e).toHaveBeenCalled();
    expect(eval_).toHaveBeenCalled();
    expect(rag).not.toHaveBeenCalled();
    expect(messageCreate).toHaveBeenCalledTimes(2);
  }, 30_000);

  it('Knowledge base document update → calls rag, NOT e2e', async () => {
    const e2e = vi.fn(() => OK);
    const eval_ = vi.fn(() => OK);
    const rag = vi.fn(() => OK);
    const messageCreate = scriptedMessageCreate(['run_rag_tests']);

    await runAgent(
      'Updated the Nexus AI QA policy documents in the knowledge base',
      {
        messageCreate,
        toolOverrides: {
          run_e2e_tests: e2e,
          run_eval_tests: eval_,
          run_rag_tests: rag,
        },
      }
    );

    expect(rag).toHaveBeenCalled(); // RAG KB changed
    expect(e2e).not.toHaveBeenCalled(); // UI unchanged
    expect(messageCreate).toHaveBeenCalledTimes(2);
  }, 30_000);
});
