import Anthropic from "@anthropic-ai/sdk";
import dotenv from "dotenv";
import { v4 as uuidv4 } from "uuid";
import type { AgentRun, AgentStep } from "@shared/types";
import { TOOL_MAP, ToolName } from "./tools";


dotenv.config();

const MAX_TOOL_CALLS = 5;

const TOOL_DEFINITIONS: Anthropic.Tool[] = [
  {
    name: "run_e2e_tests",
    description:
      "Run playwright e2e tests against the chat UI. Use when a change affects the API contract, response format, or UI behavior",
    input_schema: {
      type: "object",
      properties: {
        reason: {
          type: "string",
          description: "Why you are running this suite",
        },
      },
      required: [],
    },
  },
  {
    name: "run_eval_tests",
    description:
      "Run prompt evaluation tests (promptfoo + drift detector). Use when a change affects the system prompt, model, or output quality.",
    input_schema: { type: "object", properties: { reason: { type: "string" } }, required: [] },
  },
  {
    name: "run_rag_tests",
    description:
      "Run RAG validation tests (faithfulness, precision, negative, retrieval, stale-data). Use when a change affects the knowledge base, retrieval pipeline, or RAG prompt.",
    input_schema: { type: "object", properties: { reason: { type: "string" } }, required: [] },
  },
];

export interface RunAgentOptions{
    toolOverrides?: Partial<typeof TOOL_MAP>;
    messageCreate?: (params: Anthropic.MessageCreateParams) => Promise<Anthropic.Message>;
    maxToolCalls?: number;
}

export async function runAgent(
  changeDescription: string,
  options: RunAgentOptions = {}
): Promise<AgentRun> {
  const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
  const start = Date.now();
  const runId = uuidv4();
  const steps: AgentStep[] = [];
  const toolMap = { ...TOOL_MAP, ...options.toolOverrides };
  const systemPrompt = `You are a QA agent. You receive a description of a code
change and decide which test suites to run.
Available test suites: E2E (UI/API contract), Eval (prompt quality), RAG
(knowledge base pipeline).
Run only the suites relevant to the change. After running tests, report results
and stop.
Do not run the same suite twice. Stop after 5 tool calls maximum.`;
  const messages: Anthropic.MessageParam[] = [
    {
      role: "user",
      content: `Change description: ${changeDescription}\n\nWhich test suites should run, and why?`,
    },
  ];

    const maxToolCalls = options.maxToolCalls ?? MAX_TOOL_CALLS;
    const createMessage = options.messageCreate ?? ((params: Anthropic.MessageCreateParams) =>
      client.messages.create(params)
    );

    let iteration = 0;
    let toolCallCount = 0;
    let finalAnswer: string | undefined;
    while (toolCallCount < maxToolCalls) {
      iteration++;

      const response = await createMessage({
        model: 'claude-haiku-4-5-20251001',  // cheap + fast for orchestration
        max_tokens: 1024,
        system: systemPrompt,
        tools: TOOL_DEFINITIONS,
        messages,
      });
       // Extract the thought (any text block before a tool call)
      const textBlock = response.content.find(b => b.type === 'text');
      const toolUseBlocks = response.content.filter((b) => b.type === 'tool_use');
      console.log(`Iteration ${iteration} thought: ${textBlock?.type === 'text' ? textBlock.text : '[no text]'} ` +
  `| tool calls: ${toolUseBlocks.length}`);

      if (toolUseBlocks.length === 0) {
        // No tool call — Claude is done reasoning
        finalAnswer = textBlock?.type === 'text' ? textBlock.text : 'Agent completed with no final text.';
        console.log(`Iteration ${iteration} final answer: ${finalAnswer}`);
        steps.push({ stepIndex: steps.length + 1, thought: finalAnswer });
        break;
        }

      messages.push({ role: 'assistant', content: response.content });
      const toolResults: Anthropic.ToolResultBlockParam[] = [];

      for (const toolUseBlock of toolUseBlocks) {
        if (toolCallCount >= MAX_TOOL_CALLS) {
          break;
        }

        const toolName = toolUseBlock.name as ToolName;
        const toolFn = toolMap[toolName];
        const step: AgentStep = {
          stepIndex: steps.length + 1,
          thought: textBlock?.type === 'text' ? textBlock.text : `Calling ${toolName}`,
          toolCalled: toolName,
          toolInput: JSON.stringify(toolUseBlock.input),
        };

        let toolOutput: string;
        try {
          if (!toolFn) {
            throw new Error(`Unknown tool: ${toolName}`);
          }
          const result = toolFn();
          toolOutput = JSON.stringify({
            passed: result.passed,
            exitCode: result.exitCode,
            output: result.output.slice(0, 500),
          });
          step.observation = result.passed ? 'Tests passed' : 'Tests failed — see output';
        } catch (err) {
          toolOutput = JSON.stringify({ error: (err as Error).message });
          step.observation = `Tool error: ${(err as Error).message}`;
        }

        step.toolOutput = toolOutput;
        steps.push(step);
        toolCallCount++;
        toolResults.push({
          type: 'tool_result',
          tool_use_id: toolUseBlock.id,
          content: toolOutput,
        });
      }

      if (toolResults.length > 0) {
        messages.push({ role: 'user', content: toolResults });
      }

      // If stop_reason is end_turn (not tool_use), Claude is done
      if (response.stop_reason === 'end_turn') {
        finalAnswer = textBlock?.type === 'text' ? textBlock.text : 'Done.';
        break;
      }
    }
     // Loop guard: if we hit the tool call cap, write a partial report
    if (toolCallCount >= maxToolCalls && !finalAnswer) {
      finalAnswer = `Max tool calls (${maxToolCalls}) reached. Partial report:
  ${steps.length} steps completed.`;
      steps.push({ stepIndex: steps.length + 1, thought: finalAnswer });
    }

    return {
      runId,
      goal: changeDescription,
      steps,
      finalAnswer,
      totalLatencyMs: Date.now() - start,
      success: steps.some(s => s.toolCalled !== undefined),
    };
  }
