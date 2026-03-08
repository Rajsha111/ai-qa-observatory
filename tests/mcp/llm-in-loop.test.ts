import { afterAll, describe, it, expect } from 'vitest';
import { mkdirSync, writeFileSync } from 'fs';
import { join } from 'path';
  import Anthropic from '@anthropic-ai/sdk';
  import { Client } from '@modelcontextprotocol/sdk/client/index.js';
  import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
  import {
    createServer,
    QualityScore,
} from '../../src/mcp/analyze-quality-tool';

  const anthropic = new Anthropic();

  // Known-correct answer — judge should score high
  const GOOD = {
    response:
      'Photosynthesis is the process by which plants use sunlight, water, and CO2 to produce glucose and oxygen.',
    groundTruth:
      'Photosynthesis converts light energy into chemical energy stored as glucose, using water and CO2, releasing oxygen as a byproduct.',
  };

  // Factually wrong answer — judge should score low
  const BAD = {
    response: 'Photosynthesis is when plants absorb soil nutrients to grow.',
    groundTruth: GOOD.groundTruth,
  };

  interface LoopResult {
    score: QualityScore;
    toolUseInput: Record<string, unknown>;
    userMessage: string;
  }

  interface LoopArtifact {
    label: string;
    userMessage: string;
    toolUseInput: Record<string, unknown>;
    toolResult: QualityScore;
  }

  const loopArtifacts: LoopArtifact[] = [];

  async function runLLMInLoop(
    llm_response: string,
    ground_truth: string,
    criteria: string[]
  ): Promise<LoopResult> {
    // Fresh server with real Claude judge (this IS the integration test)
    const srv = createServer();
    const [clientTransport, serverTransport] =
      InMemoryTransport.createLinkedPair();
    const mcpClient = new Client(
      { name: 'claude-mcp-client', version: '1.0.0' },
      { capabilities: {} }
    );
    await srv.connect(serverTransport);
    await mcpClient.connect(clientTransport);

    // Get tool definitions from our MCP server
    const { tools: mcpTools } = await mcpClient.listTools();

    // Convert MCP tool format → Anthropic tool format
    const anthropicTools: Anthropic.Tool[] = mcpTools.map((t) => ({
      name: t.name,
      description: t.description ?? '',
      input_schema: t.inputSchema as Anthropic.Tool['input_schema'],
    }));

    const messages: Anthropic.MessageParam[] = [
      {
        role: 'user',
        content: `Use the analyze_response_quality tool to evaluate this response.
  Response: "${llm_response}"
  Ground truth: "${ground_truth}"
  Criteria: ${criteria.join(', ')}`,
      },
    ];

    let finalScore: QualityScore | null = null;
    let toolUseInput: Record<string, unknown> = {};

    // Agentic loop: Claude decides to call the tool, we execute it via MCP
    while (true) {
      const claudeResp = await anthropic.messages.create({
        model: 'claude-haiku-4-5-20251001',
        max_tokens: 1024,
        tools: anthropicTools,
        messages,
      });

      if (claudeResp.stop_reason === 'end_turn') break;

      if (claudeResp.stop_reason === 'tool_use') {
        const toolUse = claudeResp.content.find((b) => b.type === 'tool_use');
        if (!toolUse || toolUse.type !== 'tool_use') break;

        // Execute the tool Claude chose — via our real MCP server
        toolUseInput = toolUse.input as Record<string, unknown>;
        const toolResult = await mcpClient.callTool({
          name: toolUse.name,
          arguments: toolUseInput,
        });

        const resultText = (toolResult.content as [{ text: string }])[0].text;
        finalScore = JSON.parse(resultText) as QualityScore;

        messages.push({ role: 'assistant', content: claudeResp.content });
        messages.push({
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: toolUse.id,
              content: resultText,
            },
          ],
        });
      } else {
        break;
      }
    }

    await mcpClient.close();
    await srv.close();

    if (!finalScore) throw new Error('Claude never called the tool');
    const userMessage = messages[0].content as string;
    return { score: finalScore, toolUseInput, userMessage };
  }

  describe('analyze_response_quality — LLM in loop', () => {
    it(
      'scores a known-good response >= 0.8',
      async () => {
        const result = await runLLMInLoop(GOOD.response, GOOD.groundTruth, [
          'accuracy',
          'relevance',
        ]);
        loopArtifacts.push({ label: 'Known-good response', userMessage: result.userMessage, toolUseInput: result.toolUseInput, toolResult: result.score });
        expect(result.score.overall).toBeGreaterThanOrEqual(0.8);
        expect(result.score.breakdown.length).toBeGreaterThan(0);
      },
      30_000
    );

    it(
      'scores a known-bad response <= 0.4',
      async () => {
        const result = await runLLMInLoop(BAD.response, BAD.groundTruth, [
          'accuracy',
          'relevance',
        ]);
        loopArtifacts.push({ label: 'Known-bad response', userMessage: result.userMessage, toolUseInput: result.toolUseInput, toolResult: result.score });
        expect(result.score.overall).toBeLessThanOrEqual(0.1);
      },
      30_000
    );
  });

  afterAll(() => {
    mkdirSync(join(__dirname, 'artifacts'), { recursive: true });
    writeFileSync(
      join(__dirname, 'artifacts', 'llm-loop.json'),
      JSON.stringify(loopArtifacts, null, 2)
    );
  });