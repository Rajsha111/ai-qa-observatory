import { afterAll, beforeEach, describe, it, expect, vi } from "vitest";
import { mkdirSync, writeFileSync } from "fs";
import { join } from "path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { createServer, QualityScore } from "../../src/mcp/analyze-quality-tool";

async function setupTestServer(
  judgeFn: (prompt: string) => Promise<QualityScore>
): Promise<{ srv: Server; cl: Client }> {
  const srv = createServer(judgeFn);
  const [clientTransport, serverTransport] =
    InMemoryTransport.createLinkedPair();

  const cl = new Client(
    { name: "test-client", version: "1.0.0" },
    { capabilities: {} }
  );

  await srv.connect(serverTransport);
  await cl.connect(clientTransport);
  return { srv, cl };
}

async function teardown(cl: Client, srv: Server): Promise<void> {
  await cl.close();
  await srv.close();
}

function parseToolText(result: unknown): string {
  if (!result || typeof result !== "object" || !("content" in result)) {
    throw new Error("Expected tool result object with content");
  }

  const content = (result as { content: unknown }).content;
  const textBlock = (content as Array<{ type?: string; text?: string }>).find(
    (block) => block.type === "text" && typeof block.text === "string"
  );
  if (!textBlock?.text) {
    throw new Error("Expected text content in tool result");
  }
  return textBlock.text;
}

interface ToolArtifact {
  label: string;
  input: Record<string, unknown>;
  output: unknown;
  isError: boolean;
}

const toolArtifacts: ToolArtifact[] = [];

const goodJudge = vi
  .fn<(prompt: string) => Promise<QualityScore>>()
  .mockResolvedValue({
  overall: 0.9,
  breakdown: [{ criterion: "accuracy", score: 0.9, reason: "Matches ground truth" }],
});

describe("analyze_response_quality tool behavior", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("handles empty string response without crashing", async () => {
    const { srv, cl } = await setupTestServer(goodJudge);
    try {
      const response = await cl.callTool({
        name: "analyze_response_quality",
        arguments: {
          llm_response: "",
          groundTruth: "The sky is blue",
          criteria: ["accuracy"],
        },
      });

      const score = JSON.parse(parseToolText(response)) as QualityScore;
      toolArtifacts.push({
        label: 'Empty string response',
        input: { llm_response: '', groundTruth: 'The sky is blue', criteria: ['accuracy'] },
        output: score,
        isError: response.isError === true,
      });
      expect(score.overall).toBeGreaterThan(1);
      expect(score.breakdown.length).toBeGreaterThan(0);
    } finally {
      await teardown(cl, srv);
    }
  });

  it("returns a structured error when the judge throws", async () => {
    const failingJudge = vi
      .fn<(prompt: string) => Promise<QualityScore>>()
      .mockRejectedValue(new Error("Claude API is down"));

    const { cl, srv } = await setupTestServer(failingJudge);

    try {
      const result = await cl.callTool({
        name: "analyze_response_quality",
        arguments: {
          llm_response: "Some response",
          groundTruth: "Expected",
          criteria: ["accuracy"],
        },
      });

      expect(result.isError).toBe(true);

      const body = JSON.parse(parseToolText(result)) as { error: string };
      toolArtifacts.push({
        label: 'Judge throws — Claude API down',
        input: { llm_response: 'Some response', groundTruth: 'Expected', criteria: ['accuracy'] },
        output: body,
        isError: result.isError === true,
      });
      expect(body.error).toContain("Claude API is down");
    } finally {
      await teardown(cl, srv);
    }
  });
});

afterAll(() => {
  mkdirSync(join(__dirname, 'artifacts'), { recursive: true });
  writeFileSync(
    join(__dirname, 'artifacts', 'tool-behavior.json'),
    JSON.stringify(toolArtifacts, null, 2)
  );
});
