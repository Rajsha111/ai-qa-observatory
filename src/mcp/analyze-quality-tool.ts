import Anthropic from "@anthropic-ai/sdk";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import dotenv from "dotenv";

dotenv.config();

// Exported so schema.test.ts can assert against it
export const analyzeQualityInputSchema = {
  type: "object",
  properties: {
    llm_response: {
      type: "string",
      description: "The response from the LLM to analyze.",
    },
    groundTruth: {
      type: "string",
      description: "The ground truth answer to compare against.",
    },
    criteria: {
      type: "array",
      items: {
        type: "string",
      },
      description: "The criteria to use for analysis.",
    },
  },
  required: ["llm_response", "groundTruth", "criteria"],
};

export interface QualityScore {
  overall: number;
  breakdown: { criterion: string; score: number; reason: string }[];
}

export function buildJudgePrompt(
  llm_response: string,
  groundTruth: string,
  criteria: string[]
): string {
  return `You are an objective AI quality judge. Evaluate the LLM response against the ground truth
on each criterion.

LLM Response:
${llm_response}

Ground Truth:
${groundTruth}

Criteria: ${criteria.join(", ")}

Score each criterion 0.0-1.0. Return ONLY valid JSON in this exact format:
{
  "overall": <arithmetic mean of all scores>,
  "breakdown": [
    { "criterion": "<name>", "score": <0.0-1.0>, "reason": "<one sentence>" }
  ]
}`;
}

function extractJsonFromText(text: string): string {
  const trimmed = text.trim();
  if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
    return trimmed;
  }

  const fencedMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fencedMatch?.[1]) {
    return fencedMatch[1].trim();
  }

  const first = trimmed.indexOf("{");
  const last = trimmed.lastIndexOf("}");
  if (first !== -1 && last !== -1 && last > first) {
    return trimmed.slice(first, last + 1);
  }

  return trimmed;
}

function normalizeQualityScore(raw: unknown): QualityScore {
  const data = raw as {
    overall?: unknown;
    overall_score?: unknown;
    breakdown?: Array<{
      criterion?: unknown;
      criteria?: unknown;
      score?: unknown;
      reason?: unknown;
    }>;
  };

  const overallSource = data.overall ?? data.overall_score ?? 0;
  const overall =
    typeof overallSource === "number" ? overallSource : Number(overallSource);

  const breakdown = Array.isArray(data.breakdown)
    ? data.breakdown.map((item) => ({
        criterion: String(item.criterion ?? item.criteria ?? "unknown"),
        score:
          typeof item.score === "number" ? item.score : Number(item.score ?? 0),
        reason: String(item.reason ?? ""),
      }))
    : [];

  return { overall, breakdown };
}

export async function callClaudeJudge(prompt: string): Promise<QualityScore> {
  const client = new Anthropic();
  const message = await client.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 1000,
    messages: [
      {
        role: "user",
        content: prompt,
      },
    ],
  });

  const text = message.content[0]?.type === "text" ? message.content[0].text : "";
  const payload = extractJsonFromText(text);
  return normalizeQualityScore(JSON.parse(payload));
}

export function createServer(
  judgeFunction: (prompt: string) => Promise<QualityScore> = callClaudeJudge
): Server {
  const server = new Server(
    { name: "ai-qa-observatory", version: "1.0.0" },
    { capabilities: { tools: {} } }
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [
      {
        name: "analyze_response_quality",
        description:
          "Evaluates LLM response quality against ground truth using LLM-as-Judge",
        inputSchema: analyzeQualityInputSchema,
      },
    ],
  }));

  server.setRequestHandler(CallToolRequestSchema, async (req) => {
    if (req.params.name !== "analyze_response_quality") {
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({ error: `Unknown tool: ${req.params.name}` }),
          },
        ],
        isError: true,
      };
    }

    const { llm_response, groundTruth, criteria } = req.params.arguments as {
      llm_response: string;
      groundTruth: string;
      criteria: string[];
    };

    try {
      const prompt = buildJudgePrompt(llm_response, groundTruth, criteria);
      const score = await judgeFunction(prompt);
      return { content: [{ type: "text", text: JSON.stringify(score) }] };
    } catch (err) {
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({ error: (err as Error).message }),
          },
        ],
        isError: true,
      };
    }
  });

  return server;
}
