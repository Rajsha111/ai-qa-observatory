/**
 * CONTEXT PRECISION TEST
 * ─────────────────────────────────────────────────────────────────────────────
 * Failure mode: The top-K retrieved chunks are irrelevant to the query.
 *
 * Even if your LLM is perfect, it cannot give a good answer from irrelevant
 * chunks. This is the "garbage in, garbage out" problem for RAG.
 *
 * ━━━ WHAT IS CONTEXT PRECISION? ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 *
 * Context Precision = (number of relevant chunks in top-K) / K
 *
 * Example: top-3 retrieval where 2 chunks are relevant → precision = 2/3 = 0.67
 *
 * Why not require 3/3?
 *   - Top-1 and top-2 should almost always be relevant
 *   - Top-3 can sometimes be a marginal match
 *   - Requiring 100% precision is too strict for fuzzy semantic search
 *
 * ━━━ HOW THIS TEST WORKS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 *
 *   1. Call /query with topK=3 → get retrieved chunks
 *   2. Ask Claude to score each chunk's relevance (0.0–1.0)
 *   3. Assert: at least MIN_RELEVANT_CHUNKS chunks score >= RELEVANCE_THRESHOLD
 *
 * Thresholds are configured in config/rag.config.json.
 *
 * ━━━ HOW TO RUN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 *
 *   cd tests/rag/service && python server.py
 *   npx vitest tests/rag/precision.test.ts --reporter=verbose
 */

import { describe, it, expect, beforeAll } from "vitest";
import Anthropic from "@anthropic-ai/sdk";
import dotenv from "dotenv";
import type { RAGQueryResponse, ReterivedChunk } from "../../src/shared/types";
import { ingestTestDocuments } from "./fixtures";
import { loadRagConfig, loadTestCases, loadPrompt } from "./lib/config-loader";

dotenv.config();

const cfg       = loadRagConfig();
const testCases = loadTestCases();
const RAG_SERVICE         = cfg.service.ragServiceUrl;
const MIN_RELEVANT_CHUNKS = cfg.thresholds.minRelevantChunks;
const RELEVANCE_THRESHOLD = cfg.thresholds.chunkRelevance;

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

beforeAll(async () => {
  await ingestTestDocuments(RAG_SERVICE);
}, cfg.timeouts.setup);

// ── Chunk relevance scorer ─────────────────────────────────────────────────────
/**
 * Score a single chunk's relevance to the query.
 *
 * We ask for a REASON too — when a test fails, you need to know WHY
 * the chunk was irrelevant, not just that it was. This is the difference
 * between a test that catches bugs and one that just reports them.
 */
async function scoreChunkRelevance(
  query: string,
  chunkContent: string
): Promise<{ score: number; reason: string }> {
  const prompt = loadPrompt("chunk-relevance", { query, chunk: chunkContent });

  const response = await anthropic.messages.create({
    model:      cfg.llm.judge.model,
    max_tokens: 256,
    messages:   [{ role: "user", content: prompt }],
  });

  const text =
    response.content[0].type === "text" ? response.content[0].text : "{}";
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (!jsonMatch) return { score: 0, reason: "parse error" };

  const result = JSON.parse(jsonMatch[0]);
  return { score: result.score ?? 0, reason: result.reason ?? "" };
}

// ── Tests ─────────────────────────────────────────────────────────────────────
describe("Context Precision — Retrieval Quality", () => {
  for (const { query, description } of testCases.precisionQueries) {
    it(`top-${cfg.retrieval.defaultTopK} chunks for: ${description}`, async () => {
      const res = await fetch(`${RAG_SERVICE}/query`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ query, topK: cfg.retrieval.defaultTopK }),
      });
      const data = (await res.json()) as RAGQueryResponse;

      // Score each retrieved chunk in parallel
      const scored = await Promise.all(
        data.retrievedChunks.map(async (chunk: ReterivedChunk, i: number) => {
          const result = await scoreChunkRelevance(query, chunk.content);
          console.log(
            `\nChunk ${i + 1} (source: ${chunk.source}, similarity: ${chunk.score.toFixed(3)})`
          );
          console.log(`  Relevance score: ${result.score.toFixed(2)}`);
          console.log(`  Reason: ${result.reason}`);
          return result.score;
        })
      );

      const relevantCount    = scored.filter((s) => s >= RELEVANCE_THRESHOLD).length;
      const contextPrecision = relevantCount / data.retrievedChunks.length;

      console.log(
        `\nContext Precision: ${relevantCount}/${data.retrievedChunks.length} = ${contextPrecision.toFixed(2)}`
      );
      console.log(
        `Required: at least ${MIN_RELEVANT_CHUNKS}/${data.retrievedChunks.length}`
      );

      expect(relevantCount).toBeGreaterThanOrEqual(MIN_RELEVANT_CHUNKS);
    }, cfg.timeouts.ragas);
  }
});
