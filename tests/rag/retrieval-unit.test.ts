/**
 * RETRIEVAL UNIT TEST — Deterministic Retrieval Regression
 * ─────────────────────────────────────────────────────────────────────────────
 * Failure mode: A known query stops retrieving the expected document.
 *
 * Unlike LLM tests (probabilistic), retrieval should be deterministic:
 * the same query should ALWAYS return the same top chunk. No LLM needed here —
 * this is pure vector similarity, and it should be reproducible.
 *
 * ━━━ WHY THIS MATTERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 *
 * "We upgraded to text-embedding-3-large and now users can't find the
 *  compliance policy when they search for it."
 *
 * That's a real production regression. This test catches it in CI.
 * It should also run whenever you:
 *   - Swap embedding models (most common cause of retrieval regression)
 *   - Change chunk size or overlap (changes what gets embedded)
 *   - Modify similarity thresholds
 *   - Re-index with new or reorganised documents
 *
 * ━━━ HOW THIS TEST WORKS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 *
 *   1. Load query → expected source ID pairs from data/test-cases.json
 *   2. Call /query with topK=3
 *   3. Assert the expected document source appears in top-3 results
 *
 * This is called a "unit" test because you have complete knowledge of the
 * system: you know exactly what SHOULD be retrieved for each query.
 *
 * ━━━ HOW TO RUN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 *
 *   cd tests/rag/service && python server.py
 *   npx vitest tests/rag/retrieval-unit.test.ts --reporter=verbose
 */

import { describe, it, expect, beforeAll } from "vitest";
import dotenv from "dotenv";
import type { RAGQueryResponse } from "../../src/shared/types";
import { ingestTestDocuments } from "./fixtures";
import { loadRagConfig, loadTestCases } from "./lib/config-loader";

dotenv.config();

const cfg       = loadRagConfig();
const testCases = loadTestCases();
const RAG_SERVICE = cfg.service.ragServiceUrl;

beforeAll(async () => {
  await ingestTestDocuments(RAG_SERVICE);
}, cfg.timeouts.setup);

// ── Tests ─────────────────────────────────────────────────────────────────────
describe("Retrieval Unit Tests — Deterministic Regression Suite", () => {
  // Ground-truth expectations loaded from data/test-cases.json.
  // Each entry is a known-good mapping: query → the document that MUST appear.
  // The expectedSource matches the "id" field sent during /ingest.
  for (const { query, expectedSource, description } of testCases.retrievalExpectations) {
    it(`retrieves correct doc: ${description}`, async () => {
      const res = await fetch(`${RAG_SERVICE}/query`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ query, topK: cfg.retrieval.defaultTopK }),
      });
      const data = (await res.json()) as RAGQueryResponse;

      const sources    = data.retrievedChunks.map((c) => c.source);
      const withScores = data.retrievedChunks.map(
        (c) => `${c.source}(${c.score.toFixed(3)})`
      );

      console.log(`\nQuery:    "${query}"`);
      console.log(`Expected: ${expectedSource}`);
      console.log(`Got:      [${withScores.join(", ")}]`);
      console.log(
        `Found:    ${sources.some((s) => s.includes(expectedSource))}`
      );

      // The expected document must appear somewhere in the top-K results.
      // If this fails after changing your embedding model or chunk settings,
      // that pipeline change broke retrieval quality — fix before shipping.
      expect(
        sources.some((s) => s.includes(expectedSource)),
        `Expected "${expectedSource}" in top-${cfg.retrieval.defaultTopK}, got: [${sources.join(", ")}]`
      ).toBe(true);
    }, cfg.timeouts.query);
  }
});
