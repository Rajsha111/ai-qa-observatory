/**
 * NEGATIVE TEST — "Does the system know what it doesn't know?"
 * ─────────────────────────────────────────────────────────────────────────────
 * Failure mode: Confident answer to a question that isn't in the knowledge base.
 *
 * This is your MOST CRITICAL failure mode. A confident wrong answer is far
 * more dangerous than "I don't know" — especially in compliance, legal, or
 * financial contexts where users trust the system.
 *
 * ━━━ THE REAL-WORLD SCENARIO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 *
 * A user asks a compliance chatbot: "How many vacation days do employees get?"
 * The question is NOT in the KB.
 * But the LLM "knows" about typical company policies from training data.
 * It answers: "Employees typically receive 15 days per year."
 * That's the training data talking, not your KB.
 * The user acts on it. HR is called. Trust is destroyed.
 *
 * ━━━ HOW THIS TEST WORKS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 *
 *   1. Load out-of-KB questions from data/test-cases.json
 *   2. Assert each response contains a refusal phrase from config/rag.config.json
 *   3. If it answers confidently → test FAILS → hallucination detected
 *
 * This is the test you'd show in a YouTube Short — run it live and show it
 * catching a hallucination in real time.
 *
 * ━━━ HOW TO RUN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 *
 *   cd tests/rag/service && python server.py
 *   npx vitest tests/rag/negative.test.ts --reporter=verbose
 */

import { describe, it, expect, beforeAll } from "vitest";
import dotenv from "dotenv";
import type { RAGQueryResponse } from "../../src/shared/types";
import { ingestTestDocuments } from "./fixtures";
import { loadRagConfig, loadTestCases } from "./lib/config-loader";
import { sleep } from "../../src/shared/utils";

dotenv.config();

const cfg       = loadRagConfig();
const testCases = loadTestCases();
const RAG_SERVICE     = cfg.service.ragServiceUrl;
const REFUSAL_PHRASES = cfg.refusalPhrases;

// ── Refusal detection ──────────────────────────────────────────────────────────
// We check for multiple phrases because different LLMs phrase refusals differently.
// Being too strict here (exact match) creates false failures.
// Being too loose (substring "not") creates false passes.
// The phrase list lives in config/rag.config.json — tune it to your system.
function containsRefusal(text: string): boolean {
  const lower = text.toLowerCase();
  return REFUSAL_PHRASES.some((phrase) => lower.includes(phrase));
}

beforeAll(async () => {
  await ingestTestDocuments(RAG_SERVICE);
}, cfg.timeouts.setup);

// ── Tests ─────────────────────────────────────────────────────────────────────
describe("Negative Tests — Out-of-KB Hallucination Detection", () => {
  // Dynamically create one test per out-of-KB question (loaded from test-cases.json)
  // This pattern gives you individual pass/fail per question in CI — much more
  // useful than a single test that loops and stops at the first failure.
  for (const { query, category } of testCases.negativeQueries) {
    it(`refuses to answer: [${category}]`, async () => {
      // 1.5s gap between queries — Voyage free tier is rate-limited to ~3 RPM.
      // Ingest already consumed several requests; this keeps us under the limit.
      await sleep(1500);
      const res = await fetch(`${RAG_SERVICE}/query`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ query, topK: cfg.retrieval.defaultTopK }),
      });
      const data = (await res.json()) as RAGQueryResponse;

      console.log(`\nQuery:    "${query}"`);
      console.log(`Answer:   "${data.answer}"`);
      console.log(`Refused:  ${containsRefusal(data.answer)}`);

      // If this assertion fails → the system answered confidently with
      // information from its training data, not from the knowledge base.
      // That is a hallucination. This is your live hallucination catch.
      expect(
        containsRefusal(data.answer),
        `Expected a refusal but got: "${data.answer}"`
      ).toBe(true);
    }, cfg.timeouts.negative);
  }
});
