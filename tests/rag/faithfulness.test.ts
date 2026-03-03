/**
 * FAITHFULNESS TEST
 * ─────────────────────────────────────────────────────────────────────────────
 * Failure mode: Answer contains claims NOT supported by the retrieved chunks.
 *
 * This is the #1 RAG failure mode. The model draws on its training data
 * instead of (or in addition to) the retrieved context. Even a factually
 * correct answer is a BUG if it wasn't grounded in what was retrieved —
 * because in production, you can't tell which facts came from the KB and
 * which came from model memory.
 *
 * ━━━ TWO SCORING METHODS DEMONSTRATED HERE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 *
 * Method A — LLM-as-Judge (Claude evaluates Claude's output):
 *   - You write the prompt → full control
 *   - Fast, no extra infra beyond the Claude SDK you already have
 *   - Output varies slightly (temperature) but sufficient for thresholds
 *   - Good for: fine-grained assertions, domain-specific criteria
 *
 * Method B — Ragas /evaluate endpoint:
 *   - Standardised algorithm, comparable across projects and teams
 *   - Widely cited in research — interviewers recognise it
 *   - Good for: CI quality gates, benchmarking, executive reporting
 *
 * Use BOTH: LLM-as-Judge for test assertions, Ragas for CI dashboards.
 *
 * ━━━ THRESHOLD: configured in config/rag.config.json ━━━━━━━━━━━━━━━━━━━━━━
 *
 * 0.85 is the Ragas paper's recommended threshold. It means at most 1 in 7
 * claims can be unsupported. Below 0.85 in a compliance context = liability.
 *
 * ━━━ HOW TO RUN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 *
 *   # Terminal 1 — start Python RAG service first
 *   cd tests/rag/service && python server.py
 *
 *   # Terminal 2 — run this test
 *   npx vitest tests/rag/faithfulness.test.ts --reporter=verbose
 */

import { describe, it, expect, beforeAll } from "vitest";
import Anthropic from "@anthropic-ai/sdk";
import dotenv from "dotenv";
import type { RAGQueryResponse } from "../../src/shared/types";
import { ingestTestDocuments } from "./fixtures";
import { loadRagConfig, loadTestCases, loadPrompt } from "./lib/config-loader";

dotenv.config();

const cfg        = loadRagConfig();
const testCases  = loadTestCases();
const RAG_SERVICE = cfg.service.ragServiceUrl;
const FAITHFULNESS_THRESHOLD = cfg.thresholds.faithfulness;

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// ── Setup ──────────────────────────────────────────────────────────────────────
// beforeAll runs ONCE before any test in this file.
// Without this, the vector store is empty and every query returns "I don't know."
beforeAll(async () => {
  await ingestTestDocuments(RAG_SERVICE);
}, cfg.timeouts.setup);

// ── LLM-as-Judge scorer ────────────────────────────────────────────────────────
/**
 * Ask Claude to score how faithfully the answer is grounded in the chunks.
 *
 * The judge asks Claude to:
 *   1. List every factual claim in the answer
 *   2. For each claim, find supporting evidence in the retrieved chunks
 *   3. Score = supported_claims / total_claims
 *
 * Using claude-haiku as judge (cheaper, faster). The judge task is simpler
 * than generation — you don't need Sonnet to evaluate Haiku's output.
 */
async function scoreFaithfulnessLLMJudge(
  answer: string,
  chunks: string[]
): Promise<{ score: number; claims: Array<{ claim: string; supported: boolean }> }> {
  const context = chunks.join("\n\n---\n\n");
  const prompt  = loadPrompt("faithfulness", { context, answer });

  const response = await anthropic.messages.create({
    model:      cfg.llm.judge.model,
    max_tokens: cfg.llm.judge.maxTokens,
    messages:   [{ role: "user", content: prompt }],
  });

  const text =
    response.content[0].type === "text" ? response.content[0].text : "{}";

  // Claude sometimes wraps JSON in markdown code blocks — strip them
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (!jsonMatch) return { score: 0, claims: [] };

  const result = JSON.parse(jsonMatch[0]);
  return {
    score:  result.faithfulness_score ?? 0,
    claims: result.claims ?? [],
  };
}

// ── Tests: Method A — LLM-as-Judge ────────────────────────────────────────────
describe("Faithfulness — LLM-as-Judge (Claude evaluates Claude)", () => {
  // Use the first two faithfulnessQueries for method A
  const [q1, q2] = testCases.faithfulnessQueries;

  it(q1.description, async () => {
    const res = await fetch(`${RAG_SERVICE}/query`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ query: q1.query, topK: cfg.retrieval.defaultTopK }),
    });
    const data = (await res.json()) as RAGQueryResponse;

    const { score, claims } = await scoreFaithfulnessLLMJudge(
      data.answer,
      data.retrievedChunks.map((c) => c.content)
    );

    // Log for learning — see what the judge decided claim by claim
    console.log(`\nAnswer: "${data.answer.slice(0, 200)}"`);
    console.log(`\nClaims evaluated:`);
    claims.forEach((c, i) =>
      console.log(`  [${c.supported ? "+" : "x"}] ${i + 1}. ${c.claim}`)
    );
    console.log(
      `\nFaithfulness: ${score.toFixed(2)} (threshold: ${FAITHFULNESS_THRESHOLD})`
    );

    expect(score).toBeGreaterThanOrEqual(FAITHFULNESS_THRESHOLD);
  }, cfg.timeouts.judge);

  it(q2.description, async () => {
    const res = await fetch(`${RAG_SERVICE}/query`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ query: q2.query, topK: cfg.retrieval.defaultTopK }),
    });
    const data = (await res.json()) as RAGQueryResponse;

    const { score, claims } = await scoreFaithfulnessLLMJudge(
      data.answer,
      data.retrievedChunks.map((c) => c.content)
    );

    console.log(`\nFaithfulness: ${score.toFixed(2)}`);
    console.log(
      `Claims: ${claims.filter((c) => c.supported).length}/${claims.length} supported`
    );

    expect(score).toBeGreaterThanOrEqual(FAITHFULNESS_THRESHOLD);
  }, cfg.timeouts.judge);
});

// ── Tests: Method B — Ragas ────────────────────────────────────────────────────
describe("Faithfulness — Ragas score (standardised metric)", () => {
  // Use the third faithfulnessQuery for Ragas
  const q3 = testCases.faithfulnessQueries[2];

  it("Ragas faithfulness score meets threshold for RAG standards query", async () => {
    // Step 1: get the answer + chunks from the RAG service
    const queryRes = await fetch(`${RAG_SERVICE}/query`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ query: q3.query, topK: cfg.retrieval.defaultTopK }),
    });
    const queryData = (await queryRes.json()) as RAGQueryResponse;

    // Step 2: send to /evaluate for Ragas scoring
    // Ragas takes the same inputs as LLM-as-Judge (question, answer, contexts)
    // but uses a standardised NLI-based algorithm internally.
    const evalRes = await fetch(`${RAG_SERVICE}/evaluate`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        question: q3.query,
        answer:   queryData.answer,
        contexts: queryData.retrievedChunks.map((c) => c.content),
      }),
    });
    const evalData = (await evalRes.json()) as {
      faithfulness: number;
      threshold:    number;
      passed:       boolean;
    };

    console.log(`\nRagas faithfulness: ${evalData.faithfulness}`);
    console.log(`Threshold:          ${evalData.threshold}`);
    console.log(`Passed:             ${evalData.passed}`);

    // This score goes into CI artifacts and dashboard reporting
    expect(evalData.faithfulness).toBeGreaterThanOrEqual(FAITHFULNESS_THRESHOLD);
  }, cfg.timeouts.ragas);
});
