/**
 * STALE DATA TEST — Update Propagation
 * ─────────────────────────────────────────────────────────────────────────────
 * Failure mode: After updating a document, the RAG system still answers with
 * old information because the stale embedding is still in the vector store.
 *
 * ━━━ WHY THIS IS PRODUCTION-CRITICAL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 *
 * This bug is silent. No errors, no crashes — just wrong answers delivered
 * with confidence. Real scenario:
 *
 *   - Compliance policy changes: data retention: 90 days → 60 days
 *   - Your team re-ingests the updated document
 *   - But the OLD embedding (90 days) is still in ChromaDB
 *   - Queries still retrieve the old chunk → system answers "90 days"
 *   - Nobody notices until an audit
 *
 * ━━━ THE ROOT CAUSE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 *
 * Vector stores don't automatically replace embeddings when you re-ingest.
 * Calling VectorStoreIndex.from_documents() with an updated doc just ADDS
 * a new embedding — the old one stays. Now both versions exist in the store.
 *
 * The correct update pattern (production):
 *   collection.delete(where={"source": "doc-004-data-quality"})
 *   then re-ingest the updated document
 *
 * ━━━ HOW THIS TEST WORKS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 *
 *   Phase 1 — baseline:
 *     Clear collection → ingest v1 → query → expect v1 assertion value
 *
 *   Phase 2 — after update:
 *     Clear collection → ingest v2 → query → expect v2 assertion value
 *
 *   Phase 2 passing proves your update propagation works correctly.
 *   Phase 2 failing means your re-index is broken.
 *
 * Both docs and assertions are in data/stale-update.json.
 *
 * ━━━ IMPORTANT: RUN IN ISOLATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 *
 * This test clears the entire collection in beforeAll for each phase.
 * Run it separately to avoid interfering with other tests:
 *
 *   npx vitest tests/rag/stale-data.test.ts --reporter=verbose
 *
 * To run ALL rag tests sequentially (safe):
 *   npx vitest tests/rag/ --reporter=verbose --fileParallelism=false
 */

import { describe, it, expect, beforeAll } from "vitest";
import dotenv from "dotenv";
import type { RAGQueryResponse } from "../../src/shared/types";
import { loadRagConfig, loadStaleUpdate } from "./lib/config-loader";

dotenv.config();

const cfg    = loadRagConfig();
const stale  = loadStaleUpdate();
const RAG_SERVICE  = cfg.service.ragServiceUrl;
const STALE_QUERY  = stale.query;

// ── Helpers ────────────────────────────────────────────────────────────────────
async function clearCollection(): Promise<void> {
  const res = await fetch(`${RAG_SERVICE}/collection`, { method: "DELETE" });
  if (!res.ok)
    throw new Error(`Clear failed (${res.status}): ${await res.text()}`);
}

async function ingest(docs: typeof stale.v1[]): Promise<void> {
  const res = await fetch(`${RAG_SERVICE}/ingest`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ documents: docs }),
  });
  if (!res.ok)
    throw new Error(`Ingest failed (${res.status}): ${await res.text()}`);
}

async function queryRAG(query: string): Promise<RAGQueryResponse> {
  const res = await fetch(`${RAG_SERVICE}/query`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ query, topK: cfg.retrieval.defaultTopK }),
  });
  return res.json() as Promise<RAGQueryResponse>;
}

// ── Phase 1: baseline with v1 document ────────────────────────────────────────
describe(`Phase 1 — v1 document (${stale.v1.assertion}-day policy)`, () => {
  beforeAll(async () => {
    // Clean slate, then ingest ONLY the v1 document
    await clearCollection();
    await ingest([stale.v1]);
  }, cfg.timeouts.setup);

  it(`system returns ${stale.v1.assertion}-day value from v1 document`, async () => {
    const data = await queryRAG(STALE_QUERY);
    console.log(`\nV1 answer: "${data.answer}"`);

    expect(data.answer.toLowerCase()).toMatch(new RegExp(`\\b${stale.v1.assertion}\\b`));
  }, cfg.timeouts.query);
});

// ── Phase 2: re-index with v2 document ────────────────────────────────────────
describe(`Phase 2 — v2 document re-ingested (${stale.v2.assertion}-day policy)`, () => {
  beforeAll(async () => {
    // Simulate the correct re-index cycle:
    //   1. Clear old embeddings (the critical step most pipelines skip)
    //   2. Ingest the updated document
    // If you skip step 1, BOTH v1 and v2 embeddings exist → non-deterministic
    await clearCollection();
    await ingest([stale.v2]);
  }, cfg.timeouts.setup);

  it(`system returns ${stale.v2.assertion}-day requirement after re-ingestion of updated document`, async () => {
    const data = await queryRAG(STALE_QUERY);
    console.log(`\nV2 answer: "${data.answer}"`);

    // Must reflect the NEW policy value
    expect(data.answer.toLowerCase()).toMatch(new RegExp(`\\b${stale.v2.assertion}\\b`));
  }, cfg.timeouts.query);

  it(`old ${stale.v1.assertion}-day value is no longer returned after re-ingestion`, async () => {
    const data = await queryRAG(STALE_QUERY);

    // If this fails: old embedding is still in the store.
    // Your re-ingestion did not replace the stale chunk — this is the bug.
    expect(data.answer).not.toMatch(new RegExp(`\\b${stale.v1.assertion}\\b`));
  }, cfg.timeouts.query);
});
