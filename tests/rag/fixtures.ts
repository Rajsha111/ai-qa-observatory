/**
 * RAG Test Fixtures — Shared Document Corpus
 * ─────────────────────────────────────────────────────────────────────────────
 * Documents are defined in data/documents.json — edit that file to swap domains.
 * This module exists so test files can call ingestTestDocuments() without caring
 * about where the documents come from.
 *
 * Each document has specific testable facts embedded (see documents.json comments):
 *   doc-001  faithfulness score: 0.85          → retrieval-unit target
 *   doc-002  default top-K: 3, chunk: 512 tok  → retrieval-unit target
 *   doc-003  evaluation cadence: quarterly      → retrieval-unit target
 *   doc-004  data freshness: 90 days            → STALE-DATA v1 (becomes 60)
 *   doc-005  P0: 5 min, P1: 30 min response     → retrieval-unit target
 *
 * Topics NOT in the KB (negative.test.ts uses these):
 *   - HR (vacation, compensation)
 *   - Finance (budgets, equity)
 *   - IT infrastructure
 * If the system answers those confidently → hallucination caught.
 */

import { loadDocuments, loadRagConfig } from "./lib/config-loader";

export type { TestDoc } from "./lib/config-loader";

export function getTestDocuments() {
  return loadDocuments();
}

/** POST all test documents to the RAG service /ingest endpoint */
export async function ingestTestDocuments(
  ragServiceUrl?: string
): Promise<void> {
  const url = ragServiceUrl ?? loadRagConfig().service.ragServiceUrl;
  const docs = getTestDocuments();
  const res = await fetch(`${url}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ documents: docs }),
  });
  if (!res.ok) {
    throw new Error(`Ingest failed (${res.status}): ${await res.text()}`);
  }
}
