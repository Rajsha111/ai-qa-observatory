/**
 * RAG HTML Report Generator
 * ─────────────────────────────────────────────────────────────────────────────
 * Runs every RAG test scenario live against the service and produces a
 * self-contained HTML file showing:
 *
 *   · Query text
 *   · Retrieved chunks — similarity score, relevance score, reason
 *   · LLM-judge claims breakdown (claim-by-claim, supported or not)
 *   · Ragas faithfulness score (where applicable)
 *   · Final LLM answer
 *   · Pass / Fail result for every test case
 *
 * Usage:
 *   npm run test:rag:report
 *
 * Prerequisite: Python RAG service must be running on port 8001
 *   cd tests/rag/service && python server.py
 *
 * Output: rag-report.html (project root)
 */

import Anthropic from "@anthropic-ai/sdk";
import dotenv from "dotenv";
import { writeFileSync } from "fs";
import { join } from "path";
import type { RAGQueryResponse, ReterivedChunk } from "../../src/shared/types";
import { ingestTestDocuments } from "./fixtures";
import {
  loadRagConfig,
  loadTestCases,
  loadStaleUpdate,
  loadPrompt,
  type StaleDocEntry,
} from "./lib/config-loader";

dotenv.config();

const cfg        = loadRagConfig();
const testCases  = loadTestCases();
const stale      = loadStaleUpdate();
const RAG_SERVICE = cfg.service.ragServiceUrl;

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// ── Data types ────────────────────────────────────────────────────────────────

interface Claim { claim: string; supported: boolean; }

interface FaithfulnessResult {
  query:       string;
  description: string;
  answer:      string;
  chunks:      Array<{ content: string; source: string }>;
  llmScore:    number;
  ragasScore:  number | null;
  claims:      Claim[];
  threshold:   number;
  passed:      boolean;
  error?:      string;
}

interface ChunkScore {
  content:         string;
  source:          string;
  similarityScore: number;
  relevanceScore:  number;
  reason:          string;
  relevant:        boolean;
}

interface PrecisionResult {
  query:            string;
  description:      string;
  answer:           string;
  chunks:           ChunkScore[];
  contextPrecision: number;
  relevantCount:    number;
  totalChunks:      number;
  minRequired:      number;
  passed:           boolean;
  error?:           string;
}

interface NegativeResult {
  query:    string;
  category: string;
  answer:   string;
  refused:  boolean;
  passed:   boolean;
  error?:   string;
}

interface RetrievalResult {
  query:           string;
  description:     string;
  expectedSource:  string;
  retrievedChunks: Array<{ source: string; score: number }>;
  found:           boolean;
  passed:          boolean;
  error?:          string;
}

interface StalePhaseResult {
  label:          string;
  assertionValue: string;
  answer:         string;
  containsValue:  boolean;
}

interface StaleResult {
  query:           string;
  v1Phase:         StalePhaseResult;
  v2Phase:         StalePhaseResult;
  oldValueAbsent:  boolean;
  allPassed:       boolean;
  error?:          string;
}

// ── Utility helpers ───────────────────────────────────────────────────────────

function sleep(ms: number) { return new Promise(r => setTimeout(r, ms)); }

async function queryRAG(query: string, topK?: number): Promise<RAGQueryResponse> {
  const res = await fetch(`${RAG_SERVICE}/query`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ query, topK: topK ?? cfg.retrieval.defaultTopK }),
  });
  if (!res.ok) throw new Error(`Query failed (${res.status}): ${await res.text()}`);
  return res.json() as Promise<RAGQueryResponse>;
}

function containsRefusal(text: string): boolean {
  const lower = text.toLowerCase();
  return cfg.refusalPhrases.some(p => lower.includes(p));
}

// ── LLM-as-Judge ─────────────────────────────────────────────────────────────

async function scoreFaithfulnessLLM(
  answer: string,
  chunks: string[]
): Promise<{ score: number; claims: Claim[] }> {
  const context = chunks.join("\n\n---\n\n");
  const prompt  = loadPrompt("faithfulness", { context, answer });
  const response = await anthropic.messages.create({
    model:      cfg.llm.judge.model,
    max_tokens: cfg.llm.judge.maxTokens,
    messages:   [{ role: "user", content: prompt }],
  });
  const text  = response.content[0].type === "text" ? response.content[0].text : "{}";
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) return { score: 0, claims: [] };
  const r = JSON.parse(match[0]);
  return { score: r.faithfulness_score ?? 0, claims: (r.claims ?? []) as Claim[] };
}

async function scoreChunkRelevance(
  query: string,
  chunk: string
): Promise<{ score: number; reason: string }> {
  const prompt = loadPrompt("chunk-relevance", { query, chunk });
  const response = await anthropic.messages.create({
    model:      cfg.llm.judge.model,
    max_tokens: 256,
    messages:   [{ role: "user", content: prompt }],
  });
  const text  = response.content[0].type === "text" ? response.content[0].text : "{}";
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) return { score: 0, reason: "parse error" };
  const r = JSON.parse(match[0]);
  return { score: r.score ?? 0, reason: r.reason ?? "" };
}

// ── Test runners ──────────────────────────────────────────────────────────────

async function runFaithfulness(): Promise<FaithfulnessResult[]> {
  const results: FaithfulnessResult[] = [];
  const queries = testCases.faithfulnessQueries;

  // LLM-as-Judge for queries 1 & 2
  for (const q of queries.slice(0, 2)) {
    process.stdout.write(`  faithfulness: "${q.query.slice(0, 60)}..." `);
    try {
      const data = await queryRAG(q.query);
      const { score, claims } = await scoreFaithfulnessLLM(
        data.answer,
        data.retrievedChunks.map(c => c.content)
      );
      const passed = score >= cfg.thresholds.faithfulness;
      console.log(passed ? "✓" : "✗");
      results.push({
        query: q.query, description: q.description, answer: data.answer,
        chunks: data.retrievedChunks.map(c => ({ content: c.content, source: c.source ?? "" })),
        llmScore: score, ragasScore: null, claims,
        threshold: cfg.thresholds.faithfulness, passed,
      });
    } catch (e) {
      console.log("ERROR");
      results.push({
        query: q.query, description: q.description, answer: "", chunks: [],
        llmScore: 0, ragasScore: null, claims: [],
        threshold: cfg.thresholds.faithfulness, passed: false, error: String(e),
      });
    }
  }

  // LLM-as-Judge + Ragas for query 3
  if (queries[2]) {
    const q = queries[2];
    process.stdout.write(`  faithfulness+ragas: "${q.query.slice(0, 50)}..." `);
    try {
      const data = await queryRAG(q.query);
      const { score: llmScore, claims } = await scoreFaithfulnessLLM(
        data.answer,
        data.retrievedChunks.map(c => c.content)
      );
      const evalRes = await fetch(`${RAG_SERVICE}/evaluate`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          question: q.query,
          answer:   data.answer,
          contexts: data.retrievedChunks.map(c => c.content),
        }),
      });
      const evalData = await evalRes.json() as { faithfulness: number };
      const ragasScore = evalData.faithfulness;
      const passed = llmScore >= cfg.thresholds.faithfulness && ragasScore >= cfg.thresholds.faithfulness;
      console.log(passed ? "✓" : "✗");
      results.push({
        query: q.query, description: `${q.description} (+ Ragas)`, answer: data.answer,
        chunks: data.retrievedChunks.map(c => ({ content: c.content, source: c.source ?? "" })),
        llmScore, ragasScore, claims,
        threshold: cfg.thresholds.faithfulness, passed,
      });
    } catch (e) {
      console.log("ERROR");
      results.push({
        query: q.query, description: queries[2].description, answer: "", chunks: [],
        llmScore: 0, ragasScore: null, claims: [],
        threshold: cfg.thresholds.faithfulness, passed: false, error: String(e),
      });
    }
  }

  return results;
}

async function runPrecision(): Promise<PrecisionResult[]> {
  const results: PrecisionResult[] = [];
  for (const { query, description } of testCases.precisionQueries) {
    process.stdout.write(`  precision: "${query.slice(0, 60)}..." `);
    try {
      const data = await queryRAG(query);
      const scored = await Promise.all(
        data.retrievedChunks.map(async (chunk: ReterivedChunk) => {
          const { score, reason } = await scoreChunkRelevance(query, chunk.content);
          return {
            content: chunk.content, source: chunk.source ?? "",
            similarityScore: chunk.score, relevanceScore: score,
            reason, relevant: score >= cfg.thresholds.chunkRelevance,
          } as ChunkScore;
        })
      );
      const relevantCount    = scored.filter(c => c.relevant).length;
      const contextPrecision = relevantCount / scored.length;
      const passed           = relevantCount >= cfg.thresholds.minRelevantChunks;
      console.log(passed ? "✓" : "✗");
      results.push({
        query, description, answer: data.answer, chunks: scored,
        contextPrecision, relevantCount, totalChunks: scored.length,
        minRequired: cfg.thresholds.minRelevantChunks, passed,
      });
    } catch (e) {
      console.log("ERROR");
      results.push({
        query, description, answer: "", chunks: [],
        contextPrecision: 0, relevantCount: 0, totalChunks: 0,
        minRequired: cfg.thresholds.minRelevantChunks, passed: false, error: String(e),
      });
    }
  }
  return results;
}

async function runNegative(): Promise<NegativeResult[]> {
  const results: NegativeResult[] = [];
  for (const { query, category } of testCases.negativeQueries) {
    await sleep(1500); // rate-limit guard for embedding API
    process.stdout.write(`  negative [${category}]: "${query.slice(0, 50)}..." `);
    try {
      const data   = await queryRAG(query);
      const refused = containsRefusal(data.answer);
      console.log(refused ? "✓ refused" : "✗ hallucinated");
      results.push({ query, category, answer: data.answer, refused, passed: refused });
    } catch (e) {
      console.log("ERROR");
      results.push({ query, category, answer: "", refused: false, passed: false, error: String(e) });
    }
  }
  return results;
}

async function runRetrieval(): Promise<RetrievalResult[]> {
  const results: RetrievalResult[] = [];
  for (const { query, expectedSource, description } of testCases.retrievalExpectations) {
    process.stdout.write(`  retrieval: "${description}" `);
    try {
      const data    = await queryRAG(query);
      const sources = data.retrievedChunks.map(c => c.source ?? "");
      const found   = sources.some(s => s.includes(expectedSource));
      console.log(found ? "✓" : "✗");
      results.push({
        query, description, expectedSource,
        retrievedChunks: data.retrievedChunks.map(c => ({ source: c.source ?? "", score: c.score })),
        found, passed: found,
      });
    } catch (e) {
      console.log("ERROR");
      results.push({
        query, description, expectedSource, retrievedChunks: [],
        found: false, passed: false, error: String(e),
      });
    }
  }
  return results;
}

async function runStale(): Promise<StaleResult> {
  const clearCollection = async () => {
    const res = await fetch(`${RAG_SERVICE}/collection`, { method: "DELETE" });
    if (!res.ok) throw new Error(`Clear failed: ${await res.text()}`);
  };
  const ingestDoc = async (doc: StaleDocEntry) => {
    const res = await fetch(`${RAG_SERVICE}/ingest`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ documents: [doc] }),
    });
    if (!res.ok) throw new Error(`Ingest failed: ${await res.text()}`);
  };

  try {
    process.stdout.write(`  stale phase 1 (v1) `);
    await clearCollection();
    await ingestDoc(stale.v1);
    const v1Data      = await queryRAG(stale.query);
    const v1Contains  = new RegExp(`\\b${stale.v1.assertion}\\b`).test(v1Data.answer.toLowerCase());
    console.log(v1Contains ? "✓" : "✗");

    process.stdout.write(`  stale phase 2 (v2) `);
    await clearCollection();
    await ingestDoc(stale.v2);
    const v2Data      = await queryRAG(stale.query);
    const v2Contains  = new RegExp(`\\b${stale.v2.assertion}\\b`).test(v2Data.answer.toLowerCase());
    const oldAbsent   = !new RegExp(`\\b${stale.v1.assertion}\\b`).test(v2Data.answer);
    console.log(v2Contains && oldAbsent ? "✓" : "✗");

    return {
      query: stale.query,
      v1Phase: { label: `v1 — ${stale.v1.assertion}-day policy`, assertionValue: stale.v1.assertion, answer: v1Data.answer, containsValue: v1Contains },
      v2Phase: { label: `v2 — ${stale.v2.assertion}-day policy`, assertionValue: stale.v2.assertion, answer: v2Data.answer, containsValue: v2Contains },
      oldValueAbsent: oldAbsent,
      allPassed:      v1Contains && v2Contains && oldAbsent,
    };
  } catch (e) {
    console.log("ERROR");
    return {
      query: stale.query,
      v1Phase: { label: `v1 — ${stale.v1.assertion}-day`, assertionValue: stale.v1.assertion, answer: "", containsValue: false },
      v2Phase: { label: `v2 — ${stale.v2.assertion}-day`, assertionValue: stale.v2.assertion, answer: "", containsValue: false },
      oldValueAbsent: false, allPassed: false, error: String(e),
    };
  }
}

// ── HTML generation ───────────────────────────────────────────────────────────

function esc(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function badge(passed: boolean): string {
  return passed
    ? `<span class="badge pass">PASS</span>`
    : `<span class="badge fail">FAIL</span>`;
}

function scoreBar(value: number | null | undefined, threshold?: number): string {
  if (value == null || isNaN(value as number)) {
    return `<div class="score-bar-wrap"><span class="score-label" style="color:#94a3b8">N/A</span></div>`;
  }
  const pct   = Math.round(value * 100);
  const color = threshold !== undefined
    ? (value >= threshold ? "#22c55e" : "#ef4444")
    : "#6366f1";
  return `
    <div class="score-bar-wrap">
      <div class="score-bar" style="width:${pct}%; background:${color};"></div>
      <span class="score-label">${value.toFixed(2)}</span>
    </div>`;
}

function faithfulnessSection(results: FaithfulnessResult[]): string {
  const passed = results.filter(r => r.passed).length;
  const cards  = results.map((r, i) => `
    <div class="card ${r.passed ? "pass-card" : "fail-card"}">
      <div class="card-header">
        ${badge(r.passed)}
        <span class="test-index">#${i + 1}</span>
        <span class="test-desc">${esc(r.description)}</span>
      </div>
      ${r.error ? `<div class="error-box">Error: ${esc(r.error)}</div>` : `
      <div class="card-body">
        <div class="field-row">
          <label>Query</label>
          <div class="field-value query-text">${esc(r.query)}</div>
        </div>
        <div class="field-row">
          <label>Answer</label>
          <div class="field-value answer-text">${esc(r.answer)}</div>
        </div>
        <div class="scores-row">
          <div class="score-item">
            <label>LLM Judge Score</label>
            ${scoreBar(r.llmScore, r.threshold)}
            <span class="threshold-label">threshold: ${r.threshold}</span>
          </div>
          ${r.ragasScore != null ? `
          <div class="score-item">
            <label>Ragas Score</label>
            ${scoreBar(r.ragasScore, r.threshold)}
            <span class="threshold-label">threshold: ${r.threshold}</span>
          </div>` : ""}
        </div>
        ${r.claims.length > 0 ? `
        <details>
          <summary>Claims breakdown (${r.claims.filter(c => c.supported).length}/${r.claims.length} supported)</summary>
          <table class="claims-table">
            <thead><tr><th>Supported</th><th>Claim</th></tr></thead>
            <tbody>
              ${r.claims.map(c => `
              <tr class="${c.supported ? "claim-ok" : "claim-bad"}">
                <td class="claim-icon">${c.supported ? "✓" : "✗"}</td>
                <td>${esc(c.claim)}</td>
              </tr>`).join("")}
            </tbody>
          </table>
        </details>` : ""}
        <details>
          <summary>Retrieved chunks (${r.chunks.length})</summary>
          ${r.chunks.map((ch, ci) => `
          <div class="chunk-block">
            <div class="chunk-meta">Chunk ${ci + 1} · Source: <code>${esc(ch.source)}</code></div>
            <div class="chunk-content">${esc(ch.content)}</div>
          </div>`).join("")}
        </details>
      </div>`}
    </div>`).join("");

  return `
    <section>
      <div class="section-header">
        <h2>Faithfulness Tests</h2>
        <span class="section-stats">${passed}/${results.length} passed</span>
      </div>
      <p class="section-desc">Tests whether the LLM answer is grounded in retrieved chunks (no training-data leakage). Scored by LLM-as-Judge and Ragas.</p>
      ${cards}
    </section>`;
}

function precisionSection(results: PrecisionResult[]): string {
  const passed = results.filter(r => r.passed).length;
  const cards  = results.map((r, i) => `
    <div class="card ${r.passed ? "pass-card" : "fail-card"}">
      <div class="card-header">
        ${badge(r.passed)}
        <span class="test-index">#${i + 1}</span>
        <span class="test-desc">${esc(r.description)}</span>
      </div>
      ${r.error ? `<div class="error-box">Error: ${esc(r.error)}</div>` : `
      <div class="card-body">
        <div class="field-row">
          <label>Query</label>
          <div class="field-value query-text">${esc(r.query)}</div>
        </div>
        <div class="field-row">
          <label>Answer</label>
          <div class="field-value answer-text">${esc(r.answer)}</div>
        </div>
        <div class="precision-summary">
          Context Precision: <strong>${r.relevantCount}/${r.totalChunks}</strong>
          = <strong>${r.contextPrecision.toFixed(2)}</strong>
          &nbsp;·&nbsp; Required: ≥${r.minRequired}/${r.totalChunks} relevant
        </div>
        <details open>
          <summary>Chunk relevance breakdown</summary>
          <table class="chunk-table">
            <thead>
              <tr>
                <th>#</th><th>Source</th><th>Similarity</th>
                <th>Relevance Score</th><th>Relevant?</th><th>Reason</th>
              </tr>
            </thead>
            <tbody>
              ${r.chunks.map((ch, ci) => `
              <tr class="${ch.relevant ? "chunk-ok" : "chunk-bad"}">
                <td>${ci + 1}</td>
                <td><code>${esc(ch.source)}</code></td>
                <td>${ch.similarityScore.toFixed(3)}</td>
                <td>
                  ${scoreBar(ch.relevanceScore, cfg.thresholds.chunkRelevance)}
                </td>
                <td class="claim-icon">${ch.relevant ? "✓" : "✗"}</td>
                <td class="reason-text">${esc(ch.reason)}</td>
              </tr>`).join("")}
            </tbody>
          </table>
        </details>
      </div>`}
    </div>`).join("");

  return `
    <section>
      <div class="section-header">
        <h2>Context Precision Tests</h2>
        <span class="section-stats">${passed}/${results.length} passed</span>
      </div>
      <p class="section-desc">Are the top-K retrieved chunks actually relevant to the query? Scored per-chunk by LLM judge.</p>
      ${cards}
    </section>`;
}

function negativeSection(results: NegativeResult[]): string {
  const passed = results.filter(r => r.passed).length;
  const rows   = results.map((r, i) => `
    <tr class="${r.passed ? "row-pass" : "row-fail"}">
      <td>${i + 1}</td>
      <td><span class="category-badge">${esc(r.category)}</span></td>
      <td class="query-text">${esc(r.query)}</td>
      <td class="answer-text">${esc(r.answer)}</td>
      <td class="claim-icon">${r.refused ? "✓ refused" : "✗ answered"}</td>
      <td>${badge(r.passed)}</td>
    </tr>
    ${r.error ? `<tr><td colspan="6" class="error-box">Error: ${esc(r.error)}</td></tr>` : ""}`).join("");

  return `
    <section>
      <div class="section-header">
        <h2>Negative Tests — Hallucination Detection</h2>
        <span class="section-stats">${passed}/${results.length} passed</span>
      </div>
      <p class="section-desc">Questions outside the knowledge base. The system must refuse to answer — a confident answer is a hallucination.</p>
      <table class="full-table">
        <thead>
          <tr><th>#</th><th>Category</th><th>Query</th><th>Answer</th><th>Refused?</th><th>Result</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </section>`;
}

function retrievalSection(results: RetrievalResult[]): string {
  const passed = results.filter(r => r.passed).length;
  const cards  = results.map((r, i) => `
    <div class="card ${r.passed ? "pass-card" : "fail-card"}">
      <div class="card-header">
        ${badge(r.passed)}
        <span class="test-index">#${i + 1}</span>
        <span class="test-desc">${esc(r.description)}</span>
      </div>
      ${r.error ? `<div class="error-box">Error: ${esc(r.error)}</div>` : `
      <div class="card-body">
        <div class="field-row">
          <label>Query</label>
          <div class="field-value query-text">${esc(r.query)}</div>
        </div>
        <div class="field-row">
          <label>Expected source</label>
          <div class="field-value"><code class="${r.found ? "source-ok" : "source-bad"}">${esc(r.expectedSource)}</code></div>
        </div>
        <table class="chunk-table">
          <thead><tr><th>Rank</th><th>Retrieved Source</th><th>Similarity Score</th><th>Matches?</th></tr></thead>
          <tbody>
            ${r.retrievedChunks.map((ch, ci) => {
              const matches = ch.source.includes(r.expectedSource);
              return `
              <tr class="${matches ? "chunk-ok" : ""}">
                <td>${ci + 1}</td>
                <td><code>${esc(ch.source)}</code></td>
                <td>${ch.score.toFixed(3)}</td>
                <td class="claim-icon">${matches ? "✓" : "—"}</td>
              </tr>`;
            }).join("")}
          </tbody>
        </table>
      </div>`}
    </div>`).join("");

  return `
    <section>
      <div class="section-header">
        <h2>Retrieval Unit Tests</h2>
        <span class="section-stats">${passed}/${results.length} passed</span>
      </div>
      <p class="section-desc">Known query → expected document ID mappings. Deterministic regression suite — catches embedding model or chunking changes.</p>
      ${cards}
    </section>`;
}

function staleSection(result: StaleResult | null): string {
  if (!result) return "";
  const phaseRow = (phase: StalePhaseResult, label: string) => `
    <div class="stale-phase ${phase.containsValue ? "pass-card" : "fail-card"}">
      <div class="card-header">
        ${badge(phase.containsValue)}
        <span class="test-desc">${esc(label)}: expects <code>${esc(phase.assertionValue)}</code> in answer</span>
      </div>
      <div class="card-body">
        <div class="field-row">
          <label>Answer</label>
          <div class="field-value answer-text">${esc(phase.answer || "(empty)")}</div>
        </div>
      </div>
    </div>`;

  return `
    <section>
      <div class="section-header">
        <h2>Stale Data Test — Update Propagation</h2>
        <span class="section-stats">${result.allPassed ? "PASSED" : "FAILED"}</span>
      </div>
      <p class="section-desc">Simulates a document update. After re-ingesting a changed document the system must reflect the new value and drop the old one.</p>
      ${result.error ? `<div class="error-box">Error: ${esc(result.error)}</div>` : `
      <div class="field-row" style="margin-bottom:12px">
        <label>Query</label>
        <div class="field-value query-text">${esc(result.query)}</div>
      </div>
      ${phaseRow(result.v1Phase, "Phase 1 — ingest v1")}
      ${phaseRow(result.v2Phase, "Phase 2 — re-ingest v2")}
      <div class="card ${result.oldValueAbsent ? "pass-card" : "fail-card"}" style="margin-top:8px">
        <div class="card-header">
          ${badge(result.oldValueAbsent)}
          <span class="test-desc">Old value (<code>${esc(result.v1Phase.assertionValue)}</code>) absent from v2 answer</span>
        </div>
      </div>`}
    </section>`;
}

function buildHTML(
  faithfulness: FaithfulnessResult[],
  precision:    PrecisionResult[],
  negative:     NegativeResult[],
  retrieval:    RetrievalResult[],
  staleResult:  StaleResult | null,
  generatedAt:  string
): string {
  const allTests = [
    ...faithfulness.map(r => r.passed),
    ...precision.map(r => r.passed),
    ...negative.map(r => r.passed),
    ...retrieval.map(r => r.passed),
    ...(staleResult ? [staleResult.v1Phase.containsValue, staleResult.v2Phase.containsValue, staleResult.oldValueAbsent] : []),
  ];
  const totalPassed = allTests.filter(Boolean).length;
  const totalFailed = allTests.length - totalPassed;
  const passRate    = allTests.length ? Math.round((totalPassed / allTests.length) * 100) : 0;

  const sectionSummary = (label: string, results: Array<{ passed: boolean }>) => {
    const p = results.filter(r => r.passed).length;
    const color = p === results.length ? "#22c55e" : p === 0 ? "#ef4444" : "#f59e0b";
    return `
      <div class="summary-card" style="border-top: 3px solid ${color}">
        <div class="summary-count" style="color:${color}">${p}/${results.length}</div>
        <div class="summary-label">${label}</div>
      </div>`;
  };

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RAG Test Report — ${generatedAt}</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 14px; line-height: 1.5;
      background: #f1f5f9; color: #1e293b;
    }
    a { color: #6366f1; }
    code {
      font-family: "SF Mono", Menlo, monospace; font-size: 12px;
      background: #e2e8f0; padding: 1px 5px; border-radius: 3px;
    }

    /* ── Layout ── */
    header { background: #1e293b; color: #f8fafc; padding: 28px 40px; }
    header h1 { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
    .meta { font-size: 12px; color: #94a3b8; margin-bottom: 20px; }
    main { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }
    section { margin-bottom: 40px; }

    /* ── Overall summary ── */
    .overall-summary {
      display: flex; align-items: center; gap: 24px; margin-bottom: 28px;
    }
    .overall-stat { text-align: center; }
    .overall-stat .big-num { font-size: 36px; font-weight: 800; line-height: 1; }
    .overall-stat .big-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; }
    .overall-stat.pass-num .big-num { color: #22c55e; }
    .overall-stat.fail-num .big-num { color: #ef4444; }
    .divider { width: 1px; height: 48px; background: #334155; }
    .pass-rate-circle {
      width: 80px; height: 80px; border-radius: 50%;
      background: conic-gradient(#22c55e calc(${passRate}% * 3.6deg), #334155 0);
      display: flex; align-items: center; justify-content: center;
      font-size: 16px; font-weight: 700;
    }

    /* ── Section summary cards ── */
    .summary-grid { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 20px; }
    .summary-card {
      background: #1e293b; border: 1px solid #334155;
      border-radius: 8px; padding: 12px 18px; min-width: 140px;
    }
    .summary-count { font-size: 22px; font-weight: 700; }
    .summary-label { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: .04em; margin-top: 2px; }

    /* ── Section ── */
    .section-header {
      display: flex; align-items: baseline; gap: 12px; margin-bottom: 8px;
    }
    .section-header h2 { font-size: 17px; font-weight: 700; }
    .section-stats {
      font-size: 12px; font-weight: 600; background: #e2e8f0;
      padding: 2px 8px; border-radius: 10px; color: #475569;
    }
    .section-desc { font-size: 13px; color: #64748b; margin-bottom: 16px; }

    /* ── Cards ── */
    .card {
      background: #fff; border-radius: 10px;
      box-shadow: 0 1px 3px rgba(0,0,0,.08);
      margin-bottom: 12px; overflow: hidden;
      border-left: 4px solid #e2e8f0;
    }
    .pass-card { border-left-color: #22c55e; }
    .fail-card { border-left-color: #ef4444; }

    .card-header {
      display: flex; align-items: center; gap: 10px;
      padding: 12px 16px; background: #f8fafc;
      border-bottom: 1px solid #e2e8f0;
    }
    .test-index { font-size: 11px; color: #94a3b8; font-weight: 600; }
    .test-desc  { font-weight: 600; font-size: 13px; flex: 1; color: #1e293b; }

    .card-body { padding: 14px 16px; display: flex; flex-direction: column; gap: 12px; }

    /* ── Badges ── */
    .badge {
      display: inline-block; font-size: 10px; font-weight: 700;
      letter-spacing: .06em; padding: 2px 7px; border-radius: 4px;
      text-transform: uppercase;
    }
    .badge.pass { background: #dcfce7; color: #15803d; }
    .badge.fail { background: #fee2e2; color: #b91c1c; }

    /* ── Fields ── */
    .field-row { display: flex; gap: 10px; align-items: flex-start; }
    .field-row label {
      font-size: 10px; font-weight: 700; text-transform: uppercase;
      letter-spacing: .06em; color: #64748b;
      min-width: 80px; padding-top: 2px;
    }
    .field-value { flex: 1; }
    .query-text  { font-weight: 600; color: #1e293b; }
    .answer-text { color: #334155; font-size: 13px; }

    /* ── Score bar ── */
    .score-bar-wrap {
      display: flex; align-items: center; gap: 8px; margin-top: 4px;
    }
    .score-bar {
      height: 8px; border-radius: 4px; min-width: 2px; max-width: 200px;
      transition: width .3s;
    }
    .score-label { font-size: 13px; font-weight: 700; color: #1e293b; }
    .threshold-label { font-size: 11px; color: #94a3b8; margin-top: 2px; }

    .scores-row { display: flex; gap: 32px; flex-wrap: wrap; }
    .score-item { display: flex; flex-direction: column; gap: 2px; }
    .score-item label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; color: #64748b; }

    /* ── Tables ── */
    .claims-table, .chunk-table, .full-table {
      width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px;
    }
    .claims-table th, .chunk-table th, .full-table th {
      background: #f1f5f9; text-align: left; padding: 7px 10px;
      font-size: 11px; font-weight: 700; text-transform: uppercase;
      letter-spacing: .05em; color: #64748b; border-bottom: 1px solid #e2e8f0;
    }
    .claims-table td, .chunk-table td, .full-table td {
      padding: 7px 10px; border-bottom: 1px solid #f1f5f9; vertical-align: top;
    }
    .claim-ok td  { background: #f0fdf4; }
    .claim-bad td { background: #fef2f2; }
    .chunk-ok td  { background: #f0fdf4; }
    .chunk-bad td { background: #fef2f2; }
    .row-pass td  { background: #f0fdf4; }
    .row-fail td  { background: #fef2f2; }
    .claim-icon   { font-weight: 700; text-align: center; white-space: nowrap; }
    .reason-text  { color: #475569; font-size: 12px; }

    .source-ok  { color: #15803d; }
    .source-bad { color: #b91c1c; }

    /* ── Chunks ── */
    .chunk-block {
      background: #f8fafc; border: 1px solid #e2e8f0;
      border-radius: 6px; padding: 10px 12px; margin-top: 8px;
    }
    .chunk-meta    { font-size: 11px; color: #64748b; margin-bottom: 6px; }
    .chunk-content { font-size: 12px; color: #334155; white-space: pre-wrap; }

    /* ── Details / Summary ── */
    details { margin-top: 10px; }
    details summary {
      cursor: pointer; font-size: 12px; font-weight: 600;
      color: #6366f1; user-select: none;
      padding: 4px 0;
    }
    details summary:hover { color: #4f46e5; }

    /* ── Stale section ── */
    .stale-phase { margin-bottom: 8px; }

    /* ── Error ── */
    .error-box {
      background: #fef2f2; border: 1px solid #fecaca;
      border-radius: 6px; padding: 10px 14px;
      color: #b91c1c; font-size: 12px; margin: 8px 0;
    }

    /* ── Category badge ── */
    .category-badge {
      display: inline-block; background: #ede9fe; color: #6d28d9;
      font-size: 11px; font-weight: 600; padding: 1px 7px; border-radius: 4px;
    }

    /* ── Precision summary ── */
    .precision-summary {
      background: #f1f5f9; border-radius: 6px; padding: 8px 12px;
      font-size: 13px; color: #334155;
    }

    /* ── Full table wrapper ── */
    .full-table { border-radius: 8px; overflow: hidden; }
    .full-table td.query-text  { max-width: 220px; }
    .full-table td.answer-text { max-width: 300px; }
  </style>
</head>
<body>
  <header>
    <h1>RAG Test Report</h1>
    <div class="meta">Generated: ${generatedAt} &nbsp;·&nbsp; Service: ${esc(RAG_SERVICE)}</div>
    <div class="overall-summary">
      <div class="overall-stat pass-num">
        <div class="big-num">${totalPassed}</div>
        <div class="big-label">Passed</div>
      </div>
      <div class="divider"></div>
      <div class="overall-stat fail-num">
        <div class="big-num">${totalFailed}</div>
        <div class="big-label">Failed</div>
      </div>
      <div class="divider"></div>
      <div class="pass-rate-circle">${passRate}%</div>
    </div>
    <div class="summary-grid">
      ${sectionSummary("Faithfulness", faithfulness)}
      ${sectionSummary("Precision", precision)}
      ${sectionSummary("Negative", negative)}
      ${sectionSummary("Retrieval", retrieval)}
      ${staleResult ? sectionSummary("Stale Data", [{ passed: staleResult.allPassed }]) : ""}
    </div>
  </header>

  <main>
    ${faithfulnessSection(faithfulness)}
    ${precisionSection(precision)}
    ${negativeSection(negative)}
    ${retrievalSection(retrieval)}
    ${staleSection(staleResult)}
  </main>
</body>
</html>`;
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  console.log("\nRAG Report Generator");
  console.log("=".repeat(60));
  console.log(`Service: ${RAG_SERVICE}\n`);

  // Step 1: ingest standard test documents
  console.log("Ingesting test documents...");
  await ingestTestDocuments(RAG_SERVICE);
  console.log("Done.\n");

  // Step 2: run all test suites
  console.log("Running faithfulness tests...");
  const faithfulness = await runFaithfulness();

  console.log("\nRunning precision tests...");
  const precision = await runPrecision();

  console.log("\nRunning negative tests...");
  const negative = await runNegative();

  console.log("\nRunning retrieval unit tests...");
  const retrieval = await runRetrieval();

  // Stale test runs last — it clears the collection
  console.log("\nRunning stale data test...");
  const staleResult = await runStale();

  // Step 3: generate HTML
  const generatedAt = new Date().toISOString().replace("T", " ").slice(0, 19) + " UTC";
  const html        = buildHTML(faithfulness, precision, negative, retrieval, staleResult, generatedAt);
  const outPath     = join(__dirname, "..", "..", "rag-report.html");
  writeFileSync(outPath, html, "utf-8");

  const allPassed = [
    ...faithfulness.map(r => r.passed),
    ...precision.map(r => r.passed),
    ...negative.map(r => r.passed),
    ...retrieval.map(r => r.passed),
    staleResult.allPassed,
  ];
  const passed = allPassed.filter(Boolean).length;

  console.log("\n" + "=".repeat(60));
  console.log(`Report saved: ${outPath}`);
  console.log(`Result: ${passed}/${allPassed.length} checks passed`);
  console.log("=".repeat(60) + "\n");
}

main().catch(err => { console.error(err); process.exit(1); });
