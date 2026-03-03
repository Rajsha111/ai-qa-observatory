/**
 * RAG Config Loader
 * ─────────────────────────────────────────────────────────────────────────────
 * Single source of truth for loading all RAG test configuration.
 *
 * All values live in config/ and data/ — test files import from here and
 * contain zero hardcoded strings, thresholds, model names, or URLs.
 *
 * To swap domains: edit data/documents.json and data/test-cases.json.
 * To tune thresholds: edit config/rag.config.json.
 * To change prompts: edit config/prompts/*.txt.
 * Test files never need to change.
 */

import fs from "fs";
import path from "path";

// ── TypeScript interfaces ─────────────────────────────────────────────────────

export interface RagConfig {
  service: {
    ragServiceUrl: string;
  };
  llm: {
    judge:     { model: string; maxTokens: number };
    generator: { model: string; maxTokens: number };
    embeddings:{ model: string };
  };
  thresholds: {
    faithfulness:      number;
    minRelevantChunks: number;
    chunkRelevance:    number;
  };
  retrieval: {
    defaultTopK: number;
  };
  timeouts: {
    setup:    number;
    judge:    number;
    ragas:    number;
    query:    number;
    negative: number;
  };
  refusalPhrases: string[];
}

export interface FaithfulnessQuery {
  query:       string;
  description: string;
}

export interface PrecisionQuery {
  query:       string;
  description: string;
}

export interface NegativeQuery {
  query:    string;
  category: string;
}

export interface RetrievalExpectation {
  query:          string;
  expectedSource: string;
  description:    string;
}

export interface TestCases {
  faithfulnessQueries:   FaithfulnessQuery[];
  precisionQueries:      PrecisionQuery[];
  negativeQueries:       NegativeQuery[];
  retrievalExpectations: RetrievalExpectation[];
}

export interface TestDoc {
  id:       string;
  content:  string;
  metadata: Record<string, string>;
}

export interface StaleDocEntry {
  id:        string;
  content:   string;
  metadata:  Record<string, string>;
  assertion: string;
}

export interface StaleUpdate {
  query: string;
  v1:    StaleDocEntry;
  v2:    StaleDocEntry;
}

// ── Path helpers ──────────────────────────────────────────────────────────────

const CONFIG_DIR = path.resolve(__dirname, "..", "config");
const DATA_DIR   = path.resolve(__dirname, "..", "data");

function readJson<T>(filePath: string): T {
  return JSON.parse(fs.readFileSync(filePath, "utf-8")) as T;
}

function readText(filePath: string): string {
  return fs.readFileSync(filePath, "utf-8");
}

// ── Public loaders ────────────────────────────────────────────────────────────

/** Load typed RAG configuration (thresholds, models, URLs, timeouts, refusal phrases). */
export function loadRagConfig(): RagConfig {
  return readJson<RagConfig>(path.join(CONFIG_DIR, "rag.config.json"));
}

/** Load all test cases (faithfulness, precision, negative, retrieval expectations). */
export function loadTestCases(): TestCases {
  return readJson<TestCases>(path.join(DATA_DIR, "test-cases.json"));
}

/** Load the stale-update fixture (v1 + v2 documents and assertions). */
export function loadStaleUpdate(): StaleUpdate {
  return readJson<StaleUpdate>(path.join(DATA_DIR, "stale-update.json"));
}

/** Load the document corpus for ingestion. */
export function loadDocuments(): TestDoc[] {
  const file = readJson<{ documents: TestDoc[] }>(
    path.join(DATA_DIR, "documents.json")
  );
  return file.documents;
}

/**
 * Load a prompt template from config/prompts/ and render it by replacing
 * {{variable}} placeholders with the provided vars map.
 *
 * @param name  Filename without extension, e.g. "faithfulness"
 * @param vars  Key-value pairs to substitute, e.g. { context: "...", answer: "..." }
 */
export function loadPrompt(name: string, vars: Record<string, string> = {}): string {
  const templatePath = path.join(CONFIG_DIR, "prompts", `${name}.txt`);
  let template = readText(templatePath);
  for (const [key, value] of Object.entries(vars)) {
    template = template.replaceAll(`{{${key}}}`, value);
  }
  return template;
}
