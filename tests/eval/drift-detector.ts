import Anthropic from '@anthropic-ai/sdk';
import { HfInference } from '@huggingface/inference';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';

dotenv.config();

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const hf = new HfInference(process.env.HF_API_KEY);

const EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2';
const RUNS = 5;
const DRIFT_THRESHOLD = 0.15; // variance = 1 - minSimilarity; above this = drift
const DEFAULT_PROMPTS_FILE = path.join(__dirname, 'promptfoo.yaml');
const PROMPTS_FILE = process.env.DRIFT_PROMPTS_FILE || DEFAULT_PROMPTS_FILE;

function cosineSimilarity(a: number[], b: number[]): number {
  const dot = a.reduce((sum, v, i) => sum + v * b[i], 0);
  const magA = Math.sqrt(a.reduce((sum, v) => sum + v * v, 0));
  const magB = Math.sqrt(b.reduce((sum, v) => sum + v * v, 0));
  return dot / (magA * magB);
}

function pairwiseSimilarities(embeddings: number[][]): number[] {
  const sims: number[] = [];
  for (let i = 0; i < embeddings.length; i++) {
    for (let j = i + 1; j < embeddings.length; j++) {
      sims.push(cosineSimilarity(embeddings[i], embeddings[j]));
    }
  }
  return sims;
}

function parseYamlScalar(raw: string): string {
  const value = raw.trim();
  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    return value.slice(1, -1).replace(/\\"/g, '"').replace(/\\'/g, "'");
  }
  return value;
}

function loadPromptsFromPromptfoo(configPath: string): string[] {
  if (!fs.existsSync(configPath)) {
    throw new Error(`Prompt source file not found: ${configPath}`);
  }

  const yaml = fs.readFileSync(configPath, 'utf-8');
  const lines = yaml.split(/\r?\n/);
  const prompts: string[] = [];
  let varsIndent = -1;

  for (const line of lines) {
    const varsMatch = line.match(/^(\s*)vars:\s*$/);
    if (varsMatch) {
      varsIndent = varsMatch[1].length;
      continue;
    }

    if (varsIndent >= 0) {
      const currentIndent = (line.match(/^(\s*)/)?.[1].length ?? 0);
      const trimmed = line.trim();

      if (trimmed === '' || trimmed.startsWith('#')) {
        continue;
      }

      if (currentIndent <= varsIndent) {
        varsIndent = -1;
        continue;
      }

      const questionMatch = line.match(/^\s*question:\s*(.+)\s*$/);
      if (questionMatch) {
        const question = parseYamlScalar(questionMatch[1]);
        if (question.trim().length > 0) {
          prompts.push(question);
        }
        varsIndent = -1;
      }
    }
  }

  if (prompts.length === 0) {
    throw new Error(
      `No prompts found in ${configPath}. Expected tests[].vars.question entries in promptfoo YAML.`
    );
  }

  return prompts;
}

function toNumberVector(value: unknown): number[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error('Embedding response is empty or invalid.');
  }

  if (typeof value[0] === 'number') {
    return value as number[];
  }

  if (Array.isArray(value[0])) {
    const rows = value as number[][];
    const dims = rows[0]?.length;
    if (!dims) {
      throw new Error('Embedding matrix is empty.');
    }

    const pooled = new Array<number>(dims).fill(0);
    for (const row of rows) {
      for (let i = 0; i < dims; i++) {
        pooled[i] += row[i] ?? 0;
      }
    }

    return pooled.map((sum) => sum / rows.length);
  }

  throw new Error('Unsupported embedding response shape from Hugging Face API.');
}

async function embed(text: string): Promise<number[]> {
  const response = await hf.featureExtraction({
    model: EMBEDDING_MODEL,
    inputs: text,
  });
  return toNumberVector(response);
}

async function callClaude(prompt: string): Promise<string> {
  const res = await anthropic.messages.create({
    model: 'claude-sonnet-4-6',
    max_tokens: 256,
    messages: [{ role: 'user', content: prompt }],
  });
  return res.content[0].type === 'text' ? res.content[0].text : '';
}

interface PromptResult {
  prompt: string;
  outputs: string[];
  similarities: number[];
  minSimilarity: number;
  maxSimilarity: number;
  avgSimilarity: number;
  variance: number;
  driftDetected: boolean;
}

interface DriftReport {
  timestamp: string;
  model: string;
  embeddingModel: string;
  promptSource: string;
  runsPerPrompt: number;
  driftThreshold: number;
  results: PromptResult[];
  summary: {
    total: number;
    drifted: number;
    stable: number;
    overallStable: boolean;
  };
}

async function detectDrift(): Promise<void> {
  if (!process.env.HF_API_KEY) {
    throw new Error('HF_API_KEY is required for embedding-based drift detection.');
  }

  const probePrompts = loadPromptsFromPromptfoo(PROMPTS_FILE);

  console.log(`Drift Detection — ${probePrompts.length} prompts × ${RUNS} runs`);
  console.log(`Prompt source: ${PROMPTS_FILE}`);
  console.log(`Threshold: variance > ${DRIFT_THRESHOLD} = drift\n`);

  const results: PromptResult[] = [];

  for (const [idx, prompt] of probePrompts.entries()) {
    console.log(`[${idx + 1}/${probePrompts.length}] "${prompt}"`);

    const outputs: string[] = [];
    for (let i = 0; i < RUNS; i++) {
      const out = await callClaude(prompt);
      outputs.push(out);
      process.stdout.write(`  collecting run ${i + 1}/${RUNS}...\r`);
    }
    console.log();

    const embeddings = await Promise.all(outputs.map(embed));

    const similarities = pairwiseSimilarities(embeddings);
    const minSim = Math.min(...similarities);
    const maxSim = Math.max(...similarities);
    const avgSim = similarities.reduce((a, b) => a + b, 0) / similarities.length;
    const variance = 1 - minSim;
    const driftDetected = variance > DRIFT_THRESHOLD;

    console.log(`  min similarity : ${minSim.toFixed(4)}`);
    console.log(`  max similarity : ${maxSim.toFixed(4)}`);
    console.log(`  avg similarity : ${avgSim.toFixed(4)}`);
    console.log(
      `  variance       : ${variance.toFixed(4)}  ->  ${driftDetected ? 'DRIFT DETECTED' : 'stable'}\n`
    );

    results.push({
      prompt,
      outputs,
      similarities,
      minSimilarity: minSim,
      maxSimilarity: maxSim,
      avgSimilarity: avgSim,
      variance,
      driftDetected,
    });
  }

  const drifted = results.filter((r) => r.driftDetected).length;

  const report: DriftReport = {
    timestamp: new Date().toISOString(),
    model: 'claude-sonnet-4-6',
    embeddingModel: `${EMBEDDING_MODEL} (Hugging Face Inference API)`,
    promptSource: PROMPTS_FILE,
    runsPerPrompt: RUNS,
    driftThreshold: DRIFT_THRESHOLD,
    results,
    summary: {
      total: results.length,
      drifted,
      stable: results.length - drifted,
      overallStable: drifted === 0,
    },
  };

  const outPath = path.join(__dirname, 'drift-report.json');
  fs.writeFileSync(outPath, JSON.stringify(report, null, 2));

  console.log('-'.repeat(50));
  console.log(`Total prompts : ${report.summary.total}`);
  console.log(`Drifted       : ${drifted}`);
  console.log(`Stable        : ${report.summary.stable}`);
  console.log(`Overall       : ${report.summary.overallStable ? 'STABLE' : 'DRIFT DETECTED'}`);
  console.log(`\nReport saved -> ${outPath}`);

  if (!report.summary.overallStable) {
    process.exit(1);
  }
}

detectDrift().catch(console.error);
