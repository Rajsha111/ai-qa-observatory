import { HfInference } from '@huggingface/inference';
import dotenv from 'dotenv';

dotenv.config();

const client = new HfInference(process.env.HF_API_KEY);
const EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2';
const threshold = 0.8;

function cosineSimilarity(vecA: number[], vecB: number[]): number {
  const dotProduct = vecA.reduce((sum, a, idx) => sum + a * vecB[idx], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (magnitudeA * magnitudeB);
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

async function embeddings(text: string): Promise<number[]> {
  const response = await client.featureExtraction({
    inputs: text,
    model: EMBEDDING_MODEL,
  });

  return toNumberVector(response);
}

export async function semanticScorer(
  output: string,
  context: { vars: Record<string, string> }
): Promise<{ pass: boolean; score: number; reason: string }> {
  if (!process.env.HF_API_KEY) {
    return {
      pass: false,
      score: 0,
      reason: 'HF_API_KEY is missing; cannot compute semantic similarity.',
    };
  }

  const expected = context.vars.expected;
  if (!expected) {
    return { pass: false, score: 0, reason: 'Expected value not provided in context' };
  }

  const [actualEmbedding, expectedEmbedding] = await Promise.all([
    embeddings(output),
    embeddings(expected),
  ]);

  const similarity = cosineSimilarity(actualEmbedding, expectedEmbedding);
  const pass = similarity >= threshold;
  return {
    pass,
    score: similarity,
    reason: pass
      ? 'Output is semantically similar to expected'
      : 'Output is not semantically similar to expected',
  };
}
