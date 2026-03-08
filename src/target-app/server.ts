import express from 'express';
import path from 'path';
import fs from 'fs';
import { spawn } from 'child_process';
import Anthropic from '@anthropic-ai/sdk';
import dotenv from 'dotenv';
import { runAgent } from '../agent/runner';
import type {  ChatRequest,
    RAGQueryRequest, RAGQueryResponse,
    AgentRunRequest, AgentRunResponse,
    AgentStep, } from '../shared/types';

dotenv.config();

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const client = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY || '',
});

app.post('/chat', async (req, res) => {
    const start = Date.now();
    const { messages, systemPrompt } = req.body as ChatRequest;

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    try {
        const stream = client.messages.stream({
            model: 'claude-sonnet-4-6',
            max_tokens: 1024,
            system: systemPrompt ?? 'You are a helpful assistant.',
            messages,
        });

        for await (const event of stream) {
            if (
                event.type === 'content_block_delta' &&
                event.delta.type === 'text_delta'
            ) {
                res.write(`data: ${JSON.stringify({ type: 'delta', text: event.delta.text })}\n\n`);
            }
            if ( event.type === 'message_start') {
                 res.write(`data: ${JSON.stringify({ type: 'start', model: event.message.model })}\n\n`);
            }
            if(event.type === 'message_delta') {
                 res.write(`data: ${JSON.stringify({ type: 'stop', reason: event.delta.stop_reason })}\n\n`);
            }
        }

        const final = await stream.finalMessage();
        res.write(`data: ${JSON.stringify({
            type: 'done',
            latencyMs: Date.now() - start,
            tokenUsage: {
                inputTokens: final.usage.input_tokens,
                outputTokens: final.usage.output_tokens,
                totalTokens: final.usage.input_tokens + final.usage.output_tokens,
            },
        })}\n\n`);
        res.end();
    } catch (error) {
        console.error('Error in /chat:', error);
        if (!res.headersSent) {
            res.status(500).json({ error: 'Internal Server Error' });
        } else {
            res.write(`data: ${JSON.stringify({ type: 'error', message: 'Stream interrupted' })}\n\n`);
            res.end();
        }
    }
});


// ── RAG proxy helper ─────────────────────────────────────────────────────────
// All RAG routes (both the legacy /rag-query and the new /rag/* management
// endpoints) forward to the Flask microservice. Centralising the logic here
// keeps each route handler to a single line.
async function ragProxy(
    req: express.Request,
    res: express.Response,
    upstreamPath: string,
) {
    const ragServiceUrl = process.env.RAG_SERVICE_URL || 'http://localhost:8001';
    try {
        const options: RequestInit = {
            method: req.method,
            headers: { 'Content-Type': 'application/json' },
        };
        if (req.method !== 'GET' && req.method !== 'HEAD') {
            options.body = JSON.stringify(req.body);
        }
        const upstream = await fetch(`${ragServiceUrl}${upstreamPath}`, options);
        const data = await upstream.json();
        res.status(upstream.status).json(data);
    } catch (error) {
        const isConnRefused =
            error instanceof Error && error.message.includes('ECONNREFUSED');
        res.status(isConnRefused ? 503 : 500).json({
            error: isConnRefused
                ? 'RAG service unavailable. Run: cd tests/rag/service && python server.py'
                : 'Internal Server Error',
        });
        if (!isConnRefused) console.error(`Error proxying ${upstreamPath}:`, error);
    }
}

// post rag query — legacy route kept for backwards compatibility with tests
// ragProxy forwards the full req.body ({ query, topK }) to Flask /query
app.post('/rag-query', (req, res) => ragProxy(req, res, '/query'));

// ── RAG management routes — UI calls these, they proxy straight to Flask ──────
app.get   ('/rag/status',          (req, res) => ragProxy(req, res, '/status'));
app.get   ('/rag/prompts',         (req, res) => ragProxy(req, res, '/prompts'));
app.post  ('/rag/prompts/:name',   (req, res) => ragProxy(req, res, `/prompts/${req.params.name}`));
app.get   ('/rag/config',          (req, res) => ragProxy(req, res, '/config'));
app.post  ('/rag/config',          (req, res) => ragProxy(req, res, '/config'));
app.post  ('/rag/ingest',          (req, res) => ragProxy(req, res, '/ingest'));
app.delete('/rag/collection',      (req, res) => ragProxy(req, res, '/collection'));

// ── Chunk relevance assessor — same LLM scorer as precision.test.ts ──────────
// Takes {query, chunks[]} and returns LLM-judged relevance scores + pass/fail
// so the Query UI can show the same signal the test suite uses.
app.post('/rag/assess-chunks', async (req, res) => {
    const { query, chunks } = req.body as {
        query: string;
        chunks: Array<{ content: string; source: string }>;
    };
    try {
        const configPath  = path.join(__dirname, '../../tests/rag/config/rag.config.json');
        const promptPath  = path.join(__dirname, '../../tests/rag/config/prompts/chunk-relevance.txt');
        const config      = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
        const promptTpl   = fs.readFileSync(promptPath, 'utf-8');
        const threshold   = config.thresholds.chunkRelevance as number;
        const model       = config.llm.judge.model as string;

        const results = await Promise.all(chunks.map(async chunk => {
            const prompt = promptTpl
                .replace('{{query}}', query)
                .replace('{{chunk}}', chunk.content);
            const response = await client.messages.create({
                model,
                max_tokens: 256,
                messages: [{ role: 'user', content: prompt }],
            });
            const text = response.content[0].type === 'text' ? response.content[0].text : '{}';
            const match = text.match(/\{[\s\S]*\}/);
            if (!match) return { score: 0, reason: 'parse error', pass: false, threshold };
            const { score = 0, reason = '' } = JSON.parse(match[0]);
            return { score, reason, pass: score >= threshold, threshold };
        }));

        res.json({ results });
    } catch (error) {
        console.error('Error in /rag/assess-chunks:', error);
        res.status(500).json({ error: 'Assessment failed' });
    }
});

// ── Assess config — thresholds + refusal phrases for the Query UI ────────────
app.get('/rag/assess-config', (_req, res) => {
    try {
        const config = JSON.parse(fs.readFileSync(
            path.join(__dirname, '../../tests/rag/config/rag.config.json'), 'utf-8'));
        res.json({ thresholds: config.thresholds, refusalPhrases: config.refusalPhrases });
    } catch (error) {
        console.error('Error reading assess-config:', error);
        res.status(500).json({ error: 'Could not read config' });
    }
});

// ── Faithfulness assessor — same LLM-as-judge as faithfulness.test.ts ────────
app.post('/rag/assess-faithfulness', async (req, res) => {
    const { answer, chunks } = req.body as {
        answer: string;
        chunks: Array<{ content: string }>;
    };
    try {
        const config    = JSON.parse(fs.readFileSync(path.join(__dirname, '../../tests/rag/config/rag.config.json'), 'utf-8'));
        const promptTpl = fs.readFileSync(path.join(__dirname, '../../tests/rag/config/prompts/faithfulness.txt'), 'utf-8');
        const threshold = config.thresholds.faithfulness as number;
        const context   = chunks.map(c => c.content).join('\n\n---\n\n');
        const prompt    = promptTpl.replace('{{context}}', context).replace('{{answer}}', answer);

        const response = await client.messages.create({
            model:      config.llm.judge.model,
            max_tokens: config.llm.judge.maxTokens,
            messages:   [{ role: 'user', content: prompt }],
        });
        const text  = response.content[0].type === 'text' ? response.content[0].text : '{}';
        const match = text.match(/\{[\s\S]*\}/);
        if (!match) return res.json({ score: 0, claims: [], pass: false, threshold });
        const result = JSON.parse(match[0]);
        const score  = result.faithfulness_score ?? 0;
        res.json({ score, claims: result.claims ?? [], pass: score >= threshold, threshold });
    } catch (error) {
        console.error('Error in /rag/assess-faithfulness:', error);
        res.status(500).json({ error: 'Faithfulness assessment failed' });
    }
});

// ── RAG test runner — streams vitest output as SSE ───────────────────────────
app.post('/rag/run-tests', (req, res) => {
    const suite = (req.body.suite as string) || 'all';

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const projectRoot = path.join(__dirname, '../..');
    const testTarget  = suite === 'all' ? 'tests/rag' : `tests/rag/${suite}.test.ts`;

    const child = spawn('npx', ['vitest', 'run', testTarget, '--reporter=verbose', '--fileParallelism=false'], {
        cwd: projectRoot,
        env: process.env,
    });

    const send = (data: object) => res.write(`data: ${JSON.stringify(data)}\n\n`);
    child.stdout.on('data', d => send({ type: 'log', text: d.toString() }));
    child.stderr.on('data', d => send({ type: 'log', text: d.toString() }));
    child.on('close', code => { send({ type: 'done', exitCode: code }); res.end(); });
    req.on('close', () => child.kill());
});

// post agent run
app.post('/agent-run', async (req, res) => {
    const { goal } = req.body as AgentRunRequest;
    try {
      const run = await runAgent(goal);
      const result: AgentRunResponse = { run };
      res.json(result);
    } catch (error) {
      console.error('Error in /agent-run:', error);
      res.status(500).json({ error: 'Internal Server Error' });
    }
  });

// ── Health check — zero-cost endpoint for k6 load tests ─────────────────────
app.get('/health', (_req, res) => {
    res.json({
        status: 'ok',
        uptimeSeconds: Math.floor(process.uptime()),
        timestamp: new Date().toISOString(),
    });
});

app.listen(process.env.PORT || 3000, () => {
    console.log(`Server running on port ${process.env.PORT || 3000}`);
});

export default app;

