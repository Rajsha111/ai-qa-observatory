# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**AI QA Observatory** — an open-source TypeScript framework for testing every layer of AI systems. Clone it, point it at your own AI app, and get a full test suite covering UI, prompt quality, RAG pipelines, MCP tools, and agent orchestration.

The framework ships with a reference target app (an Express + Claude chat server) that all test layers run against. Swap in your own app URL and the test modules work against your system.

The 5 test layers are fully independent — use any one without the others:

| Layer | Tool | Location |
|---|---|---|
| E2E / UI | Playwright | `tests/e2e/` |
| Prompt Evaluation | Promptfoo + semantic scorer | `tests/eval/` |
| RAG Validation | Python microservice (LlamaIndex + Chroma + Ragas) | `tests/rag/` |
| MCP Tool Testing | MCP TypeScript SDK | `tests/mcp/` |
| Agent Orchestration | Custom ReAct loop | `tests/agent/` |

## Commands

```bash
npm run dev          # Start the target app → http://localhost:3000
npm run typecheck    # Type-check without emitting
npm run build        # Compile TypeScript → dist/
npm run test         # Run all Vitest tests (eval, rag, mcp, agent layers)
npm run test:e2e     # Run Playwright E2E tests (requires target app running)
```

The RAG layer requires the Python microservice running separately:
```bash
cd tests/rag/service && pip install llama-index chromadb ragas flask
python server.py     # Starts on port 8001
```

## Repo Architecture

```
src/
  shared/
    types.ts          # Single source of truth for all types — every layer imports from here
    utils.ts          # timer(), createEvalResult(), sleep(), formatScore(), assertInRange()
  target-app/
    server.ts         # Express: POST /chat (Claude), POST /rag-query, POST /agent-run
    public/
      index.html      # Vanilla JS chat UI — the E2E test surface
  mcp/
    analyze-quality-tool.ts  # MCP tool: analyze_response_quality (LLM-as-Judge)

tests/
  e2e/
    chat.spec.ts              # Happy path, streaming wait, error states, API contract
    pages/ChatPage.ts         # Page Object Model — tests never touch raw selectors
    utils/streamWait.ts       # waitForStreamComplete() — polling utility for streaming AI responses
  eval/
    promptfoo.yaml            # 20 test cases: accuracy, hallucination, tone, edge, format
    semantic-scorer.ts        # Cosine similarity scorer via text-embedding-3-small
    drift-detector.ts         # Run same prompt N times, flag if variance > threshold
  rag/
    faithfulness.test.ts      # Does answer stay grounded in retrieved chunks?
    precision.test.ts         # Are retrieved chunks actually relevant? (Context Precision)
    negative.test.ts          # Questions outside KB — must not hallucinate an answer
    retrieval-unit.test.ts    # Known query must always retrieve specific doc IDs
    stale-data.test.ts        # After re-ingestion, response must reflect updated facts
  mcp/
    schema.test.ts            # Tool inputSchema matches TypeScript type contract
    tool-behavior.test.ts     # Boundary inputs, error propagation inside the tool
    llm-in-loop.test.ts       # Claude as MCP client — validates LLM-as-Judge end-to-end
  agent/
    tool-audit.test.ts        # Assert only expected tools are called for a given input
    loop-guard.test.ts        # Agent must halt at max iterations, not loop forever
    bad-tool.test.ts          # Mock a tool to fail — agent must recover, not crash
    completion-rate.test.ts   # Run same task 5x, assert ≥ 4/5 produce a valid report
  load/
    api-load.js               # k6: 5 VUs, 60s, measures TTFT + p95/p99 latency

.github/workflows/
  ci.yml              # CI jobs: e2e, eval, rag, mcp. Agent + k6 as additional jobs.
```

## Key Architectural Decisions

**Shared type system**: All 5 test layers import exclusively from `src/shared/types.ts`. Never define types inside a test module. `EvalResult<T>` is the universal return type across all layers.

**RAG layer is a Python microservice**: LlamaIndex and Ragas are Python-native with no TypeScript equivalents. The service runs on port 8001 (Flask) and the TypeScript tests call it via HTTP — same pattern as calling any real RAG service in production. The Express `/rag-query` endpoint proxies to it.

**Agent is a custom ReAct loop**: Intentionally not LangGraph — keeping internals visible. Loop: reason → act → observe → repeat. Hard limit of 5 iterations. 3 available tools: `run_e2e_tests`, `run_eval_tests`, `run_rag_tests`.

**MCP tool — `analyze_response_quality`**: Takes `{llm_response, ground_truth, criteria[]}`, uses Claude internally as a judge, returns a structured quality score with per-criterion breakdown. This is the LLM-as-Judge implementation.

**Metrics-first design**: Every test produces a numeric score (0–1), not just pass/fail. CI quality gates: eval pass rate ≥ 80%, faithfulness ≥ 0.85, LLM-judge accuracy ≥ 0.80. Scores are saved as CI artifacts for regression tracking.

**Playwright streaming pattern**: All AI response assertions go through `waitForStreamComplete()` in `tests/e2e/utils/streamWait.ts`. Never use raw `waitForSelector` on an AI response — it resolves before streaming finishes. Playwright `timeout` is set to 60000ms globally.

## Path Aliases

`@shared/*` → `src/shared/*`
`@tests/*` → `tests/*`

Resolved at runtime via `ts-node -r tsconfig-paths/register`. Use these aliases for all cross-module imports.

## Environment

Copy `.env.example` → `.env`. Required variables:
- `ANTHROPIC_API_KEY` — target app `/chat` endpoint and MCP tool's internal Claude judge
- `OPENAI_API_KEY` — semantic scorer (`text-embedding-3-small`)
- `PORT` — target app port, defaults to 3000

## Target App Endpoints

| Endpoint | What it does |
|---|---|
| `POST /chat` | Calls `claude-sonnet-4-6`, returns `ChatResponse` with `tokenUsage` |
| `POST /rag-query` | Proxies to Python RAG service (stub until service is running) |
| `POST /agent-run` | Runs the ReAct agent loop, returns full `AgentRun` trace (stub until agent is built) |
