


import { existsSync, readFileSync, writeFileSync } from 'fs';
import { join } from 'path';

const ROOT      = join(__dirname, '..', '..');
const ARTIFACTS = join(__dirname, 'artifacts');

// ── Types ──────────────────────────────────────────────────────────────────────

interface VitestResult {
  success: boolean;
  numTotalTests: number;
  numPassedTests: number;
  numFailedTests: number;
  testResults: Array<{
    testFilePath: string;
    status: string;
    assertionResults: Array<{
      title: string;
      fullName: string;
      status: 'passed' | 'failed';
      duration: number | null;
      failureMessages: string[];
    }>;
  }>;
}

interface ToolArtifact {
  label: string;
  input: Record<string, unknown>;
  output: unknown;
  isError: boolean;
}

interface LoopArtifact {
  label: string;
  userMessage: string;
  toolUseInput: Record<string, unknown>;
  toolResult: {
    overall: number;
    breakdown: { criterion: string; score: number; reason: string }[];
  };
}

// ── Read helpers ───────────────────────────────────────────────────────────────

function readJSON<T>(path: string): T | null {
  if (!existsSync(path)) return null;
  return JSON.parse(readFileSync(path, 'utf-8')) as T;
}

// ── HTML helpers ───────────────────────────────────────────────────────────────

function esc(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function badge(passed: boolean): string {
  return passed
    ? `<span class="badge pass">PASS</span>`
    : `<span class="badge fail">FAIL</span>`;
}

function scoreBar(value: number): string {
  const pct   = Math.round(value * 100);
  const color = value >= 0.7 ? '#22c55e' : value >= 0.4 ? '#f59e0b' : '#ef4444';
  return `<div class="score-bar-wrap">
    <div class="score-bar" style="width:${pct}%;background:${color};"></div>
    <span class="score-label">${value.toFixed(2)}</span>
  </div>`;
}

function jsonBlock(obj: unknown): string {
  return `<pre class="json-block">${esc(JSON.stringify(obj, null, 2))}</pre>`;
}

// ── Section: Schema contract ───────────────────────────────────────────────────

function renderSchemaSection(
  vitestFile: VitestResult['testResults'][0] | undefined,
  schemaArt:  { schema: unknown } | null
): string {
  if (!vitestFile) return missingSection('Schema Contract');

  const tests  = vitestFile.assertionResults;
  const passed = tests.filter(t => t.status === 'passed').length;

  const rows = tests.map(t => `
    <tr class="${t.status === 'passed' ? 'row-pass' : 'row-fail'}">
      <td class="mono">${esc(t.title)}</td>
      <td class="center">${t.duration != null ? t.duration + 'ms' : '—'}</td>
      <td class="center">${badge(t.status === 'passed')}</td>
      <td class="error-text">${t.failureMessages[0] ? esc(t.failureMessages[0]) : '—'}</td>
    </tr>`).join('');

  const schemaBlock = schemaArt
    ? `<details><summary>Schema object (exported from analyze-quality-tool.ts)</summary>${jsonBlock(schemaArt.schema)}</details>`
    : '';

  return `
  <section>
    <div class="section-header">
      <div>
        <h2>Schema Contract</h2>
        <p class="section-desc">Asserts <code>analyzeQualityInputSchema</code> has the correct shape.
        No server boot, no network — pure property assertions.
        If a field is renamed or retyped, these fail before any integration test runs.</p>
      </div>
      <span class="section-stats">${passed}/${tests.length} passed</span>
    </div>
    <table class="data-table">
      <thead><tr><th>Assertion</th><th>Duration</th><th>Result</th><th>Error</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    ${schemaBlock}
  </section>`;
}

// ── Section: Tool behavior ─────────────────────────────────────────────────────

function renderToolBehaviorSection(
  vitestFile: VitestResult['testResults'][0] | undefined,
  artifacts:  ToolArtifact[] | null
): string {
  if (!vitestFile) return missingSection('Tool Behavior');

  const tests  = vitestFile.assertionResults;
  const passed = tests.filter(t => t.status === 'passed').length;

  const cards = tests.map((t, i) => {
    const art = artifacts?.[i];
    return `
    <div class="card ${t.status === 'passed' ? 'pass-card' : 'fail-card'}">
      <div class="card-header">
        ${badge(t.status === 'passed')}
        <span class="test-index">#${i + 1}</span>
        <span class="test-desc">${esc(t.title)}</span>
        ${t.duration != null ? `<span class="duration">${t.duration}ms</span>` : ''}
      </div>
      ${t.failureMessages[0] ? `<div class="error-box">${esc(t.failureMessages[0])}</div>` : ''}
      ${art ? `
      <div class="card-body">
        <p class="section-desc">Sent via <code>InMemoryTransport</code> → MCP Server handler → mock judge</p>
        <div class="two-col">
          <div>
            <div class="field-label">MCP Input</div>
            ${jsonBlock(art.input)}
          </div>
          <div>
            <div class="field-label">MCP Output</div>
            ${jsonBlock(art.output)}
            ${art.isError ? `<div class="error-tag">isError: true</div>` : ''}
          </div>
        </div>
      </div>` : ''}
    </div>`;
  }).join('');

  return `
  <section>
    <div class="section-header">
      <div>
        <h2>Tool Behavior</h2>
        <p class="section-desc">Each test wires MCP Client → Server via <code>InMemoryTransport</code>.
        Judge is a <code>vi.fn()</code> mock — isolates the MCP protocol layer from Claude's judgment.</p>
      </div>
      <span class="section-stats">${passed}/${tests.length} passed</span>
    </div>
    ${cards}
  </section>`;
}

// ── Section: LLM in loop ───────────────────────────────────────────────────────

function renderLLMLoopSection(
  vitestFile: VitestResult['testResults'][0] | undefined,
  artifacts:  LoopArtifact[] | null
): string {
  if (!vitestFile) return missingSection('LLM in Loop');

  const tests  = vitestFile.assertionResults;
  const passed = tests.filter(t => t.status === 'passed').length;

  const cards = tests.map((t, i) => {
    const art = artifacts?.[i];
    return `
    <div class="card ${t.status === 'passed' ? 'pass-card' : 'fail-card'}">
      <div class="card-header">
        ${badge(t.status === 'passed')}
        <span class="test-index">#${i + 1}</span>
        <span class="test-desc">${esc(t.title)}</span>
        ${t.duration != null ? `<span class="duration">${Math.round(t.duration / 1000)}s</span>` : ''}
      </div>
      ${t.failureMessages[0] ? `<div class="error-box">${esc(t.failureMessages[0])}</div>` : ''}
      ${art ? `
      <div class="card-body">

        <div class="flow-step">
          <div class="step-label">① User message → Claude (Anthropic API)</div>
          <pre class="msg-block">${esc(art.userMessage)}</pre>
        </div>

        <div class="flow-arrow">↓ Claude reads tool list from MCP server · decides to call analyze_response_quality · generates arguments</div>

        <div class="flow-step">
          <div class="step-label">② Claude's tool_use input (Claude generated this — not hardcoded)</div>
          ${jsonBlock(art.toolUseInput)}
        </div>

        <div class="flow-arrow">↓ Input travels: MCP Client → InMemoryTransport → MCP Server → callClaudeJudge → back through transport</div>

        <div class="flow-step">
          <div class="step-label">③ Tool result returned to Claude via MCP</div>
          <div class="overall-row" style="margin-bottom:10px;">
            <span class="field-label">Overall&nbsp;</span>
            ${scoreBar(art.toolResult.overall)}
          </div>
          <table class="data-table">
            <thead><tr><th>Criterion</th><th>Score</th><th>Claude's Reasoning</th></tr></thead>
            <tbody>
              ${art.toolResult.breakdown.map(b => `
              <tr>
                <td class="mono">${esc(b.criterion)}</td>
                <td>${scoreBar(b.score)}</td>
                <td class="reason-text">${esc(b.reason)}</td>
              </tr>`).join('')}
            </tbody>
          </table>
        </div>

      </div>` : ''}
    </div>`;
  }).join('');

  return `
  <section>
    <div class="section-header">
      <div>
        <h2>LLM in Loop</h2>
        <p class="section-desc">Claude acts as the MCP client. It receives the tool list, decides to call
        <code>analyze_response_quality</code>, and generates the input arguments itself.
        Full trace: user message → Claude's tool_use → MCP transport → score breakdown.</p>
      </div>
      <span class="section-stats">${passed}/${tests.length} passed</span>
    </div>
    ${cards}
  </section>`;
}

function missingSection(name: string): string {
  return `<section><h2>${name}</h2><p class="error-box">No vitest results found. Run <code>npm run test:mcp:report</code> first.</p></section>`;
}

// ── Build full HTML ────────────────────────────────────────────────────────────

function buildHTML(
  vitest:    VitestResult,
  schemaArt: { schema: unknown } | null,
  toolArt:   ToolArtifact[] | null,
  loopArt:   LoopArtifact[] | null,
  generatedAt: string
): string {
  const schemaFile   = vitest.testResults.find(r => r.testFilePath.includes('schema'));
  const behaviorFile = vitest.testResults.find(r => r.testFilePath.includes('tool-behavior'));
  const loopFile     = vitest.testResults.find(r => r.testFilePath.includes('llm-in-loop'));

  const passRate = Math.round((vitest.numPassedTests / vitest.numTotalTests) * 100);

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MCP Test Report — AI QA Observatory</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 14px; line-height: 1.5; background: #f1f5f9; color: #1e293b; }
    code, .mono { font-family: "SF Mono", Menlo, monospace; font-size: 12px; background: #e2e8f0; padding: 1px 5px; border-radius: 3px; }
    header { background: #0f172a; color: #f8fafc; padding: 28px 40px; }
    header h1 { font-size: 20px; font-weight: 700; margin-bottom: 2px; }
    .subtitle { font-size: 13px; color: #64748b; margin-bottom: 20px; }
    .meta { font-size: 11px; color: #475569; margin-bottom: 20px; }
    .summary-row { display: flex; align-items: center; gap: 24px; }
    .stat .num { font-size: 32px; font-weight: 800; line-height: 1; }
    .stat .lbl { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: .05em; }
    .stat.pass .num { color: #22c55e; }
    .stat.fail .num { color: #ef4444; }
    .divider { width: 1px; height: 40px; background: #1e293b; }
    .pass-ring { width: 72px; height: 72px; border-radius: 50%; background: conic-gradient(#22c55e ${passRate * 3.6}deg, #1e293b 0); display: flex; align-items: center; justify-content: center; font-size: 15px; font-weight: 700; }
    main { max-width: 1020px; margin: 0 auto; padding: 28px 20px; }
    section { margin-bottom: 40px; }
    .section-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px; gap: 12px; }
    .section-header h2 { font-size: 17px; font-weight: 700; margin-bottom: 4px; }
    .section-desc { font-size: 13px; color: #64748b; max-width: 700px; }
    .section-stats { font-size: 12px; font-weight: 600; background: #e2e8f0; padding: 3px 10px; border-radius: 10px; color: #475569; white-space: nowrap; }
    .card { background: #fff; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 14px; border-left: 4px solid #e2e8f0; overflow: hidden; }
    .pass-card { border-left-color: #22c55e; }
    .fail-card { border-left-color: #ef4444; }
    .card-header { display: flex; align-items: center; gap: 10px; padding: 12px 16px; background: #f8fafc; border-bottom: 1px solid #e2e8f0; }
    .test-index { font-size: 11px; color: #94a3b8; font-weight: 600; }
    .test-desc { font-weight: 600; font-size: 13px; flex: 1; }
    .duration { font-size: 11px; color: #94a3b8; }
    .card-body { padding: 16px; display: flex; flex-direction: column; gap: 14px; }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .json-block, .msg-block { font-family: "SF Mono", Menlo, monospace; font-size: 12px; background: #0f172a; color: #e2e8f0; padding: 12px 14px; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; line-height: 1.6; margin-top: 6px; }
    .msg-block { background: #1e293b; }
    .flow-step { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 14px; }
    .step-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; color: #6366f1; margin-bottom: 6px; }
    .flow-arrow { font-size: 12px; color: #94a3b8; padding: 4px 14px; font-style: italic; }
    .score-bar-wrap { display: flex; align-items: center; gap: 8px; }
    .score-bar { height: 10px; border-radius: 5px; min-width: 2px; max-width: 200px; }
    .score-label { font-size: 13px; font-weight: 700; }
    .overall-row { display: flex; align-items: center; gap: 10px; }
    .data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .data-table th { background: #f1f5f9; padding: 8px 12px; text-align: left; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; color: #64748b; border-bottom: 1px solid #e2e8f0; }
    .data-table td { padding: 9px 12px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }
    .data-table tr:last-child td { border-bottom: none; }
    .row-pass td { background: #f0fdf4; }
    .row-fail td { background: #fef2f2; }
    .reason-text { font-size: 12px; color: #475569; line-height: 1.5; }
    .center { text-align: center; }
    .error-text { font-size: 11px; color: #b91c1c; font-family: monospace; max-width: 300px; }
    .badge { display: inline-block; font-size: 10px; font-weight: 700; letter-spacing: .06em; padding: 2px 8px; border-radius: 4px; text-transform: uppercase; }
    .badge.pass { background: #dcfce7; color: #15803d; }
    .badge.fail { background: #fee2e2; color: #b91c1c; }
    .field-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: #64748b; margin-bottom: 2px; }
    .error-box { background: #fef2f2; border: 1px solid #fecaca; padding: 12px 16px; color: #b91c1c; font-size: 12px; }
    .error-tag { display: inline-block; background: #fef2f2; color: #b91c1c; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px; margin-top: 6px; }
    details { margin-top: 12px; }
    details summary { cursor: pointer; font-size: 12px; font-weight: 600; color: #6366f1; padding: 4px 0; }
  </style>
</head>
<body>
  <header>
    <h1>MCP Test Report — <code style="font-size:16px;background:rgba(255,255,255,.12);color:#f8fafc;">analyze_response_quality</code></h1>
    <div class="subtitle">AI QA Observatory · Schema contract · Tool behavior via InMemoryTransport · LLM in loop</div>
    <div class="meta">Generated: ${generatedAt}</div>
    <div class="summary-row">
      <div class="stat pass"><div class="num">${vitest.numPassedTests}</div><div class="lbl">Passed</div></div>
      <div class="divider"></div>
      <div class="stat fail"><div class="num">${vitest.numFailedTests}</div><div class="lbl">Failed</div></div>
      <div class="divider"></div>
      <div class="pass-ring">${passRate}%</div>
    </div>
  </header>
  <main>
    ${renderSchemaSection(schemaFile, schemaArt)}
    ${renderToolBehaviorSection(behaviorFile, toolArt)}
    ${renderLLMLoopSection(loopFile, loopArt)}
  </main>
</body>
</html>`;
}

// ── Main ───────────────────────────────────────────────────────────────────────

function main() {
  const vitestPath = join(ROOT, 'mcp-test-results.json');
  if (!existsSync(vitestPath)) {
    console.error('mcp-test-results.json not found. Run: npm run test:mcp:report');
    process.exit(1);
  }

  const vitest    = readJSON<VitestResult>(vitestPath)!;
  const schemaArt = readJSON<{ schema: unknown }>(join(ARTIFACTS, 'schema.json'));
  const toolArt   = readJSON<ToolArtifact[]>(join(ARTIFACTS, 'tool-behavior.json'));
  const loopArt   = readJSON<LoopArtifact[]>(join(ARTIFACTS, 'llm-loop.json'));

  const generatedAt = new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
  const html        = buildHTML(vitest, schemaArt, toolArt, loopArt, generatedAt);
  const outPath     = join(ROOT, 'mcp-report.html');

  writeFileSync(outPath, html, 'utf-8');
  console.log(`\nReport: ${outPath}`);
  console.log(`${vitest.numPassedTests}/${vitest.numTotalTests} tests passed\n`);
}

main();
