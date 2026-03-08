import { execSync } from 'child_process';
import path from 'path';

export interface ToolResult {
    tool: string;
    exitCode: number;
    output: string;
    passed: boolean;
}

const ROOT = path.join(__dirname, '..', '..');

function runViTest(testPath: string, toolName: string): ToolResult {
    try {
      const output = execSync(
        `npx vitest run ${testPath} --reporter=verbose --fileParallelism=false`,
        { cwd: ROOT, encoding: 'utf-8', timeout: 120_000 }
      );
      return { tool: toolName, exitCode: 0, output, passed: true };
    } catch (err: any) {
      return {
        tool: toolName,
        exitCode: err.status ?? 1,
        output: err.stdout ?? err.message ?? 'unknown error',
        passed: false,
      };
    }
}

export function run_e2e_tests(): ToolResult {
    return runViTest('tests/e2e', 'run_e2e_tests');
}

export function run_eval_tests(): ToolResult{
    return runViTest('tests/eval', 'run_eval_tests');
}

export function run_rag_tests(): ToolResult {
    return runViTest('tests/rag', 'run_rag_tests')
}

export type ToolName = 'run_e2e_tests' | 'run_eval_tests' | 'run_rag_tests';

  export const TOOL_MAP: Record<ToolName, () => ToolResult> = {
    run_e2e_tests,
    run_eval_tests,
    run_rag_tests,
  };