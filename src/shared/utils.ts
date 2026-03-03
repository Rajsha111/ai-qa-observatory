import type { EvalResult } from './types';

  /** Returns a function that, when called, gives elapsed ms since creation */
  export function timer(): () => number {
    const start = Date.now();
    return () => Date.now() - start;
  }

  export function sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /** Formats a 0–1 score as a percentage string */
  export function formatScore(score: number): string {
    return `${(score * 100).toFixed(1)}%`;
  }

  export function assertInRange(value: number, min: number, max: number): boolean {
    return value >= min && value <= max;
  }