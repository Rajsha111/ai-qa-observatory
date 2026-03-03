// Core eval framework Types
export interface EvalResult<T = unknown> {
  passed: boolean;
  score: number;
  expected: T;
  actual: T;
  reasoning?: string;
  latencyMs?: number;
  metadata?: Record<string, unknown>;
}

export interface TestCase<TInput = unknown, TExpected = unknown> {
    id: string;
    description: string;
    input: TInput;
    expected: TExpected;
    tags?: string[];
}

export interface RagTestCase extends TestCase<string, string> {
    groundTruth: string;
    relevantChunkIds?: string[];
    minRelevantScore?: number;
}

export interface ReterivedChunk {
    id: string;
    content: string;
    score: number;
    source?: string;
}

export interface AgentStep {
    stepIndex?: number;
    thought: string;
    toolCalled?: string;
    toolInput?: string;
    toolOutput?: string;
    observation?: string;
}

export interface AgentRun{
    runId: string;
    goal: string;
    steps: AgentStep[];
    finalAnswer?: string;
    totalLatencyMs?: number;
    tokenuse?: number;
    success?: boolean;
}

export interface MCPToolCall {
    toolName: string;
    input: string;
    output: string;
    durantionMs: number;
    error?: string;


}

export interface TokenUsage {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
}

export interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
}

export interface ChatRequest {
    messages: ChatMessage[];
    systemPrompt?: string;
}

export interface ChatResponse {
    message: string;
    latencyMs: number;
    tokenUsage?: TokenUsage;
}

export interface RAGQueryRequest {
    query: string;
    topK?: number;
}

export interface RAGQueryResponse {
    retrievedChunks: ReterivedChunk[];
    latencyMs: number;
    answer: string;
}

export interface AgentRunRequest {
    goal: string;
    context?: Record<string, unknown>;
}

export interface AgentRunResponse {
    run : AgentRun;
}