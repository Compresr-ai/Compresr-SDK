/** Public types returned by `ResearchAgent.run(...)`. */

export interface Citation {
  readonly url: string;
  readonly title?: string;
  readonly snippet?: string;
}

export type StepKind = 'search' | 'tool_result' | 'answer' | 'error';

export interface Step {
  readonly type: StepKind;
  readonly query?: string;
  readonly text?: string;
  readonly chars?: number;
  readonly latencyS?: number;
}

/** Token + tool-call counts across the research loop. */
export interface ResearchUsage {
  readonly input_tokens: number;
  readonly output_tokens: number;
  readonly cache_read_tokens: number;
  readonly cache_creation_tokens: number;
  readonly calls: number;
  readonly search_calls: number;
}

export interface ResearchResult {
  readonly answer: string;
  readonly explanation: string;
  readonly confidence: number | null;
  readonly text: string;
  readonly citations: ReadonlyArray<Citation>;
  readonly trajectory: ReadonlyArray<Step>;
  readonly usage: ResearchUsage;
  readonly raw: unknown;
}
