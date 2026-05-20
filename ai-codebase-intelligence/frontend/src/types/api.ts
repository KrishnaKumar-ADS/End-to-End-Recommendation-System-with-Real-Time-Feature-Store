export type IngestionStatus =
  | "queued"
  | "cloning"
  | "scanning"
  | "parsing"
  | "embedding"
  | "completed"
  | "failed";

export interface IngestRequest {
  github_url: string;
  branch?: string;
}

export interface IngestResponse {
  repo_id: string;
  task_id: string;
  status: IngestionStatus | string;
  message: string;
}

export interface StatusResponse {
  task_id: string;
  status: IngestionStatus | string;
  progress: number;
  message: string;
  repo_id: string | null;
  error: string | null;
  total_files: number;
  processed_files: number;
  total_chunks: number;
}

export interface RepositorySummary {
  id: string;
  github_url: string;
  name: string;
  branch: string;
  status: IngestionStatus | string;
  task_id: string | null;
  error_message: string | null;
  total_files: number;
  processed_files: number;
  total_chunks: number;
  created_at: string;
  updated_at: string;
}

export interface RepositoryFile {
  id: string;
  file_path: string;
  language: string;
  size_bytes: number;
  line_count: number;
  chunk_count: number;
}

export interface RepositoryDetail extends RepositorySummary {
  files: RepositoryFile[];
}

export interface DeleteRepositoryResponse {
  repo_id: string;
  repo_name: string;
  deleted_files: number;
  deleted_chunks: number;
  deleted_vectors: number;
  deleted_graph_nodes: number;
  deleted_cache_keys: number;
  warnings: string[];
  message: string;
}

export interface AskRequest {
  repo_id: string;
  question: string;
  session_id?: string | null;
  task_type?: "code_qa" | "reasoning" | "security" | "summarize" | "architecture" | null;
  stream?: boolean;
  top_k?: number;
  include_graph?: boolean;
  language_filter?: string | null;
  chunk_type_filter?: string | null;
}

export interface SourceReference {
  file_path: string;
  function_name: string;
  start_line: number;
  end_line: number;
  score: number;
  chunk_type: string;
}

export interface QualityScore {
  faithfulness: number;
  relevance: number;
  completeness: number;
  overall: number;
  critique: string;
  eval_ms?: number;
  judge_model?: string;
  skipped?: boolean;
  skip_reason?: string;
}

export interface AskResponse {
  request_id: string;
  answer: string;
  session_id: string | null;
  provider_used: string;
  model_used: string;
  task_type: string;
  sources: SourceReference[];
  graph_path: string[];
  context_chunks_used: number;
  estimated_tokens: number;
  vector_search_ms: number;
  graph_expansion_ms: number;
  total_latency_ms: number;
  top_result_score: number;
  quality_score: QualityScore | null;
  cached: boolean;
  cache_similarity: number;
  intent: string | null;
}

export interface SearchParams {
  repo_id: string;
  q: string;
  top_k?: number;
  mode?: "vector" | "bm25" | "hybrid";
  rerank?: boolean;
  expand_query?: boolean;
  chunk_type?: string;
  language?: string;
}

export interface SearchTiming {
  embed_ms: number;
  expand_ms: number;
  vector_ms: number;
  bm25_ms: number;
  fusion_ms: number;
  rerank_ms: number;
  total_ms: number;
}

export interface SearchResult {
  id: string;
  name: string;
  file_path: string;
  chunk_type: string;
  start_line: number;
  end_line: number;
  content: string;
  docstring: string | null;
  language: string | null;
  parent_name: string | null;
  vector_score: number;
  bm25_score: number;
  hybrid_score: number;
  rerank_score: number | null;
  vector_rank: number | null;
  bm25_rank: number | null;
  hybrid_rank: number;
  final_rank: number;
}

export interface SearchResponse {
  query: string;
  expanded_queries: string[];
  repo_id: string;
  mode: string;
  reranked: boolean;
  results: SearchResult[];
  total_results: number;
  timing: SearchTiming;
}

export interface GraphNode {
  id: string;
  _type?: string;
  label?: string;
  name?: string;
  display_name?: string;
  file?: string;
  file_path?: string;
  path?: string;
  start_line?: number;
  end_line?: number;
  description?: string;
  [key: string]: unknown;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: string;
  id?: string;
}

export interface GraphResponse {
  repo_id: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  node_count: number;
  edge_count: number;
  centre_node_id?: string;
  depth?: number;
}

export type GraphData = GraphResponse;

export type GraphNodeType = "function" | "class" | "module" | "file";

export interface GraphFilterState {
  nodeTypes: Record<GraphNodeType, boolean>;
  search: string;
  showIsolated: boolean;
  minDegree: number;
  depth: number;
}

export interface GraphNeighborInfo {
  callers: string[];
  callees: string[];
}

export interface ExplainRequest {
  repo_id: string;
  function_name?: string | null;
  file_path?: string | null;
  chunk_id?: string | null;
  max_callers?: number;
  max_callees?: number;
}

export interface ExplainParameterInfo {
  name: string;
  type_annotation: string | null;
  default_value: string | null;
  description: string | null;
}

export interface ExplainReturnInfo {
  type_annotation: string | null;
  description: string;
}

export interface ExplainCallerCalleeInfo {
  function_name: string;
  file_path: string;
  node_id: string;
}

export interface ExplainResponse {
  function_name: string;
  file_path: string;
  start_line: number;
  end_line: number;
  summary: string;
  parameters: ExplainParameterInfo[];
  returns: ExplainReturnInfo;
  side_effects: string[];
  callers: ExplainCallerCalleeInfo[];
  callees: ExplainCallerCalleeInfo[];
  complexity_score: number;
  provider_used: string;
  model_used: string;
  explanation_ms: number;
}

export interface BugAnalysisRequest {
  repo_id: string;
  error_description: string;
  max_hops?: number;
}

export interface BugAnalysisResponse {
  error_signal: string;
  call_chain: string[];
  callers: string[];
  callees: string[];
  root_cause_file: string;
  root_cause_function: string;
  root_cause_line: number | null;
  explanation: string;
  fix_suggestion: string;
  confidence: string;
  provider_used: string;
  model_used: string;
  graph_nodes_explored: number;
}

export interface SecurityAnalysisRequest {
  repo_id: string;
  file_filter?: string;
  max_llm_calls?: number;
}

export interface SecurityFinding {
  file_path: string;
  function: string;
  line_number: number;
  severity: string;
  category: string;
  description: string;
  matched_text: string;
  rule_id: string;
  cwe_id: string;
  llm_analysis: string;
  false_positive: boolean;
}

export interface SecurityAnalysisResponse {
  repo_id: string;
  findings: SecurityFinding[];
  false_positives_removed: number;
  summary_stats: Record<string, number>;
  scan_duration_ms: number;
  static_findings_count: number;
  chunks_scanned: number;
  file_filter: string;
  provider_used: string;
  model_used: string;
}

export interface ProviderHealthEntry {
  state: string;
  consecutive_failures: number;
  api_key_configured: boolean;
  recovery_in_seconds?: number;
  last_failure_ago_seconds?: number;
  last_success_ago_seconds?: number;
}

export interface ProvidersHealthResponse {
  providers: Record<string, ProviderHealthEntry>;
  circuit_breaker: {
    failure_threshold: number;
    recovery_timeout_sec: number;
  };
}

export interface MetricsBudget {
  daily_limit_usd: number;
  used_today_usd: number;
  remaining_usd: number;
  used_pct: number;
  over_budget: boolean;
}

export interface MetricsResponse {
  token_usage: Record<string, Record<string, number>>;
  budget: MetricsBudget;
  cache: Record<string, unknown>;
  circuit_breakers: Record<string, Record<string, unknown>>;
  eval_scores: Record<string, number>;
}

export interface EvaluateItem {
  question: string;
  answer: string;
  context_chunks?: string[];
  item_id?: string | null;
}

export interface EvaluateResult {
  item_id: string;
  question: string;
  score: QualityScore;
}

export interface EvaluateAggregate {
  avg_faithfulness: number;
  avg_relevance: number;
  avg_completeness: number;
  avg_overall: number;
  total_items: number;
  evaluated_items: number;
  skipped_items: number;
  failed_items: number;
  total_eval_ms: number;
}

export interface EvaluateResponse {
  results: EvaluateResult[];
  aggregate: EvaluateAggregate;
}

export interface ApiErrorPayload {
  detail?: string;
  message?: string;
  error?: string;
  [key: string]: unknown;
}

export interface FileTreeNode {
  id: string;
  name: string;
  path: string;
  type: "file" | "dir";
  children?: FileTreeNode[];
  file?: RepositoryFile;
}

export interface ToastItem {
  id: string;
  title: string;
  description?: string;
  variant?: "info" | "success" | "warning" | "error";
}

// Week 12 additions: streaming chat and session-aware conversation types.

export interface SourceCitation {
  file: string;
  function: string;
  lines: string;
  snippet?: string;
}

export interface TokenEvent {
  type: "token";
  content: string;
}

export interface SourcesEvent {
  type: "sources";
  sources: SourceCitation[];
}

export interface DoneEvent {
  type: "done";
  session_id: string | null;
  quality_score: QualityScore | null;
  graph_path: string[];
  model_used?: string;
  provider_used?: string;
  model?: string;
  provider?: string;
  cached?: boolean;
  total_ms?: number;
  timing?: {
    search_ms: number;
    graph_ms: number;
    context_ms: number;
    total_ms: number;
  };
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export interface StepEvent {
  type: "step";
  stage: string;
  message: string;
}

export type ChatEvent = TokenEvent | SourcesEvent | DoneEvent | ErrorEvent | StepEvent;

export interface UserMessage {
  id: string;
  role: "user";
  content: string;
  timestamp: number;
}

export interface AssistantMessage {
  id: string;
  role: "assistant";
  content: string;
  sources: SourceCitation[];
  qualityScore: QualityScore | null;
  graphPath: string[];
  providerUsed: string | null;
  modelUsed: string | null;
  cached: boolean;
  totalMs: number | null;
  isStreaming: boolean;
  timestamp: number;
}

export type ChatMessage = UserMessage | AssistantMessage;

export interface ChatSession {
  sessionId: string;
  repoId: string;
  startedAt: number;
  messageCount: number;
}
