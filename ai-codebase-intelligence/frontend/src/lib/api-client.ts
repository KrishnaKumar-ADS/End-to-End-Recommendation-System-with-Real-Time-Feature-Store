import type {
  ApiErrorPayload,
  AskRequest,
  AskResponse,
  BugAnalysisRequest,
  BugAnalysisResponse,
  EvaluateItem,
  EvaluateResponse,
  ExplainRequest,
  ExplainResponse,
  DeleteRepositoryResponse,
  GraphData,
  IngestRequest,
  IngestResponse,
  MetricsResponse,
  ProvidersHealthResponse,
  RepositoryDetail,
  RepositorySummary,
  SearchParams,
  SearchResponse,
  SecurityAnalysisRequest,
  SecurityAnalysisResponse,
  StatusResponse,
} from "@/types/api";

const API_PREFIX = "/api/v1";

type Primitive = string | number | boolean | null | undefined;

export class ApiClientError extends Error {
  status: number;
  payload: ApiErrorPayload | null;

  constructor(message: string, status = 0, payload: ApiErrorPayload | null = null) {
    super(message);
    this.name = "ApiClientError";
    this.status = status;
    this.payload = payload;
  }
}

function getApiBaseUrl(): string {
  if (typeof window !== "undefined") {
    return "";
  }

  return (process.env.INTERNAL_API_URL || process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");
}

function toQueryString(query?: Record<string, Primitive>): string {
  if (!query) {
    return "";
  }

  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined || value === null || value === "") {
      continue;
    }
    params.set(key, String(value));
  }

  const serialized = params.toString();
  return serialized ? `?${serialized}` : "";
}

function buildUrl(path: string, query?: Record<string, Primitive>) {
  const queryString = toQueryString(query);
  const base = getApiBaseUrl();
  if (!base) {
    return `${API_PREFIX}${path}${queryString}`;
  }
  return `${base}${API_PREFIX}${path}${queryString}`;
}

function extractErrorMessage(payload: ApiErrorPayload | null, status: number): string {
  const detail = payload?.detail;
  const message = payload?.message;
  const error = payload?.error;

  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }
  if (typeof message === "string" && message.trim()) {
    return message;
  }
  if (typeof error === "string" && error.trim()) {
    return error;
  }

  return `Request failed with status ${status}.`;
}

async function parseJsonSafe(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text) as unknown;
  } catch {
    return { message: text };
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  query?: Record<string, Primitive>,
): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");

  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  let response: Response;
  try {
    response = await fetch(buildUrl(path, query), {
      ...init,
      headers,
      cache: "no-store",
    });
  } catch (error) {
    const message =
      error instanceof Error
        ? `Network error: ${error.message || "could not reach the backend."}`
        : "Network error: could not reach the backend.";
    throw new ApiClientError(message, 0, null);
  }

  const payload = (await parseJsonSafe(response)) as ApiErrorPayload | T | null;

  if (!response.ok) {
    throw new ApiClientError(
      extractErrorMessage(payload as ApiErrorPayload | null, response.status),
      response.status,
      (payload as ApiErrorPayload | null) ?? null,
    );
  }

  return payload as T;
}

export async function ingestRepository(payload: IngestRequest): Promise<IngestResponse> {
  return request<IngestResponse>("/ingest", {
    method: "POST",
    body: JSON.stringify({
      github_url: payload.github_url,
      branch: payload.branch ?? "main",
    }),
  });
}

export async function fetchStatus(taskId: string): Promise<StatusResponse> {
  return request<StatusResponse>(`/status/${taskId}`);
}

export async function fetchRepos(params?: {
  limit?: number;
  offset?: number;
}): Promise<RepositorySummary[]> {
  return request<RepositorySummary[]>("/repos", undefined, params);
}

export async function fetchRepo(repoId: string): Promise<RepositoryDetail> {
  return request<RepositoryDetail>(`/repos/${repoId}`);
}

export async function deleteRepository(repoId: string): Promise<DeleteRepositoryResponse> {
  return request<DeleteRepositoryResponse>(`/repos/${repoId}`, {
    method: "DELETE",
  });
}

export async function askQuestion(payload: AskRequest): Promise<AskResponse> {
  return request<AskResponse>("/ask", {
    method: "POST",
    body: JSON.stringify({
      ...payload,
      stream: false,
    }),
  });
}

export async function searchCode(params: SearchParams): Promise<SearchResponse> {
  return request<SearchResponse>("/search", undefined, {
    repo_id: params.repo_id,
    q: params.q,
    top_k: params.top_k ?? 8,
    mode: params.mode ?? "hybrid",
    rerank: params.rerank ?? true,
    expand_query: params.expand_query ?? false,
    chunk_type: params.chunk_type,
    language: params.language,
  });
}

export async function fetchGraph(
  repoId: string,
  params?: {
    limit?: number;
    node_type?: string;
    include_hierarchy?: boolean;
    depth?: number;
  },
): Promise<GraphData> {
  return request<GraphData>(`/graph/${repoId}`, undefined, params);
}

export async function fetchGraphSubgraph(
  repoId: string,
  startNodeId: string,
  depth: number,
  relTypes?: string[],
): Promise<GraphData> {
  const relTypeFilter = relTypes?.length ? relTypes.join(",") : undefined;
  return request<GraphData>(
    `/graph/${repoId}/subgraph/${startNodeId}`,
    undefined,
    {
      depth,
      rel_types: relTypeFilter,
    },
  );
}

export async function explainSymbol(payload: ExplainRequest): Promise<ExplainResponse> {
  return request<ExplainResponse>("/explain", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function analyzeBug(payload: BugAnalysisRequest): Promise<BugAnalysisResponse> {
  return request<BugAnalysisResponse>("/analyze/bug", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function analyzeSecurity(
  payload: SecurityAnalysisRequest,
): Promise<SecurityAnalysisResponse> {
  return request<SecurityAnalysisResponse>("/analyze/security", {
    method: "POST",
    body: JSON.stringify({
      file_filter: "",
      max_llm_calls: 10,
      ...payload,
    }),
  });
}

export async function fetchMetrics(): Promise<MetricsResponse> {
  return request<MetricsResponse>("/metrics");
}

export async function evaluateAnswers(items: EvaluateItem[]): Promise<EvaluateResponse> {
  return request<EvaluateResponse>("/evaluate", {
    method: "POST",
    body: JSON.stringify({ items }),
  });
}

export async function fetchProvidersHealth(): Promise<ProvidersHealthResponse> {
  return request<ProvidersHealthResponse>("/providers/health");
}

export async function streamAsk(
  payload: AskRequest & { session_id?: string | null },
  signal: AbortSignal,
): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(buildUrl("/ask"), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify({
        ...payload,
        stream: true,
      }),
      cache: "no-store",
      signal,
    });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw error;
    }

    const message =
      error instanceof Error
        ? `Network error: ${error.message || "could not reach the backend."}`
        : "Network error: could not reach the backend.";
    throw new ApiClientError(message, 0, null);
  }

  if (!response.ok) {
    const parsed = (await parseJsonSafe(response)) as ApiErrorPayload | null;
    throw new ApiClientError(
      extractErrorMessage(parsed, response.status),
      response.status,
      parsed,
    );
  }

  if (!response.body) {
    throw new ApiClientError("Response body is null; streaming is unavailable.", 500, null);
  }

  return response;
}
