import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  ApiClientError,
  askQuestion,
  fetchRepo,
  fetchRepos,
  fetchStatus,
  ingestRepository,
} from "@/lib/api-client";

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("api-client", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.stubGlobal("fetch", vi.fn());
  });

  it("ingestRepository sends POST /api/v1/ingest with the expected body", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({ repo_id: "r1", task_id: "t1", status: "queued", message: "ok" }),
    );

    await ingestRepository({ github_url: "https://github.com/org/repo", branch: "dev" });

    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/ingest",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          github_url: "https://github.com/org/repo",
          branch: "dev",
        }),
      }),
    );
  });

  it("fetchStatus calls the task status endpoint", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({ task_id: "t1", status: "queued", progress: 0, message: "", repo_id: null, error: null, total_files: 0, processed_files: 0, total_chunks: 0 }),
    );

    await fetchStatus("task-123");

    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/status/task-123",
      expect.objectContaining({ cache: "no-store" }),
    );
  });

  it("fetchRepos serializes query parameters", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse([]));

    await fetchRepos({ limit: 20, offset: 10 });

    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/repos?limit=20&offset=10",
      expect.objectContaining({ cache: "no-store" }),
    );
  });

  it("fetchRepo calls the repository detail endpoint", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({ id: "r1", github_url: "", name: "repo", branch: "main", status: "completed", task_id: null, error_message: null, total_files: 0, processed_files: 0, total_chunks: 0, created_at: "", updated_at: "", files: [] }));

    await fetchRepo("repo-123");

    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/repos/repo-123",
      expect.objectContaining({ cache: "no-store" }),
    );
  });

  it("askQuestion always sends stream=false for Week 11", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({
        request_id: "a1",
        answer: "Hello",
        session_id: null,
        provider_used: "openrouter",
        model_used: "qwen",
        task_type: "code_qa",
        sources: [],
        graph_path: [],
        context_chunks_used: 0,
        estimated_tokens: 0,
        vector_search_ms: 0,
        graph_expansion_ms: 0,
        total_latency_ms: 0,
        top_result_score: 0,
        quality_score: null,
        cached: false,
        cache_similarity: 0,
        intent: null,
      }),
    );

    await askQuestion({ repo_id: "r1", question: "How does auth work?" });

    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/ask",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          repo_id: "r1",
          question: "How does auth work?",
          stream: false,
        }),
      }),
    );
  });

  it("throws ApiClientError with the response status on HTTP errors", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({ detail: "Repository already ingested." }, 409),
    );

    await expect(
      ingestRepository({ github_url: "https://github.com/org/repo" }),
    ).rejects.toMatchObject({
      name: "ApiClientError",
      status: 409,
      message: "Repository already ingested.",
    });
  });

  it("wraps fetch failures in ApiClientError with status 0", async () => {
    vi.mocked(fetch).mockRejectedValueOnce(new Error("connection refused"));

    await expect(fetchRepos()).rejects.toMatchObject({
      name: "ApiClientError",
      status: 0,
    });
  });
});
