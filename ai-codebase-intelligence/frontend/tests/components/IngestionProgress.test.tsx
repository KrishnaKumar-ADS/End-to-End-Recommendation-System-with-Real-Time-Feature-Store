import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { IngestionProgress } from "@/components/ingest/IngestionProgress";
import type { StatusResponse } from "@/types/api";

function makeStatus(overrides: Partial<StatusResponse> = {}): StatusResponse {
  return {
    task_id: "task-1",
    status: "queued",
    progress: 0,
    message: "Queued",
    repo_id: "repo-1",
    error: null,
    total_files: 312,
    processed_files: 120,
    total_chunks: 2847,
    ...overrides,
  };
}

describe("IngestionProgress", () => {
  it("renders nothing when status is null", () => {
    const { container } = render(
      <IngestionProgress elapsedSec={0} onRetry={vi.fn()} status={null} />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("shows guide-aligned default provider during active ingestion", () => {
    render(
      <IngestionProgress
        elapsedSec={23}
        onRetry={vi.fn()}
        status={makeStatus({
          status: "embedding",
          progress: 42,
          message: "Embedding 1200/2847 chunks...",
        })}
      />,
    );

    expect(screen.getByText("Ingestion in progress")).toBeInTheDocument();
    expect(screen.getByText("Provider:")).toBeInTheDocument();
    expect(
      screen.getByText("qwen/qwen-2.5-coder-32b-instruct via OpenRouter"),
    ).toBeInTheDocument();
    expect(screen.getByText("00:23 elapsed")).toBeInTheDocument();
  });

  it("uses provider hints from status messages when available", () => {
    render(
      <IngestionProgress
        elapsedSec={5}
        onRetry={vi.fn()}
        status={makeStatus({
          status: "parsing",
          progress: 31,
          message: "qwen/qwen-max via OpenRouter",
        })}
      />,
    );

    expect(screen.getAllByText("qwen/qwen-max via OpenRouter")).toHaveLength(2);
  });

  it("renders failure state and retries", async () => {
    const onRetry = vi.fn();

    render(
      <IngestionProgress
        elapsedSec={0}
        onRetry={onRetry}
        status={makeStatus({
          status: "failed",
          error: "Token budget exceeded",
          message: "Ingestion failed",
        })}
      />,
    );

    expect(screen.getByText("Ingestion failed")).toBeInTheDocument();
    expect(screen.getByText("Token budget exceeded")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Try Again" }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("renders completed summary and calls browse/ask actions", async () => {
    const onBrowse = vi.fn();
    const onAsk = vi.fn();

    render(
      <IngestionProgress
        elapsedSec={90}
        onAsk={onAsk}
        onBrowse={onBrowse}
        onRetry={vi.fn()}
        status={makeStatus({ status: "completed", progress: 100 })}
        summary={{
          repoId: "repo-1",
          totalFiles: 312,
          functions: 1847,
          totalChunks: 2847,
          graphNodes: 4201,
        }}
      />,
    );

    expect(screen.getByText("Repository indexed successfully")).toBeInTheDocument();
    expect(screen.getByText("1,847")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Browse Repository" }));
    await userEvent.click(screen.getByRole("button", { name: "Start Asking Questions" }));

    expect(onBrowse).toHaveBeenCalledTimes(1);
    expect(onAsk).toHaveBeenCalledTimes(1);
  });
});
