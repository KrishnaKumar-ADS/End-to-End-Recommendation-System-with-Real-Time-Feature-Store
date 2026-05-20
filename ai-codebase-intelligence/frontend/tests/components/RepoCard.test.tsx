import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

import { RepoCard } from "@/components/repos/RepoCard";
import type { RepositorySummary } from "@/types/api";

vi.mock("next/link", () => ({
  default: ({ href, children, ...props }: { href: string; children: React.ReactNode }) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
}));

function makeRepo(overrides: Partial<RepositorySummary> = {}): RepositorySummary {
  return {
    id: "repo-1",
    github_url: "https://github.com/tiangolo/fastapi",
    name: "fastapi",
    branch: "main",
    status: "completed",
    task_id: null,
    error_message: null,
    total_files: 312,
    processed_files: 312,
    total_chunks: 2847,
    created_at: "2026-04-01T00:00:00.000Z",
    updated_at: "2026-04-01T00:00:00.000Z",
    ...overrides,
  };
}

describe("RepoCard", () => {
  it("renders repository metadata and status", () => {
    render(<RepoCard repo={makeRepo()} />);

    expect(screen.getByText("fastapi")).toBeInTheDocument();
    expect(screen.getByText("https://github.com/tiangolo/fastapi")).toBeInTheDocument();
    expect(screen.getByText("Ready")).toBeInTheDocument();
    expect(screen.getByText("Branch: main")).toBeInTheDocument();
  });

  it("links to browse and ask routes", () => {
    render(<RepoCard repo={makeRepo({ id: "abc-123" })} />);

    expect(screen.getByRole("link", { name: "Browse" })).toHaveAttribute(
      "href",
      "/repos/abc-123",
    );
    expect(screen.getByRole("link", { name: "Ask" })).toHaveAttribute(
      "href",
      "/repos/abc-123/chat",
    );
  });

  it("shows formatted counters", () => {
    render(<RepoCard repo={makeRepo({ total_files: 1200, total_chunks: 45000, processed_files: 999 })} />);

    expect(screen.getByText("1,200")).toBeInTheDocument();
    expect(screen.getByText("45,000")).toBeInTheDocument();
    expect(screen.getByText("999")).toBeInTheDocument();
  });
});
