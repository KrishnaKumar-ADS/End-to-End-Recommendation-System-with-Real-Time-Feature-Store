import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SearchPanel } from "@/components/search/SearchPanel";

const useSearchMock = vi.fn();

vi.mock("@/hooks/useSearch", () => ({
  useSearch: (...args: unknown[]) => useSearchMock(...args),
}));

describe("SearchPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useSearchMock.mockReturnValue({
      query: "",
      setQuery: vi.fn(),
      results: [],
      isLoading: false,
      error: null,
      timing: null,
    });
  });

  it("calls useSearch with repo id and defaults", () => {
    render(<SearchPanel onClose={vi.fn()} onSelect={vi.fn()} repoId="repo-1" />);

    expect(useSearchMock).toHaveBeenCalledWith("repo-1", "", 8);
  });

  it("renders empty search state", () => {
    render(<SearchPanel onClose={vi.fn()} onSelect={vi.fn()} repoId="repo-1" />);

    expect(screen.getByText("Type to search functions, classes, and patterns")).toBeInTheDocument();
  });

  it("updates query from input", async () => {
    const user = userEvent.setup();
    const setQuery = vi.fn();
    useSearchMock.mockReturnValue({
      query: "",
      setQuery,
      results: [],
      isLoading: false,
      error: null,
      timing: null,
    });

    render(<SearchPanel onClose={vi.fn()} onSelect={vi.fn()} repoId="repo-1" />);

    await user.type(screen.getByLabelText("Semantic code search"), "auth service");

    expect(setQuery).toHaveBeenCalled();
  });

  it("closes when pressing Escape", () => {
    const onClose = vi.fn();
    render(<SearchPanel onClose={onClose} onSelect={vi.fn()} repoId="repo-1" />);

    fireEvent.keyDown(window, { key: "Escape" });

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("closes when overlay is clicked", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();

    render(<SearchPanel onClose={onClose} onSelect={vi.fn()} repoId="repo-1" />);

    const overlay = document.querySelector("div[aria-hidden='true']");
    expect(overlay).toBeTruthy();

    if (overlay) {
      await user.click(overlay);
    }

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("selects a result and transforms it into a question", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    const onClose = vi.fn();

    useSearchMock.mockReturnValue({
      query: "auth",
      setQuery: vi.fn(),
      results: [
        {
          id: "chunk-1",
          name: "login",
          file_path: "app/auth.py",
          chunk_type: "function",
          start_line: 10,
          end_line: 22,
          content: "",
          docstring: null,
          language: "python",
          parent_name: null,
          vector_score: 0.9,
          bm25_score: 0.7,
          hybrid_score: 0.85,
          rerank_score: null,
          vector_rank: 1,
          bm25_rank: 1,
          hybrid_rank: 1,
          final_rank: 1,
        },
      ],
      isLoading: false,
      error: null,
      timing: {
        embed_ms: 1,
        expand_ms: 1,
        vector_ms: 2,
        bm25_ms: 1,
        fusion_ms: 1,
        rerank_ms: 0,
        total_ms: 5,
      },
    });

    render(<SearchPanel onClose={onClose} onSelect={onSelect} repoId="repo-1" />);

    await user.click(screen.getByRole("button", { name: /Explain this symbol/i }));

    expect(onSelect).toHaveBeenCalledWith("Explain login in app/auth.py");
    expect(onClose).toHaveBeenCalled();
  });
});
