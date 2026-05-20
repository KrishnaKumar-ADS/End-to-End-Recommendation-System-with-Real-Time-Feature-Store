import { describe, expect, it } from "vitest";

import {
  buildFileTree,
  cn,
  extractFileSymbols,
  formatBytes,
  formatDuration,
  getFileLanguageLabel,
  getProviderTone,
  getRepoStats,
  getStatusLabel,
  getStatusTone,
  isGithubUrl,
} from "@/lib/utils";
import type { GraphResponse, RepositoryDetail } from "@/types/api";

describe("utils", () => {
  it("cn merges duplicate Tailwind classes", () => {
    expect(cn("px-2", "px-4")).toContain("px-4");
  });

  it("formatDuration formats minutes and seconds", () => {
    expect(formatDuration(65)).toBe("01:05");
  });

  it("formatDuration formats hours when needed", () => {
    expect(formatDuration(3661)).toBe("1:01:01");
  });

  it("formatBytes returns a readable string", () => {
    expect(formatBytes(2048)).toBe("2 KB");
  });

  it("recognizes valid GitHub repository URLs", () => {
    expect(isGithubUrl("https://github.com/openai/openai-python")).toBe(true);
  });

  it("rejects invalid GitHub URLs", () => {
    expect(isGithubUrl("https://example.com/not-github")).toBe(false);
  });

  it("maps completed status to Ready", () => {
    expect(getStatusLabel("completed")).toBe("Ready");
  });

  it("maps failed status to danger tone", () => {
    expect(getStatusTone("failed")).toBe("danger");
  });

  it("maps qwen models to qwen badge tone", () => {
    expect(getProviderTone("openrouter", "qwen/qwen-max")).toBe("qwen");
  });

  it("builds a nested file tree with directories first", () => {
    const tree = buildFileTree([
      {
        id: "1",
        file_path: "src/app/page.tsx",
        language: "typescript",
        size_bytes: 1,
        line_count: 1,
        chunk_count: 1,
      },
      {
        id: "2",
        file_path: "src/lib/utils.ts",
        language: "typescript",
        size_bytes: 1,
        line_count: 1,
        chunk_count: 1,
      },
      {
        id: "3",
        file_path: "README.md",
        language: "markdown",
        size_bytes: 1,
        line_count: 1,
        chunk_count: 1,
      },
    ]);

    expect(tree).toHaveLength(2);
    expect(tree[0].name).toBe("src");
    const srcDir = tree.find((node) => node.name === "src");
    expect(srcDir?.children).toHaveLength(2);
  });

  it("extracts functions and classes for a selected file from graph nodes", () => {
    const graph: GraphResponse = {
      repo_id: "r1",
      node_count: 2,
      edge_count: 0,
      edges: [],
      nodes: [
        { id: "n1", _type: "Function", display_name: "handle_login", file_path: "app/auth.py", start_line: 12 },
        { id: "n2", _type: "Class", display_name: "AuthService", file_path: "app/auth.py", start_line: 2 },
      ],
    };

    const result = extractFileSymbols(graph, "app/auth.py");
    expect(result.functions).toHaveLength(1);
    expect(result.classes).toHaveLength(1);
  });

  it("computes repository stats from repo detail and graph", () => {
    const repo: RepositoryDetail = {
      id: "r1",
      github_url: "",
      name: "repo",
      branch: "main",
      status: "completed",
      task_id: null,
      error_message: null,
      total_files: 8,
      processed_files: 8,
      total_chunks: 44,
      created_at: "",
      updated_at: "",
      files: [],
    };

    const graph: GraphResponse = {
      repo_id: "r1",
      node_count: 5,
      edge_count: 0,
      edges: [],
      nodes: [
        { id: "1", _type: "Function" },
        { id: "2", _type: "Function" },
        { id: "3", _type: "Class" },
        { id: "4", _type: "File" },
        { id: "5", _type: "File" },
      ],
    };

    expect(getRepoStats(repo, graph)).toEqual({
      files: 8,
      functions: 2,
      classes: 1,
      embeddings: 44,
      graphNodes: 5,
    });
  });

  it("formats file language labels", () => {
    expect(getFileLanguageLabel("python")).toBe("Python");
  });
});
