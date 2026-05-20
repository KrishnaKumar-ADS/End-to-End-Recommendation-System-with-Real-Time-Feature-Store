import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

import type {
  FileTreeNode,
  GraphNode,
  GraphResponse,
  IngestionStatus,
  RepositoryDetail,
  RepositoryFile,
} from "@/types/api";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

export function formatPercent(value: number, decimals = 2): string {
  const safe = Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
  return `${(safe * 100).toFixed(decimals)}%`;
}

export function formatBytes(bytes: number): string {
  if (bytes <= 0) {
    return "0 B";
  }

  const units = ["B", "KB", "MB", "GB"];
  const unitIndex = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const amount = bytes / 1024 ** unitIndex;
  const formatted = Number.isInteger(amount)
    ? amount.toString()
    : amount.toFixed(amount >= 10 || unitIndex === 0 ? 0 : 1);
  return `${formatted} ${units[unitIndex]}`;
}

export function formatDuration(totalSeconds: number): string {
  const safe = Math.max(0, Math.floor(totalSeconds));
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const seconds = safe % 60;

  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }

  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

export function formatRelativeDate(value: string | number | Date): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Unknown";
  }

  const diffMs = Date.now() - date.getTime();
  const diffMinutes = Math.round(diffMs / 60000);

  if (Math.abs(diffMinutes) < 1) {
    return "Just now";
  }
  if (Math.abs(diffMinutes) < 60) {
    return `${Math.abs(diffMinutes)}m ${diffMinutes >= 0 ? "ago" : "from now"}`;
  }

  const diffHours = Math.round(diffMinutes / 60);
  if (Math.abs(diffHours) < 24) {
    return `${Math.abs(diffHours)}h ${diffHours >= 0 ? "ago" : "from now"}`;
  }

  const diffDays = Math.round(diffHours / 24);
  if (Math.abs(diffDays) < 7) {
    return `${Math.abs(diffDays)}d ${diffDays >= 0 ? "ago" : "from now"}`;
  }

  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function repoNameFromUrl(url: string): string {
  try {
    const parsed = new URL(url);
    const parts = parsed.pathname.split("/").filter(Boolean);
    if (parts.length >= 2) {
      return `${parts[0]}/${parts[1]}`;
    }
  } catch {
    // Ignore parsing errors and fall back to the original string.
  }
  return url;
}

export function isGithubUrl(value: string): boolean {
  return /^https:\/\/github\.com\/[^/\s]+\/[^/\s]+\/?$/.test(value.trim());
}

export function getStatusLabel(status: IngestionStatus | string): string {
  const normalized = status.toLowerCase();
  switch (normalized) {
    case "queued":
      return "Queued";
    case "cloning":
      return "Cloning";
    case "scanning":
      return "Scanning";
    case "parsing":
      return "Parsing";
    case "embedding":
      return "Embedding";
    case "completed":
      return "Ready";
    case "failed":
      return "Failed";
    default:
      return normalized.charAt(0).toUpperCase() + normalized.slice(1);
  }
}

export function getStatusTone(status: IngestionStatus | string): "neutral" | "warning" | "info" | "success" | "danger" {
  const normalized = status.toLowerCase();
  if (normalized === "completed") {
    return "success";
  }
  if (normalized === "failed") {
    return "danger";
  }
  if (normalized === "parsing") {
    return "info";
  }
  if (["cloning", "scanning", "embedding"].includes(normalized)) {
    return "warning";
  }
  return "neutral";
}

export function getProviderTone(provider?: string, model?: string): "neutral" | "qwen" | "gemini" | "deepseek" {
  const fingerprint = `${provider ?? ""} ${model ?? ""}`.toLowerCase();
  if (fingerprint.includes("qwen") || fingerprint.includes("openrouter")) {
    return "qwen";
  }
  if (fingerprint.includes("gemini")) {
    return "gemini";
  }
  if (fingerprint.includes("deepseek")) {
    return "deepseek";
  }
  return "neutral";
}

export function getProviderLabel(provider?: string, model?: string): string {
  if (model && provider) {
    return `${model} via ${provider}`;
  }
  if (model) {
    return model;
  }
  if (provider) {
    return provider;
  }
  return "Pipeline";
}

function sortTree(nodes: FileTreeNode[]): FileTreeNode[] {
  return [...nodes]
    .sort((a, b) => {
      if (a.type !== b.type) {
        return a.type === "dir" ? -1 : 1;
      }
      return a.name.localeCompare(b.name);
    })
    .map((node) =>
      node.type === "dir" && node.children
        ? { ...node, children: sortTree(node.children) }
        : node,
    );
}

export function buildFileTree(files: RepositoryFile[]): FileTreeNode[] {
  const root: FileTreeNode[] = [];

  for (const file of files) {
    const parts = file.file_path.split("/").filter(Boolean);
    let cursor = root;
    let currentPath = "";

    parts.forEach((part, index) => {
      currentPath = currentPath ? `${currentPath}/${part}` : part;
      const isFile = index === parts.length - 1;
      let existing = cursor.find((node) => node.name === part && node.path === currentPath);

      if (!existing) {
        existing = {
          id: isFile ? file.id : `dir:${currentPath}`,
          name: part,
          path: currentPath,
          type: isFile ? "file" : "dir",
          children: isFile ? undefined : [],
          file: isFile ? file : undefined,
        };
        cursor.push(existing);
      }

      if (!isFile) {
        if (!existing.children) {
          existing.children = [];
        }
        cursor = existing.children;
      }
    });
  }

  return sortTree(root);
}

export function getNodeFilePath(node: GraphNode): string {
  const raw = node.file_path ?? node.file ?? node.path;
  return typeof raw === "string" ? raw : "";
}

export function extractFileSymbols(graph: GraphResponse | null | undefined, filePath: string) {
  const nodes = graph?.nodes ?? [];
  const symbols = nodes
    .filter((node) => getNodeFilePath(node) === filePath)
    .map((node) => ({
      id: node.id,
      type: String(node._type ?? node.label ?? "Unknown"),
      name: String(node.display_name ?? node.name ?? node.id),
      startLine: typeof node.start_line === "number" ? node.start_line : 0,
    }))
    .sort((a, b) => a.startLine - b.startLine);

  return {
    functions: symbols.filter((symbol) => symbol.type.toLowerCase() === "function"),
    classes: symbols.filter((symbol) => symbol.type.toLowerCase() === "class"),
  };
}

export function getRepoStats(repo: RepositoryDetail, graph?: GraphResponse | null) {
  const nodes = graph?.nodes ?? [];
  const functionCount = nodes.filter(
    (node) => String(node._type ?? node.label ?? "").toLowerCase() === "function",
  ).length;
  const classCount = nodes.filter(
    (node) => String(node._type ?? node.label ?? "").toLowerCase() === "class",
  ).length;

  return {
    files: repo.total_files,
    functions: functionCount,
    classes: classCount,
    embeddings: repo.total_chunks,
    graphNodes: graph?.node_count ?? 0,
  };
}

export function getFileLanguageLabel(language: string): string {
  if (!language) {
    return "Unknown";
  }
  return language.charAt(0).toUpperCase() + language.slice(1);
}
