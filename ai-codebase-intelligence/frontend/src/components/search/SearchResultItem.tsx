"use client";

import { CodeBlock } from "@/components/Chat/CodeBlock";
import { cn } from "@/lib/utils";
import type { SearchResult } from "@/types/api";

interface SearchResultItemProps {
  result: SearchResult;
  onSelect: (question: string) => void;
}

function languageFromPath(path: string): string {
  const extension = path.split(".").pop()?.toLowerCase() ?? "text";
  const map: Record<string, string> = {
    py: "python",
    ts: "typescript",
    js: "javascript",
    tsx: "tsx",
    jsx: "jsx",
    go: "go",
    rs: "rust",
  };

  return map[extension] ?? "text";
}

export function SearchResultItem({ result, onSelect }: SearchResultItemProps) {
  const score = Number.isFinite(result.hybrid_score)
    ? result.hybrid_score
    : Number.isFinite(result.vector_score)
      ? result.vector_score
      : 0;
  const scorePercent = Math.max(0, Math.min(100, Math.round(score * 100)));

  const question = result.name
    ? `Explain ${result.name} in ${result.file_path}`
    : `Explain the code in ${result.file_path} around lines ${result.start_line}-${result.end_line}`;

  return (
    <div
      className={cn(
        "overflow-hidden rounded-xl border border-surface-border bg-surface-card",
        "transition hover:border-brand-500/30",
      )}
    >
      <div className="flex items-center gap-2 px-3 py-2">
        <div className="min-w-0 flex-1">
          <p className="truncate font-mono text-xs text-brand-300" title={result.file_path}>
            {result.file_path}
          </p>
          {result.name ? (
            <p className="truncate font-mono text-xs text-violet-300">{result.name}</p>
          ) : null}
          <p className="font-mono text-[10px] text-surface-muted">
            lines {result.start_line}-{result.end_line}
          </p>
        </div>

        <div className="flex shrink-0 flex-col items-end gap-1">
          <span className="font-mono text-[10px] text-surface-muted">{scorePercent}%</span>
          <div className="h-1 w-12 overflow-hidden rounded-full bg-surface-border">
            <div className="h-full rounded-full bg-brand-500" style={{ width: `${scorePercent}%` }} />
          </div>
        </div>
      </div>

      {result.content ? (
        <CodeBlock
          className="my-0 rounded-none border-x-0 border-b-0 border-t"
          code={result.content}
          language={languageFromPath(result.file_path)}
        />
      ) : null}

      <div className="border-t border-surface-border bg-surface px-3 py-2">
        <button
          className="w-full text-left font-mono text-xs text-surface-muted transition hover:text-brand-200"
          onClick={() => onSelect(question)}
          type="button"
        >
          → Explain this {result.name ? "symbol" : "code"}
        </button>
      </div>
    </div>
  );
}
