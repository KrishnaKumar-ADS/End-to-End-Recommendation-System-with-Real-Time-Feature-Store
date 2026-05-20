"use client";

import Link from "next/link";
import { useState } from "react";

import { CodeBlock } from "@/components/Chat/CodeBlock";
import { cn } from "@/lib/utils";
import type { SourceCitation } from "@/types/api";

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
    json: "json",
    yaml: "yaml",
    yml: "yaml",
    md: "markdown",
    sh: "bash",
    sql: "sql",
  };
  return map[extension] ?? "text";
}

function SourceCitationItem({
  index,
  repoId,
  source,
}: {
  index: number;
  repoId: string;
  source: SourceCitation;
}) {
  const [isExpanded, setExpanded] = useState(false);

  return (
    <div className="overflow-hidden rounded-lg border border-surface-border">
      <div className="flex items-center gap-2 bg-surface-card px-3 py-2">
        <span className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-brand-500/15 text-[10px] font-bold text-brand-200">
          {index + 1}
        </span>

        <Link
          className="flex-1 truncate font-mono text-xs text-brand-300 transition hover:text-brand-200 hover:underline"
          href={`/repos/${repoId}?file=${encodeURIComponent(source.file)}`}
          title={source.file}
        >
          {source.file}
        </Link>

        {source.function ? (
          <span className="shrink-0 truncate font-mono text-xs text-violet-300">{source.function}()</span>
        ) : null}

        <span className="shrink-0 font-mono text-xs text-surface-muted">:{source.lines}</span>

        {source.snippet ? (
          <button
            aria-label={isExpanded ? "Hide code snippet" : "Show code snippet"}
            className="shrink-0 text-xs text-surface-muted transition hover:text-slate-200"
            onClick={() => setExpanded((current) => !current)}
            type="button"
          >
            {isExpanded ? "▲ hide" : "▼ view"}
          </button>
        ) : null}
      </div>

      {isExpanded && source.snippet ? (
        <div className="animate-fade-in border-t border-surface-border">
          <CodeBlock
            className="my-0 rounded-none border-none"
            code={source.snippet}
            language={languageFromPath(source.file)}
          />
        </div>
      ) : null}
    </div>
  );
}

interface SourceCitationPanelProps {
  sources: SourceCitation[];
  repoId: string;
  className?: string;
}

export function SourceCitationPanel({
  sources,
  repoId,
  className,
}: SourceCitationPanelProps) {
  const [isOpen, setOpen] = useState(true);

  if (!sources.length) {
    return null;
  }

  return (
    <div className={cn("space-y-2", className)}>
      <button
        className="flex w-full items-center gap-2 text-left text-xs font-medium uppercase tracking-[0.08em] text-surface-muted transition hover:text-slate-200"
        onClick={() => setOpen((current) => !current)}
        type="button"
      >
        <span className={cn("inline-block transition-transform", isOpen ? "rotate-90" : "rotate-0")}>▶</span>
        Sources
        <span className="ml-auto font-mono normal-case">{sources.length}</span>
      </button>

      {isOpen ? (
        <div className="space-y-2 animate-fade-in">
          {sources.map((source, index) => (
            <SourceCitationItem
              key={`${source.file}-${source.lines}-${index}`}
              index={index}
              repoId={repoId}
              source={source}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}
