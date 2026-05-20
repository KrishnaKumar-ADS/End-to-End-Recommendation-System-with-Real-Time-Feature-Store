"use client";

import { useEffect, useRef } from "react";

import { SearchResultItem } from "@/components/search/SearchResultItem";
import { Button } from "@/components/ui/Button";
import { Spinner } from "@/components/ui/Spinner";
import { useSearch } from "@/hooks/useSearch";
import { cn } from "@/lib/utils";

interface SearchPanelProps {
  repoId: string;
  onSelect: (question: string) => void;
  onClose: () => void;
}

export function SearchPanel({ repoId, onSelect, onClose }: SearchPanelProps) {
  const { query, setQuery, results, isLoading, error, timing } = useSearch(repoId, "", 8);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const handleSelect = (question: string) => {
    onSelect(question);
    onClose();
  };

  return (
    <>
      <div aria-hidden="true" className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm" onClick={onClose} />

      <div
        aria-label="Code search panel"
        aria-modal="true"
        className={cn(
          "fixed inset-0 z-50 flex h-full w-full flex-col",
          "bg-surface shadow-2xl animate-slide-up",
          "md:inset-y-0 md:right-0 md:left-auto md:max-w-md md:border-l md:border-surface-border",
        )}
        role="dialog"
      >
        <div className="flex shrink-0 items-center gap-3 border-b border-surface-border px-4 py-3 md:py-3">
          <span aria-hidden="true" className="text-surface-muted">
            🔍
          </span>
          <h2 className="text-sm font-medium text-slate-200">Semantic Search</h2>
          <Button
            aria-label="Close search panel"
            className="ml-auto"
            onClick={onClose}
            size="sm"
            variant="ghost"
          >
            ✕
          </Button>
        </div>

        <div className="shrink-0 border-b border-surface-border px-4 py-3">
          <div className="relative">
            <input
              aria-label="Semantic code search"
              className={cn(
                "w-full rounded-lg border border-surface-border bg-surface-input px-3 py-2 pr-8",
                "text-sm text-slate-200 placeholder:text-surface-muted",
                "transition focus:border-brand-500/50 focus:outline-none",
              )}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search functions, classes, patterns..."
              ref={inputRef}
              type="text"
              value={query}
            />
            {isLoading ? (
              <div className="absolute right-2 top-1/2 -translate-y-1/2">
                <Spinner size="sm" />
              </div>
            ) : null}
          </div>

          {timing && !isLoading ? (
            <p className="mt-1 font-mono text-[10px] text-surface-muted">
              {results.length} results · {timing.total_ms}ms (embed: {timing.embed_ms}ms · vector: {timing.vector_ms}ms)
            </p>
          ) : null}
        </div>

        <div className="flex-1 space-y-3 overflow-y-auto px-4 py-3">
          {error ? <p className="py-4 text-center text-sm text-red-300">{error}</p> : null}

          {!query && !isLoading ? (
            <div className="space-y-2 py-12 text-center">
              <p aria-hidden="true" className="text-4xl">
                🔍
              </p>
              <p className="text-sm text-surface-muted">Type to search functions, classes, and patterns</p>
              <p className="text-xs text-surface-muted">Powered by Gemini embeddings + Qdrant search</p>
            </div>
          ) : null}

          {query && !isLoading && !results.length && !error ? (
            <div className="space-y-2 py-12 text-center">
              <p className="text-sm text-surface-muted">No results for &quot;{query}&quot;</p>
              <p className="text-xs text-surface-muted">Try broader terms or synonyms.</p>
            </div>
          ) : null}

          {results.map((result) => (
            <SearchResultItem key={`${result.id}-${result.file_path}-${result.start_line}`} onSelect={handleSelect} result={result} />
          ))}
        </div>
      </div>
    </>
  );
}
