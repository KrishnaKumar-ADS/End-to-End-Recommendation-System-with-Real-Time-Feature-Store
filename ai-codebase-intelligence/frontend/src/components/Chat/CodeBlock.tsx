"use client";

import { useEffect, useRef, useState } from "react";

import { highlightCode } from "@/lib/highlighter";
import { languageDisplayName } from "@/lib/markdown";
import { cn } from "@/lib/utils";

interface CodeBlockProps {
  code: string;
  language: string;
  className?: string;
}

export function CodeBlock({ code, language, className }: CodeBlockProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [copied, setCopied] = useState(false);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    let isCancelled = false;

    void highlightCode(code, language).then((html) => {
      if (isCancelled || !containerRef.current) {
        return;
      }

      containerRef.current.innerHTML = html;
      setIsReady(true);
    });

    return () => {
      isCancelled = true;
    };
  }, [code, language]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Ignore clipboard failures.
    }
  };

  const displayLanguage = languageDisplayName(language);

  return (
    <div
      className={cn(
        "group my-3 overflow-hidden rounded-xl border border-surface-border",
        className,
      )}
    >
      <div className="flex items-center justify-between border-b border-surface-border bg-surface-card px-4 py-2">
        <span className="font-mono text-xs text-surface-muted">{displayLanguage}</span>
        <button
          aria-label={copied ? "Copied!" : "Copy code"}
          className={cn(
            "rounded px-2 py-0.5 font-mono text-xs transition",
            "opacity-0 group-hover:opacity-100",
            copied
              ? "bg-emerald-500/15 text-emerald-300"
              : "text-surface-muted hover:bg-surface-hover hover:text-slate-200",
          )}
          onClick={handleCopy}
          type="button"
        >
          {copied ? "✓ Copied" : "Copy"}
        </button>
      </div>

      {!isReady ? (
        <pre
          aria-label={`${displayLanguage} code`}
          className="overflow-x-auto bg-[#0d1117] p-4 font-mono text-sm text-slate-200"
        >
          <code>{code}</code>
        </pre>
      ) : null}

      <div
        aria-label={`${displayLanguage} highlighted code`}
        className={cn(
          "overflow-x-auto [&_.shiki]:my-0 [&_.shiki]:rounded-none [&_.shiki]:p-4 [&_.shiki]:text-sm",
          !isReady && "hidden",
        )}
        ref={containerRef}
      />
    </div>
  );
}
