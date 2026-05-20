"use client";

import { cn } from "@/lib/utils";

const SUGGESTED_QUESTIONS = [
  "How does the dependency injection system work?",
  "Explain the routing mechanism in main.py",
  "What happens when a request validation fails?",
  "Are there any security vulnerabilities?",
  "How is the database connection managed?",
  "Trace the execution path for the most common use case.",
];

interface WelcomeScreenProps {
  repoName: string;
  onQuestion: (question: string) => void;
}

export function WelcomeScreen({ repoName, onQuestion }: WelcomeScreenProps) {
  const suggestions = SUGGESTED_QUESTIONS.slice(0, 4);

  return (
    <div className="flex flex-1 flex-col items-center justify-center space-y-8 px-6 py-12 text-center animate-fade-in">
      <div className="space-y-3">
        <div
          aria-hidden="true"
          className={cn(
            "inline-flex h-14 w-14 items-center justify-center rounded-2xl",
            "border border-violet-500/30 bg-violet-600/20 text-2xl",
          )}
        >
          💬
        </div>
        <h2 className="text-xl font-semibold text-slate-100">Ask anything about</h2>
        <p className="text-lg font-mono text-brand-300">{repoName}</p>
        <p className="mx-auto max-w-sm text-sm leading-relaxed text-surface-muted">
          Powered by Qwen via OpenRouter, Gemini embeddings, and a Neo4j call graph.
          Answers are grounded in indexed code with source citations.
        </p>
      </div>

      <div className="w-full max-w-lg space-y-2">
        <p className="text-xs font-medium uppercase tracking-[0.1em] text-surface-muted">Try asking</p>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {suggestions.map((suggestion) => (
            <button
              className={cn(
                "rounded-xl border border-surface-border bg-surface-card px-4 py-3",
                "text-left text-sm text-slate-300 transition",
                "hover:border-brand-500/40 hover:bg-surface-hover hover:text-slate-100",
              )}
              key={suggestion}
              onClick={() => onQuestion(suggestion)}
              type="button"
            >
              <span aria-hidden="true" className="mr-2 text-brand-300">
                →
              </span>
              {suggestion}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
