"use client";

import { useState } from "react";

import { cn, formatPercent } from "@/lib/utils";
import type { QualityScore } from "@/types/api";

function scoreFillColor(score: number): string {
  if (score >= 0.8) {
    return "bg-emerald-500";
  }
  if (score >= 0.6) {
    return "bg-amber-500";
  }
  return "bg-red-500";
}

function scoreTextColor(score: number): string {
  if (score >= 0.8) {
    return "text-emerald-300";
  }
  if (score >= 0.6) {
    return "text-amber-300";
  }
  return "text-red-300";
}

function scoreStatus(score: number): string {
  if (score >= 0.8) {
    return "OK";
  }
  if (score >= 0.6) {
    return "WARN";
  }
  return "LOW";
}

function ScoreRow({ label, score }: { label: string; score: number }) {
  const percent = Math.round(score * 100);

  return (
    <div className="flex items-center gap-3">
      <span className="w-28 shrink-0 text-xs text-surface-muted">{label}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-border">
        <div
          aria-label={`${label}: ${percent}%`}
          aria-valuemax={100}
          aria-valuemin={0}
          aria-valuenow={percent}
          className={cn("h-full rounded-full transition-[width] duration-700", scoreFillColor(score))}
          role="progressbar"
          style={{ width: `${percent}%` }}
        />
      </div>
      <span className={cn("w-12 text-right font-mono text-xs tabular-nums", scoreTextColor(score))}>
        {formatPercent(score, 0)}
      </span>
      <span className="w-8 text-center font-mono text-[10px] text-surface-muted">{scoreStatus(score)}</span>
    </div>
  );
}

interface QualityScorePanelProps {
  qualityScore: QualityScore;
  className?: string;
}

export function QualityScorePanel({ qualityScore, className }: QualityScorePanelProps) {
  const [isCritiqueOpen, setCritiqueOpen] = useState(false);

  if (qualityScore.skipped) {
    return (
      <div className={cn("text-xs italic text-surface-muted", className)}>
        Quality scoring skipped{qualityScore.skip_reason ? ` - ${qualityScore.skip_reason}` : ""}.
      </div>
    );
  }

  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-[0.08em] text-surface-muted">Answer quality</span>
        <span className="font-mono text-xs text-slate-300">Overall: {formatPercent(qualityScore.overall, 0)}</span>
      </div>

      <div className="space-y-1.5">
        <ScoreRow label="Faithfulness" score={qualityScore.faithfulness} />
        <ScoreRow label="Relevance" score={qualityScore.relevance} />
        <ScoreRow label="Completeness" score={qualityScore.completeness} />
      </div>

      {qualityScore.critique ? (
        <div>
          <button
            className="flex items-center gap-1 text-xs text-surface-muted transition hover:text-slate-200"
            onClick={() => setCritiqueOpen((current) => !current)}
            type="button"
          >
            <span className={cn("inline-block transition-transform", isCritiqueOpen ? "rotate-90" : "rotate-0")}>▶</span>
            {isCritiqueOpen ? "Hide critique" : "See critique"}
          </button>

          {isCritiqueOpen ? (
            <blockquote className="mt-2 animate-fade-in border-l-2 border-surface-border pl-3 text-xs italic leading-relaxed text-slate-300">
              &quot;{qualityScore.critique}&quot;
            </blockquote>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
