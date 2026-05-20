"use client";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { ProviderBadge, StatusBadge } from "@/components/ui/Badge";
import { formatDuration, formatNumber } from "@/lib/utils";
import type { StatusResponse } from "@/types/api";

const STEPS = ["queued", "cloning", "scanning", "parsing", "embedding", "completed"];
const DEFAULT_PROVIDER_INFO = {
  provider: "OpenRouter",
  model: "qwen/qwen-2.5-coder-32b-instruct",
};

function getMatchValue(match: RegExpMatchArray | null, index: number): string | null {
  const value = match?.[index]?.trim();
  return value ? value : null;
}

function parseProviderHint(message: string | null | undefined) {
  const normalized = (message ?? "").trim();
  if (!normalized) {
    return null;
  }

  const viaMatch = normalized.match(/([\w./:-]+)\s+via\s+([\w .:-]+)/i);
  const model = getMatchValue(viaMatch, 1);
  const provider = getMatchValue(viaMatch, 2);
  if (model && provider) {
    return { model, provider };
  }

  const qwenMatch = normalized.match(/(qwen\/[\w.-]+)/i);
  if (qwenMatch) {
    return {
      model: getMatchValue(qwenMatch, 1) ?? DEFAULT_PROVIDER_INFO.model,
      provider: normalized.toLowerCase().includes("openrouter")
        ? "OpenRouter"
        : DEFAULT_PROVIDER_INFO.provider,
    };
  }

  const deepseekMatch = normalized.match(/(deepseek[\w./-]*)/i);
  if (deepseekMatch) {
    return {
      model: getMatchValue(deepseekMatch, 1) ?? "deepseek-coder-v2",
      provider: "DeepSeek",
    };
  }

  const geminiEmbeddingMatch = normalized.match(/(text-embedding-004)/i);
  if (geminiEmbeddingMatch) {
    return {
      model: getMatchValue(geminiEmbeddingMatch, 1) ?? "text-embedding-004",
      provider: "Gemini",
    };
  }

  const geminiMatch = normalized.match(/(gemini[\w.-]*)/i);
  if (geminiMatch) {
    return {
      model: getMatchValue(geminiMatch, 1) ?? "gemini-2.0-flash",
      provider: "Gemini",
    };
  }

  if (normalized.toLowerCase().includes("openrouter")) {
    return DEFAULT_PROVIDER_INFO;
  }

  return null;
}

function getProviderForStatus(status: string, message: string | null | undefined) {
  const parsedHint = parseProviderHint(message);
  if (parsedHint) {
    return parsedHint;
  }

  return DEFAULT_PROVIDER_INFO;
}

export function IngestionProgress({
  status,
  elapsedSec,
  summary,
  onRetry,
  onBrowse,
  onAsk,
}: {
  status: StatusResponse | null;
  elapsedSec: number;
  summary?: {
    repoId: string;
    totalFiles: number;
    functions: number;
    totalChunks: number;
    graphNodes: number;
  } | null;
  onRetry: () => void;
  onBrowse?: () => void;
  onAsk?: () => void;
}) {
  if (!status) {
    return null;
  }

  const stageIndex = Math.max(STEPS.indexOf(status.status), 0);
  const providerInfo = getProviderForStatus(status.status, status.message);

  if (status.status === "failed") {
    return (
      <Card className="space-y-5" padding="lg">
        <div className="space-y-2">
          <p className="text-xs uppercase tracking-[0.14em] text-red-200">Step 4</p>
          <h3 className="text-2xl font-semibold text-white">Ingestion failed</h3>
          <p className="text-sm text-red-200">
            {status.error || status.message || "The backend reported an unknown error."}
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <Button onClick={onRetry}>Try Again</Button>
        </div>
      </Card>
    );
  }

  if (status.status === "completed") {
    return (
      <Card className="space-y-5" padding="lg">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="space-y-2">
            <p className="text-xs uppercase tracking-[0.14em] text-emerald-200">Step 3</p>
            <h3 className="text-2xl font-semibold text-white">Repository indexed successfully</h3>
            <p className="text-sm text-surface-muted">
              The repository is ready for browsing, search, and question answering.
            </p>
          </div>
          <StatusBadge status={status.status} />
        </div>

        <div className="grid gap-3 md:grid-cols-4">
          <Card padding="sm">
            <p className="text-xs uppercase tracking-[0.12em] text-surface-muted">Files</p>
            <p className="mt-2 text-2xl font-semibold text-white">
              {formatNumber(summary?.totalFiles ?? status.total_files)}
            </p>
          </Card>
          <Card padding="sm">
            <p className="text-xs uppercase tracking-[0.12em] text-surface-muted">Functions</p>
            <p className="mt-2 text-2xl font-semibold text-white">
              {formatNumber(summary?.functions ?? 0)}
            </p>
          </Card>
          <Card padding="sm">
            <p className="text-xs uppercase tracking-[0.12em] text-surface-muted">Chunks</p>
            <p className="mt-2 text-2xl font-semibold text-white">
              {formatNumber(summary?.totalChunks ?? status.total_chunks)}
            </p>
          </Card>
          <Card padding="sm">
            <p className="text-xs uppercase tracking-[0.12em] text-surface-muted">Graph nodes</p>
            <p className="mt-2 text-2xl font-semibold text-white">
              {formatNumber(summary?.graphNodes ?? 0)}
            </p>
          </Card>
        </div>

        <div className="flex flex-wrap gap-3">
          {onBrowse ? (
            <Button variant="secondary" onClick={onBrowse}>
              Browse Repository
            </Button>
          ) : null}
          {onAsk ? <Button onClick={onAsk}>Start Asking Questions</Button> : null}
        </div>
      </Card>
    );
  }

  return (
    <Card className="space-y-5" padding="lg">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-2">
          <p className="text-xs uppercase tracking-[0.14em] text-brand-200">Step 2</p>
          <h3 className="text-2xl font-semibold text-white">Ingestion in progress</h3>
          <p className="text-sm text-surface-muted">
            {status.message || "Waiting for the next backend update..."}
          </p>
        </div>
        <StatusBadge status={status.status} />
      </div>

      <ProgressBar
        indeterminate={status.progress === 0 && status.status !== "queued"}
        label={`${status.progress}% complete`}
        value={status.progress}
      />

      <div className="flex flex-wrap items-center gap-3 text-sm text-slate-200">
        <span className="text-xs uppercase tracking-[0.08em] text-surface-muted">Provider:</span>
        <ProviderBadge model={providerInfo.model} provider={providerInfo.provider} />
        <span className="rounded-full border border-surface-border px-3 py-1 text-xs uppercase tracking-[0.08em] text-surface-muted">
          {formatDuration(elapsedSec)} elapsed
        </span>
        <span className="rounded-full border border-surface-border px-3 py-1 text-xs uppercase tracking-[0.08em] text-surface-muted">
          {formatNumber(status.processed_files)} / {formatNumber(status.total_files)} files
        </span>
      </div>

      <div className="grid gap-2 md:grid-cols-3 xl:grid-cols-6">
        {STEPS.map((step, index) => {
          const complete = index < stageIndex;
          const active = step === status.status;
          return (
            <div
              key={step}
              className={`rounded-2xl border px-3 py-3 text-sm ${
                active
                  ? "border-brand-500/40 bg-brand-500/10 text-brand-100"
                  : complete
                    ? "border-emerald-500/30 bg-emerald-500/8 text-emerald-100"
                    : "border-surface-border bg-white/4 text-surface-muted"
              }`}
            >
              <p className="text-[11px] uppercase tracking-[0.12em]">{index + 1}</p>
              <p className="mt-2 font-medium capitalize">{step}</p>
            </div>
          );
        })}
      </div>
    </Card>
  );
}
