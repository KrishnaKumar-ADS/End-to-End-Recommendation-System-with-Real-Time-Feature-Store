import Link from "next/link";

import { Badge, StatusBadge } from "@/components/ui/Badge";
import { Card } from "@/components/ui/Card";
import { cn, formatNumber, formatRelativeDate } from "@/lib/utils";
import type { RepositorySummary } from "@/types/api";

const actionClasses =
  "inline-flex h-11 items-center justify-center rounded-xl px-4 text-sm font-medium transition";

export function RepoCard({ repo }: { repo: RepositorySummary }) {
  return (
    <Card className="flex h-full flex-col justify-between gap-6" padding="lg">
      <div className="space-y-4">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0 space-y-1">
            <h3 className="truncate text-xl font-semibold text-white" title={repo.name}>{repo.name}</h3>
            <p className="truncate text-sm text-surface-muted" title={repo.github_url}>{repo.github_url}</p>
          </div>
          <StatusBadge status={repo.status} />
        </div>

        <div className="flex flex-wrap gap-2">
          <Badge>Branch: {repo.branch}</Badge>
          <Badge>Updated {formatRelativeDate(repo.updated_at)}</Badge>
        </div>

        <div className="grid grid-cols-3 gap-3">
          <div className="rounded-2xl border border-surface-border bg-white/4 p-3">
            <p className="text-xs uppercase tracking-[0.12em] text-surface-muted">Files</p>
            <p className="mt-2 text-lg font-semibold text-white">{formatNumber(repo.total_files)}</p>
          </div>
          <div className="rounded-2xl border border-surface-border bg-white/4 p-3">
            <p className="text-xs uppercase tracking-[0.12em] text-surface-muted">Chunks</p>
            <p className="mt-2 text-lg font-semibold text-white">{formatNumber(repo.total_chunks)}</p>
          </div>
          <div className="rounded-2xl border border-surface-border bg-white/4 p-3">
            <p className="text-xs uppercase tracking-[0.12em] text-surface-muted">Indexed</p>
            <p className="mt-2 text-lg font-semibold text-white">{formatNumber(repo.processed_files)}</p>
          </div>
        </div>
      </div>

      <div className="flex flex-wrap gap-3">
        <Link
          className={cn(actionClasses, "border border-surface-border bg-surface-card text-slate-100 hover:border-brand-500 hover:bg-surface-hover")}
          href={`/repos/${repo.id}`}
        >
          Browse
        </Link>
        <Link
          className={cn(actionClasses, "bg-brand-600 text-white hover:bg-brand-500")}
          href={`/repos/${repo.id}/chat`}
        >
          Ask
        </Link>
      </div>
    </Card>
  );
}
