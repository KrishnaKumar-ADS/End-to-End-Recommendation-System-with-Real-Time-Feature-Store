"use client";

import Link from "next/link";

import { RepoCard } from "@/components/repos/RepoCard";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { useRepos } from "@/hooks/useRepos";
import { formatNumber } from "@/lib/utils";

export default function HomePage() {
  const { repos, isLoading } = useRepos(6, 0);

  return (
    <div className="space-y-8">
      <Card className="overflow-hidden bg-[radial-gradient(circle_at_top_right,rgba(59,150,245,0.18),transparent_35%),linear-gradient(180deg,rgba(255,255,255,0.04),rgba(255,255,255,0.02))]" padding="lg">
        <div className="grid gap-8 lg:grid-cols-[1.4fr,0.8fr] lg:items-end">
          <div className="space-y-5">
            <p className="text-xs uppercase tracking-[0.18em] text-brand-200">Repository intelligence</p>
            <h1 className="max-w-3xl text-4xl font-semibold tracking-tight text-white sm:text-5xl">
              Ingest repositories, browse indexed code, and prepare the platform for conversational Q&A.
            </h1>
            <p className="max-w-2xl text-base text-slate-300">
              This frontend scaffold mirrors the FastAPI backend with typed clients, polling hooks,
              repository browsing, and route-level loading and error states.
            </p>
            <div className="flex flex-wrap gap-3">
              <Link
                className="inline-flex h-11 items-center justify-center rounded-xl bg-brand-600 px-5 text-sm font-medium text-white transition hover:bg-brand-500"
                href="/ingest"
              >
                Ingest Repository
              </Link>
              <Link
                className="inline-flex h-11 items-center justify-center rounded-xl border border-surface-border bg-white/5 px-5 text-sm font-medium text-slate-100 transition hover:bg-surface-hover"
                href="/repos"
              >
                Browse Repositories
              </Link>
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-1">
            <Card padding="sm">
              <p className="text-xs uppercase tracking-[0.14em] text-surface-muted">Indexed repositories</p>
              <p className="mt-2 text-3xl font-semibold text-white">
                {isLoading ? "…" : formatNumber(repos.length)}
              </p>
            </Card>
            <Card padding="sm">
              <p className="text-xs uppercase tracking-[0.14em] text-surface-muted">Primary flow</p>
              <p className="mt-2 text-sm text-slate-200">URL submit → polling → repository detail → ask page</p>
            </Card>
            <Card padding="sm">
              <p className="text-xs uppercase tracking-[0.14em] text-surface-muted">Platform stack</p>
              <p className="mt-2 text-sm text-slate-200">App Router, Tailwind UI kit, typed hooks, Docker and CI wiring.</p>
            </Card>
          </div>
        </div>
      </Card>

      <section className="space-y-4">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h2 className="text-2xl font-semibold text-white">Recent repositories</h2>
            <p className="text-sm text-surface-muted">The latest indexed repositories are ready to browse.</p>
          </div>
          <Link className="text-sm font-medium text-brand-200 transition hover:text-brand-100" href="/repos">
            View all
          </Link>
        </div>

        {isLoading ? (
          <div className="grid gap-4 lg:grid-cols-3">
            {Array.from({ length: 3 }).map((_, index) => (
              <Card key={index} padding="lg">
                <Skeleton className="h-6 w-32" />
                <Skeleton className="mt-4 h-28 w-full" />
              </Card>
            ))}
          </div>
        ) : repos.length ? (
          <div className="grid gap-4 lg:grid-cols-3">
            {repos.slice(0, 3).map((repo) => (
              <RepoCard key={repo.id} repo={repo} />
            ))}
          </div>
        ) : (
          <Card padding="lg">
            <p className="text-lg font-medium text-white">No repositories indexed yet.</p>
            <p className="mt-2 text-sm text-surface-muted">
              Start with the ingestion page and point the platform at a GitHub repository.
            </p>
          </Card>
        )}
      </section>
    </div>
  );
}
