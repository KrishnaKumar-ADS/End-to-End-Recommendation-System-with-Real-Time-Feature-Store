"use client";

import Link from "next/link";
import { useCallback } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

import { RepoCard } from "@/components/repos/RepoCard";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Select } from "@/components/ui/Select";
import { useRepos } from "@/hooks/useRepos";
import { formatNumber } from "@/lib/utils";

const DEFAULT_PAGE = 1;
const DEFAULT_PAGE_SIZE = 12;
const PAGE_SIZE_OPTIONS = [6, 12, 24, 48];

function parsePositiveInteger(value: string | null, fallback: number): number {
  if (!value) {
    return fallback;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

export default function ReposPage() {
  const pathname = usePathname();
  const router = useRouter();
  const searchParams = useSearchParams();

  const page = parsePositiveInteger(searchParams.get("page"), DEFAULT_PAGE);
  const selectedPageSize = parsePositiveInteger(
    searchParams.get("pageSize"),
    DEFAULT_PAGE_SIZE,
  );
  const pageSize = PAGE_SIZE_OPTIONS.includes(selectedPageSize)
    ? selectedPageSize
    : DEFAULT_PAGE_SIZE;
  const offset = (page - 1) * pageSize;

  const { repos, isLoading, error } = useRepos(pageSize, offset);
  const canGoPrevious = page > 1;
  const canGoNext = repos.length === pageSize && !isLoading && !error;

  const updateQuery = useCallback(
    (nextPage: number, nextPageSize = pageSize) => {
      const params = new URLSearchParams(searchParams.toString());

      if (nextPage <= DEFAULT_PAGE) {
        params.delete("page");
      } else {
        params.set("page", String(nextPage));
      }

      if (nextPageSize === DEFAULT_PAGE_SIZE) {
        params.delete("pageSize");
      } else {
        params.set("pageSize", String(nextPageSize));
      }

      const nextQuery = params.toString();
      router.replace(nextQuery ? `${pathname}?${nextQuery}` : pathname, { scroll: false });
    },
    [pageSize, pathname, router, searchParams],
  );

  return (
    <div className="space-y-6">
      <Card padding="lg">
        <p className="text-xs uppercase tracking-[0.18em] text-brand-200">Repository catalog</p>
        <h1 className="mt-3 text-4xl font-semibold text-white">All indexed repositories</h1>
        <p className="mt-3 max-w-3xl text-sm text-surface-muted">
          Browse the repositories stored in PostgreSQL, inspect indexing status, and jump straight into detail or ask flows.
        </p>
      </Card>

      <Card padding="sm">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div className="space-y-1">
            <p className="text-xs uppercase tracking-[0.14em] text-brand-200">Pagination</p>
            <p className="text-sm text-surface-muted">
              Page {formatNumber(page)} · showing {formatNumber(repos.length)} repositories on this page
            </p>
          </div>

          <div className="flex flex-wrap items-end gap-3">
            <Select
              className="min-w-[130px]"
              label="Per page"
              onChange={(event) => updateQuery(DEFAULT_PAGE, Number(event.target.value))}
              value={String(pageSize)}
            >
              {PAGE_SIZE_OPTIONS.map((sizeOption) => (
                <option key={sizeOption} value={sizeOption}>
                  {sizeOption}
                </option>
              ))}
            </Select>

            <Button
              disabled={!canGoPrevious || isLoading}
              onClick={() => updateQuery(page - 1)}
              variant="secondary"
            >
              Previous
            </Button>
            <Button
              disabled={!canGoNext || isLoading}
              onClick={() => updateQuery(page + 1)}
              variant="secondary"
            >
              Next
            </Button>
          </div>
        </div>
      </Card>

      {error ? (
        <Card padding="lg">
          <p className="text-sm text-red-300">{error}</p>
        </Card>
      ) : null}

      {isLoading ? null : repos.length ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {repos.map((repo) => (
            <RepoCard key={repo.id} repo={repo} />
          ))}
        </div>
      ) : (
        <Card padding="lg">
          {page > DEFAULT_PAGE ? (
            <>
              <p className="text-lg font-medium text-white">No repositories found on this page.</p>
              <p className="mt-2 text-sm text-surface-muted">
                Try moving to the previous page or reset to page 1.
              </p>
              <div className="mt-4 flex flex-wrap gap-3">
                <Button
                  disabled={!canGoPrevious}
                  onClick={() => updateQuery(page - 1)}
                  variant="secondary"
                >
                  Previous page
                </Button>
                <Button onClick={() => updateQuery(DEFAULT_PAGE, pageSize)}>
                  Go to first page
                </Button>
              </div>
            </>
          ) : (
            <>
              <p className="text-lg font-medium text-white">No repositories indexed yet.</p>
              <p className="mt-2 text-sm text-surface-muted">
                Head to the ingest page to start indexing your first GitHub repository.
              </p>
              <Link
                className="mt-4 inline-flex h-11 items-center justify-center rounded-xl bg-brand-600 px-4 text-sm font-medium text-white transition hover:bg-brand-500"
                href="/ingest"
              >
                Ingest Repository
              </Link>
            </>
          )}
        </Card>
      )}
    </div>
  );
}
