"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { FileTree } from "@/components/CodeViewer/FileTree";
import { Sidebar } from "@/components/Layout/Sidebar";
import { Badge, ProviderBadge, StatusBadge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { useToast } from "@/components/ui/Toast";
import { useGraph } from "@/hooks/useGraph";
import { useRepo } from "@/hooks/useRepo";
import { deleteRepository } from "@/lib/api-client";
import {
  buildFileTree,
  extractFileSymbols,
  formatBytes,
  formatNumber,
  getFileLanguageLabel,
  getRepoStats,
} from "@/lib/utils";
import type { RepositoryFile } from "@/types/api";

export default function RepoDetailPage({
  params,
}: {
  params: { repoId: string };
}) {
  const router = useRouter();
  const { toast } = useToast();
  const searchParams = useSearchParams();
  const requestedPath = searchParams.get("file");
  const { repo, isLoading, error } = useRepo(params.repoId);
  const { graph, error: graphError } = useGraph(params.repoId, 1600);
  const [selectedFile, setSelectedFile] = useState<RepositoryFile | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const tree = useMemo(() => buildFileTree(repo?.files ?? []), [repo?.files]);
  const stats = useMemo(() => (repo ? getRepoStats(repo, graph) : null), [graph, repo]);

  useEffect(() => {
    if (!repo?.files.length) {
      setSelectedFile(null);
      return;
    }

    const matchingFile = requestedPath
      ? repo.files.find((file) => file.file_path === requestedPath)
      : null;

    setSelectedFile((current) => {
      if (matchingFile) {
        return matchingFile;
      }
      if (current) {
        const stillExists = repo.files.find((file) => file.id === current.id);
        if (stillExists) {
          return stillExists;
        }
      }
      return repo.files[0];
    });
  }, [repo, requestedPath]);

  const symbols = useMemo(
    () => (selectedFile ? extractFileSymbols(graph, selectedFile.file_path) : { functions: [], classes: [] }),
    [graph, selectedFile],
  );

  if (isLoading && !repo) {
    return null;
  }

  if (error || !repo) {
    return (
      <Card padding="lg">
        <p className="text-sm text-red-300">{error ?? "Repository not found."}</p>
      </Card>
    );
  }

  const handleDelete = async () => {
    const confirmed = window.confirm(
      `Delete ${repo.name}? This removes indexed files, chunks, vectors, and graph data for this repository.`,
    );

    if (!confirmed) {
      return;
    }

    setIsDeleting(true);
    try {
      const result = await deleteRepository(repo.id);
      toast({
        title: "Repository deleted",
        description: result.message,
        variant: "success",
      });
      router.push("/repos");
    } catch (err) {
      toast({
        title: "Delete failed",
        description: err instanceof Error ? err.message : "Could not delete this repository.",
        variant: "error",
      });
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[320px,1fr]">
      <Sidebar
        description="Click a file to open chat with a prefilled explanation question."
        title="Repository file tree"
      >
        <FileTree
          nodes={tree}
          onFileClick={(file) => {
            const question = encodeURIComponent(`Explain the file ${file.file_path}`);
            router.push(`/repos/${repo.id}/chat?question=${question}`);
          }}
          onFileGraphClick={(file) => {
            router.push(`/repos/${repo.id}/graph?file=${encodeURIComponent(file.file_path)}`);
          }}
          selectedPath={selectedFile?.file_path}
        />
      </Sidebar>

      <div className="space-y-6">
        <Card padding="lg">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="space-y-2">
              <p className="text-xs uppercase tracking-[0.18em] text-brand-200">Repository detail</p>
              <h1 className="text-4xl font-semibold text-white">{repo.name}</h1>
              <p className="text-sm text-surface-muted">{repo.github_url}</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <StatusBadge status={repo.status} />
              <Badge>Branch: {repo.branch}</Badge>
              <ProviderBadge model="Ready for Q&A" provider="pipeline" />
            </div>
          </div>

          <div className="mt-5 flex flex-wrap gap-2">
            <Link
              className="inline-flex items-center rounded-full border border-brand-500/40 bg-brand-500/10 px-3 py-1.5 text-xs font-medium text-brand-100"
              href={`/repos/${repo.id}`}
            >
              Files
            </Link>
            <Link
              className="inline-flex items-center rounded-full border border-surface-border bg-surface-card px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:border-brand-500/40 hover:text-white"
              href={`/repos/${repo.id}/chat`}
            >
              Chat
            </Link>
            <Link
              className="inline-flex items-center gap-1 rounded-full border border-surface-border bg-surface-card px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:border-brand-500/40 hover:text-white"
              href={`/repos/${repo.id}/graph`}
            >
              <span aria-hidden="true">⬡</span>
              Graph
            </Link>
            <Button
              className="rounded-full px-3 py-1.5 text-xs"
              isLoading={isDeleting}
              onClick={() => {
                void handleDelete();
              }}
              size="sm"
              variant="danger"
            >
              Delete Repository
            </Button>
          </div>

          <div className="mt-6 flex gap-3 overflow-x-auto pb-1">
            <Card className="min-w-[170px]" padding="sm">
              <p className="text-xs uppercase tracking-[0.12em] text-surface-muted">Files</p>
              <p className="mt-2 text-2xl font-semibold text-white">{formatNumber(stats?.files ?? 0)}</p>
            </Card>
            <Card className="min-w-[170px]" padding="sm">
              <p className="text-xs uppercase tracking-[0.12em] text-surface-muted">Functions</p>
              <p className="mt-2 text-2xl font-semibold text-white">{formatNumber(stats?.functions ?? 0)}</p>
            </Card>
            <Card className="min-w-[170px]" padding="sm">
              <p className="text-xs uppercase tracking-[0.12em] text-surface-muted">Classes</p>
              <p className="mt-2 text-2xl font-semibold text-white">{formatNumber(stats?.classes ?? 0)}</p>
            </Card>
            <Card className="min-w-[170px]" padding="sm">
              <p className="text-xs uppercase tracking-[0.12em] text-surface-muted">Embeddings</p>
              <p className="mt-2 text-2xl font-semibold text-white">{formatNumber(stats?.embeddings ?? 0)}</p>
            </Card>
            <Card className="min-w-[170px]" padding="sm">
              <p className="text-xs uppercase tracking-[0.12em] text-surface-muted">Graph nodes</p>
              <p className="mt-2 text-2xl font-semibold text-white">{formatNumber(stats?.graphNodes ?? 0)}</p>
            </Card>
          </div>
        </Card>

        <Card padding="lg">
          {selectedFile ? (
            <div className="space-y-6">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div className="space-y-2">
                  <p className="text-xs uppercase tracking-[0.14em] text-brand-200">Selected file</p>
                  <h2 className="text-2xl font-semibold text-white">{selectedFile.file_path}</h2>
                  <div className="flex flex-wrap gap-2">
                    <Badge>{getFileLanguageLabel(selectedFile.language)}</Badge>
                    <Badge>{formatBytes(selectedFile.size_bytes)}</Badge>
                    <Badge>{formatNumber(selectedFile.line_count)} lines</Badge>
                    <Badge>{formatNumber(selectedFile.chunk_count)} chunks</Badge>
                  </div>
                </div>

                <Link
                  className="inline-flex h-11 items-center justify-center rounded-xl bg-brand-600 px-4 text-sm font-medium text-white transition hover:bg-brand-500"
                  href={{
                    pathname: `/repos/${repo.id}/chat`,
                    query: { question: `Explain the file ${selectedFile.file_path}` },
                  }}
                >
                  Ask about this file
                </Link>
              </div>

              <div className="grid gap-4 lg:grid-cols-2">
                <Card padding="sm">
                  <p className="text-xs uppercase tracking-[0.12em] text-surface-muted">Functions</p>
                  <div className="mt-3 space-y-2">
                    {symbols.functions.length ? (
                      symbols.functions.map((symbol) => (
                        <div key={symbol.id} className="rounded-2xl border border-surface-border bg-white/4 px-3 py-2">
                          <p className="font-medium text-white">{symbol.name}</p>
                          <p className="text-xs text-surface-muted">Line {formatNumber(symbol.startLine || 0)}</p>
                        </div>
                      ))
                    ) : (
                      <p className="text-sm text-surface-muted">No function nodes were found for this file.</p>
                    )}
                  </div>
                </Card>

                <Card padding="sm">
                  <p className="text-xs uppercase tracking-[0.12em] text-surface-muted">Classes</p>
                  <div className="mt-3 space-y-2">
                    {symbols.classes.length ? (
                      symbols.classes.map((symbol) => (
                        <div key={symbol.id} className="rounded-2xl border border-surface-border bg-white/4 px-3 py-2">
                          <p className="font-medium text-white">{symbol.name}</p>
                          <p className="text-xs text-surface-muted">Line {formatNumber(symbol.startLine || 0)}</p>
                        </div>
                      ))
                    ) : (
                      <p className="text-sm text-surface-muted">No class nodes were found for this file.</p>
                    )}
                  </div>
                </Card>
              </div>
            </div>
          ) : (
            <p className="text-sm text-surface-muted">Select a file from the tree to inspect it.</p>
          )}
        </Card>

        {graphError ? (
          <Card padding="sm">
            <p className="text-sm text-amber-200">
              Graph data could not be loaded, so function and class extraction may be incomplete.
            </p>
          </Card>
        ) : null}
      </div>
    </div>
  );
}
