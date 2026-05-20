"use client";

import { useEffect, useMemo } from "react";
import { useRouter } from "next/navigation";

import { IngestForm } from "@/components/ingest/IngestForm";
import { IngestionProgress } from "@/components/ingest/IngestionProgress";
import { Card } from "@/components/ui/Card";
import { useToast } from "@/components/ui/Toast";
import { useGraph } from "@/hooks/useGraph";
import { useIngest } from "@/hooks/useIngest";
import { useRepo } from "@/hooks/useRepo";
import { useStatus } from "@/hooks/useStatus";
import { getRepoStats } from "@/lib/utils";

export default function IngestPage() {
  const router = useRouter();
  const { toast } = useToast();
  const ingest = useIngest();
  const statusState = useStatus(ingest.taskId);

  const completedRepoId =
    statusState.data?.status === "completed"
      ? statusState.data.repo_id ?? ingest.repoId
      : null;

  const { repo } = useRepo(completedRepoId);
  const { graph } = useGraph(completedRepoId, 1600);

  useEffect(() => {
    if (ingest.phase === "submitted" && ingest.taskId) {
      toast({
        title: "Ingestion queued",
        description: "The backend accepted the repository and started the indexing pipeline.",
        variant: "success",
      });
    }
  }, [ingest.phase, ingest.taskId, toast]);

  useEffect(() => {
    if (ingest.phase === "error" && ingest.error) {
      toast({
        title: ingest.isConflict ? "Repository already exists" : "Ingestion failed",
        description: ingest.error,
        variant: ingest.isConflict ? "warning" : "error",
      });
    }
  }, [ingest.error, ingest.isConflict, ingest.phase, toast]);

  const summary = useMemo(() => {
    if (!repo || !graph) {
      return null;
    }

    const stats = getRepoStats(repo, graph);
    return {
      repoId: repo.id,
      totalFiles: stats.files,
      functions: stats.functions,
      totalChunks: stats.embeddings,
      graphNodes: stats.graphNodes,
    };
  }, [graph, repo]);

  return (
    <div className="space-y-6">
      <Card padding="lg">
        <p className="text-xs uppercase tracking-[0.18em] text-brand-200">Repository ingestion</p>
        <h1 className="mt-3 text-4xl font-semibold text-white">Turn a GitHub repository into a queryable knowledge base.</h1>
        <p className="mt-3 max-w-3xl text-sm text-surface-muted">
          This route covers the core flow: submit a repository URL, poll live status every 2 seconds,
          and transition into the repository or ask experience once indexing completes.
        </p>
      </Card>

      <IngestForm error={ingest.error} isLoading={ingest.isLoading} onSubmit={ingest.submit} />

      <IngestionProgress
        elapsedSec={statusState.elapsedSec}
        onAsk={() => {
          if (summary?.repoId) {
            router.push(`/repos/${summary.repoId}/chat`);
          }
        }}
        onBrowse={() => {
          if (summary?.repoId) {
            router.push(`/repos/${summary.repoId}`);
          }
        }}
        onRetry={ingest.reset}
        status={statusState.data}
        summary={summary}
      />
    </div>
  );
}
