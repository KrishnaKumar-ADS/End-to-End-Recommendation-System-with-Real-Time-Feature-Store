"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import { Button } from "@/components/ui/Button";
import {
  NODE_COLORS,
  getNeighbors,
  getNodeFilePath,
  getNodeLabel,
  normalizeGraphNodeType,
} from "@/lib/graph-utils";
import { explainSymbol } from "@/lib/api-client";
import type { GraphEdge, GraphNode } from "@/types/api";

const descriptionCache = new Map<string, string>();

interface NodeDetailPanelProps {
  repoId: string;
  node: GraphNode | null;
  allNodes: GraphNode[];
  allEdges: GraphEdge[];
  depth: number;
  onDepthChange: (value: number) => void;
  onExplore: (nodeId: string, depth: number) => void;
  onSelectNode: (nodeId: string) => void;
  onClose: () => void;
}

function asNodeMap(nodes: GraphNode[]): Map<string, GraphNode> {
  return new Map(nodes.map((node) => [node.id, node]));
}

export function NodeDetailPanel({
  repoId,
  node,
  allNodes,
  allEdges,
  depth,
  onDepthChange,
  onExplore,
  onSelectNode,
  onClose,
}: NodeDetailPanelProps) {
  const router = useRouter();
  const [description, setDescription] = useState<string>("");
  const [isLoadingDescription, setIsLoadingDescription] = useState(false);

  const nodeMap = useMemo(() => asNodeMap(allNodes), [allNodes]);

  const neighborInfo = useMemo(() => {
    if (!node) {
      return { callers: [], callees: [] };
    }
    return getNeighbors(node.id, allEdges);
  }, [allEdges, node]);

  useEffect(() => {
    if (!node) {
      setDescription("");
      return;
    }

    const cached = descriptionCache.get(node.id);
    if (cached) {
      setDescription(cached);
      return;
    }

    let mounted = true;
    setIsLoadingDescription(true);

    const load = async () => {
      try {
        const filePath = getNodeFilePath(node);
        const symbolName = getNodeLabel(node);
        const response = await explainSymbol({
          repo_id: repoId,
          file_path: filePath || undefined,
          function_name: symbolName || undefined,
        });

        const summary = response.summary || "No description available.";
        descriptionCache.set(node.id, summary);
        if (mounted) {
          setDescription(summary);
        }
      } catch {
        const fallback = "No generated description is available for this node yet.";
        descriptionCache.set(node.id, fallback);
        if (mounted) {
          setDescription(fallback);
        }
      } finally {
        if (mounted) {
          setIsLoadingDescription(false);
        }
      }
    };

    void load();

    return () => {
      mounted = false;
    };
  }, [node, repoId]);

  if (!node) {
    return (
      <aside className="rounded-2xl border border-surface-border bg-surface-card p-4 text-sm text-surface-muted">
        Click a node to inspect it.
      </aside>
    );
  }

  const nodeType = normalizeGraphNodeType(node);
  const label = getNodeLabel(node);
  const filePath = getNodeFilePath(node);
  const lineRange =
    typeof node.start_line === "number" && typeof node.end_line === "number"
      ? `${node.start_line}-${node.end_line}`
      : typeof node.start_line === "number"
        ? String(node.start_line)
        : "n/a";

  return (
    <aside className="h-full overflow-y-auto rounded-2xl border border-surface-border bg-surface-card p-4">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="text-base font-semibold text-white">{label}</h3>
          <p className="mt-1 text-xs text-surface-muted">{node.id}</p>
        </div>
        <button
          className="rounded-lg border border-surface-border px-2 py-1 text-xs text-surface-muted transition hover:text-white"
          onClick={onClose}
          type="button"
        >
          Close
        </button>
      </div>

      <div className="mt-3 space-y-2 text-sm text-slate-200">
        <div className="flex items-center gap-2">
          <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: NODE_COLORS[nodeType] }} />
          <span className="capitalize">{nodeType}</span>
        </div>
        <p className="truncate text-xs text-surface-muted">{filePath || "No file"}</p>
        <p className="text-xs text-surface-muted">Lines: {lineRange}</p>
      </div>

      <div className="mt-4 space-y-2">
        <p className="text-xs uppercase tracking-[0.09em] text-surface-muted">Description</p>
        <p className="text-sm leading-relaxed text-slate-200">
          {isLoadingDescription ? "Generating one-line summary..." : description}
        </p>
      </div>

      <div className="mt-4 space-y-2">
        <p className="text-xs uppercase tracking-[0.09em] text-surface-muted">Callers ({neighborInfo.callers.length})</p>
        <div className="flex flex-wrap gap-2">
          {neighborInfo.callers.length ? (
            neighborInfo.callers.map((callerId) => (
              <button
                className="rounded-full border border-surface-border bg-surface px-2.5 py-1 text-xs text-slate-200 transition hover:border-brand-500/40 hover:text-white"
                key={callerId}
                onClick={() => onSelectNode(callerId)}
                type="button"
              >
                {getNodeLabel(nodeMap.get(callerId) ?? { id: callerId })}
              </button>
            ))
          ) : (
            <span className="text-xs text-surface-muted">No callers</span>
          )}
        </div>
      </div>

      <div className="mt-4 space-y-2">
        <p className="text-xs uppercase tracking-[0.09em] text-surface-muted">Callees ({neighborInfo.callees.length})</p>
        <div className="flex flex-wrap gap-2">
          {neighborInfo.callees.length ? (
            neighborInfo.callees.map((calleeId) => (
              <button
                className="rounded-full border border-surface-border bg-surface px-2.5 py-1 text-xs text-slate-200 transition hover:border-brand-500/40 hover:text-white"
                key={calleeId}
                onClick={() => onSelectNode(calleeId)}
                type="button"
              >
                {getNodeLabel(nodeMap.get(calleeId) ?? { id: calleeId })}
              </button>
            ))
          ) : (
            <span className="text-xs text-surface-muted">No callees</span>
          )}
        </div>
      </div>

      <div className="mt-5">
        <div className="mb-1 flex items-center justify-between text-sm text-slate-200">
          <span>Depth</span>
          <span className="font-mono text-xs text-surface-muted">{depth}</span>
        </div>
        <input
          className="w-full"
          max={6}
          min={1}
          onChange={(event) => onDepthChange(Number(event.target.value))}
          type="range"
          value={depth}
        />
      </div>

      <div className="mt-5 space-y-2">
        <Button className="w-full" onClick={() => onExplore(node.id, depth)} size="sm" variant="secondary">
          Explore from here
        </Button>
        <Button
          className="w-full"
          onClick={() => {
            const question = encodeURIComponent(`Explain ${label}`);
            router.push(`/repos/${repoId}/chat?question=${question}`);
          }}
          size="sm"
          variant="ghost"
        >
          Ask about this
        </Button>
        <Button
          className="w-full"
          disabled={!filePath}
          onClick={() => {
            if (!filePath) {
              return;
            }
            router.push(`/repos/${repoId}?file=${encodeURIComponent(filePath)}`);
          }}
          size="sm"
          variant="ghost"
        >
          Go to file
        </Button>
      </div>
    </aside>
  );
}
