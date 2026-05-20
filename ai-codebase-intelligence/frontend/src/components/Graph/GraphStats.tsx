"use client";

import { getNodeLabel } from "@/lib/graph-utils";
import { formatNumber } from "@/lib/utils";
import type { GraphNode } from "@/types/api";

interface GraphStatsProps {
  totalNodes: number;
  totalEdges: number;
  visibleNodes: number;
  visibleEdges: number;
  connectedComponents: number;
  entryPointCount: number;
  mostConnectedNode: GraphNode | null;
  alpha: number;
  subgraphMode: "none" | "client" | "server";
}

function statPill(label: string, value: string) {
  return (
    <div className="rounded-full border border-surface-border bg-surface-card px-3 py-1 text-xs text-slate-200">
      <span className="text-surface-muted">{label}: </span>
      <span className="font-semibold text-white">{value}</span>
    </div>
  );
}

export function GraphStats({
  totalNodes,
  totalEdges,
  visibleNodes,
  visibleEdges,
  connectedComponents,
  entryPointCount,
  mostConnectedNode,
  alpha,
  subgraphMode,
}: GraphStatsProps) {
  const settled = alpha <= 0.01;

  return (
    <div className="flex min-w-0 flex-wrap items-center gap-2">
      {statPill("Nodes", `${formatNumber(visibleNodes)} / ${formatNumber(totalNodes)}`)}
      {statPill("Edges", `${formatNumber(visibleEdges)} / ${formatNumber(totalEdges)}`)}
      {statPill("Components", formatNumber(connectedComponents))}
      {statPill("Entry points", formatNumber(entryPointCount))}
      {statPill("Top hub", mostConnectedNode ? getNodeLabel(mostConnectedNode) : "n/a")}

      <div
        className={`rounded-full border px-3 py-1 text-xs ${settled ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-200" : "border-amber-500/40 bg-amber-500/10 text-amber-200"}`}
      >
        {settled ? "Stable" : "Settling..."}
      </div>

      {subgraphMode !== "none" ? (
        <div className="rounded-full border border-brand-500/40 bg-brand-500/10 px-3 py-1 text-xs text-brand-100">
          Subgraph mode
        </div>
      ) : null}
    </div>
  );
}
