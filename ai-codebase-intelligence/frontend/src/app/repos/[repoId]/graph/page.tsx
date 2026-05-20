"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";

import { ForceGraph, type ForceGraphHandle, type HoverPayload } from "@/components/Graph/ForceGraph";
import { GraphControls } from "@/components/Graph/GraphControls";
import { GraphFilterPanel } from "@/components/Graph/GraphFilterPanel";
import { GraphLegend } from "@/components/Graph/GraphLegend";
import { GraphStats } from "@/components/Graph/GraphStats";
import { GraphTooltip } from "@/components/Graph/GraphTooltip";
import { NodeDetailPanel } from "@/components/Graph/NodeDetailPanel";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Spinner } from "@/components/ui/Spinner";
import { useGraph } from "@/hooks/useGraph";
import { useRepo } from "@/hooks/useRepo";
import { getNeighbors, getNodeFilePath, getNodeLabel } from "@/lib/graph-utils";
import { repoNameFromUrl } from "@/lib/utils";

export default function RepoGraphPage({
  params,
}: {
  params: { repoId: string };
}) {
  const searchParams = useSearchParams();
  const { repo } = useRepo(params.repoId);

  const {
    graph,
    nodes,
    edges,
    isLoading,
    isSubgraphLoading,
    error,
    refresh,
    filters,
    typeCounts,
    toggleNodeType,
    setSearch,
    setShowIsolated,
    setMinDegree,
    setDepth,
    resetFilters,
    selectedNode,
    selectedNodeId,
    selectNode,
    clearSelection,
    subgraphMode,
    exploreSubgraph,
    clearSubgraph,
    graphStats,
    searchMatches,
    degreeCentrality,
  } = useGraph(params.repoId, 5000);

  const graphRef = useRef<ForceGraphHandle>(null);
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const [alpha, setAlpha] = useState(1);
  const [hovered, setHovered] = useState<HoverPayload | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [simulationPaused, setSimulationPaused] = useState(false);
  const [filtersOpen, setFiltersOpen] = useState(false);

  const focusPath = searchParams.get("file");

  useEffect(() => {
    const element = canvasRef.current;
    if (!element) {
      return;
    }

    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (!rect) {
        return;
      }
      setSize({
        width: Math.floor(rect.width),
        height: Math.floor(rect.height),
      });
    });

    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!focusPath || !nodes.length || selectedNodeId) {
      return;
    }

    const match = nodes.find((node) => {
      const path = getNodeFilePath(node);
      return path === focusPath || path.endsWith(focusPath);
    });

    if (match) {
      selectNode(match.id);
    }
  }, [focusPath, nodes, selectNode, selectedNodeId]);

  const hoveredNeighbors = useMemo(() => {
    if (!hovered) {
      return { callers: [], callees: [] };
    }
    return getNeighbors(hovered.node.id, edges);
  }, [edges, hovered]);

  const repoName = repo ? repoNameFromUrl(repo.github_url) : params.repoId;

  const toggleSimulation = () => {
    if (graphRef.current?.isPaused()) {
      graphRef.current.resumeSimulation();
      setSimulationPaused(false);
      return;
    }
    graphRef.current?.pauseSimulation();
    setSimulationPaused(true);
  };

  if (isLoading && !graph) {
    return (
      <Card className="space-y-4" padding="lg">
        <div className="flex items-center gap-3">
          <Spinner size="sm" />
          <p className="text-sm text-slate-200">Laying out graph...</p>
        </div>
        <div className="h-[60vh] rounded-2xl border border-surface-border bg-surface-card" />
      </Card>
    );
  }

  return (
    <div className="flex h-[calc(100vh-73px)] min-h-0 flex-col gap-3">
      <div className="rounded-2xl border border-surface-border bg-surface-card p-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-3">
            <Link className="text-sm text-brand-200 transition hover:text-brand-100" href={`/repos/${params.repoId}`}>
              ← Files
            </Link>
            <h1 className="truncate text-lg font-semibold text-white">{repoName} Graph</h1>
            {subgraphMode !== "none" ? (
              <Button onClick={clearSubgraph} size="sm" variant="secondary">
                Clear subgraph
              </Button>
            ) : null}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button className="md:hidden" onClick={() => setFiltersOpen(true)} size="sm" variant="secondary">
              Filters
            </Button>
            <GraphControls
              edges={edges}
              graphRef={graphRef}
              nodes={nodes}
              onToggleSimulation={toggleSimulation}
              repoId={params.repoId}
              simulationPaused={simulationPaused}
            />
          </div>
        </div>

        <div className="mt-3 overflow-x-auto">
          <GraphStats
            alpha={alpha}
            connectedComponents={graphStats.connectedComponents}
            entryPointCount={graphStats.entryPointCount}
            mostConnectedNode={graphStats.mostConnectedNode}
            subgraphMode={subgraphMode}
            totalEdges={graphStats.totalEdges}
            totalNodes={graphStats.totalNodes}
            visibleEdges={graphStats.visibleEdges}
            visibleNodes={graphStats.visibleNodes}
          />
        </div>
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 lg:grid-cols-[280px,1fr,320px]">
        <div className="hidden min-h-0 lg:block">
          <GraphFilterPanel
            filters={filters}
            onDepthChange={setDepth}
            onMinDegreeChange={setMinDegree}
            onReset={resetFilters}
            onSearchChange={setSearch}
            onShowIsolatedChange={setShowIsolated}
            onToggleNodeType={toggleNodeType}
            typeCounts={typeCounts}
          />
        </div>

        <div className="relative min-h-0 overflow-hidden rounded-2xl border border-surface-border bg-surface-card">
          {error ? (
            <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-3 bg-black/40 backdrop-blur-sm">
              <p className="text-sm text-red-200">{error}</p>
              <Button onClick={() => void refresh()} size="sm" variant="secondary">
                Retry graph load
              </Button>
            </div>
          ) : null}

          {isSubgraphLoading ? (
            <div className="absolute left-3 top-3 z-30 rounded-lg border border-surface-border bg-surface/85 px-3 py-2 text-xs text-slate-200">
              Loading subgraph...
            </div>
          ) : null}

          <div className="h-full w-full" ref={canvasRef}>
            {size.width > 0 && size.height > 0 ? (
              <ForceGraph
                degreeCentrality={degreeCentrality}
                edges={edges}
                height={size.height}
                nodes={nodes}
                onNodeClick={selectNode}
                onNodeHover={setHovered}
                onSimulationAlphaChange={setAlpha}
                ref={graphRef}
                searchMatches={searchMatches}
                searchTerm={filters.search}
                selectedNodeId={selectedNodeId}
                width={size.width}
              />
            ) : null}
          </div>

          <GraphLegend />
        </div>

        <div className="hidden min-h-0 lg:block">
          <NodeDetailPanel
            allEdges={edges}
            allNodes={nodes}
            depth={filters.depth}
            node={selectedNode}
            onClose={clearSelection}
            onDepthChange={setDepth}
            onExplore={exploreSubgraph}
            onSelectNode={(nodeId) => selectNode(nodeId)}
            repoId={params.repoId}
          />
        </div>
      </div>

      <GraphTooltip
        calleeCount={hoveredNeighbors.callees.length}
        callerCount={hoveredNeighbors.callers.length}
        node={hovered?.node ?? null}
        x={hovered?.x ?? 0}
        y={hovered?.y ?? 0}
      />

      {filtersOpen ? (
        <div className="fixed inset-0 z-50 flex items-end bg-black/50 p-3 lg:hidden" onClick={() => setFiltersOpen(false)}>
          <div className="w-full" onClick={(event) => event.stopPropagation()}>
            <GraphFilterPanel
              filters={filters}
              onDepthChange={setDepth}
              onMinDegreeChange={setMinDegree}
              onReset={resetFilters}
              onSearchChange={setSearch}
              onShowIsolatedChange={setShowIsolated}
              onToggleNodeType={toggleNodeType}
              typeCounts={typeCounts}
            />
          </div>
        </div>
      ) : null}

      {selectedNode ? (
        <div className="fixed inset-0 z-50 flex items-end bg-black/50 p-3 lg:hidden" onClick={clearSelection}>
          <div className="max-h-[80vh] w-full" onClick={(event) => event.stopPropagation()}>
            <NodeDetailPanel
              allEdges={edges}
              allNodes={nodes}
              depth={filters.depth}
              node={selectedNode}
              onClose={clearSelection}
              onDepthChange={setDepth}
              onExplore={exploreSubgraph}
              onSelectNode={(nodeId) => selectNode(nodeId)}
              repoId={params.repoId}
            />
          </div>
        </div>
      ) : null}
    </div>
  );
}
