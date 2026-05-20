"use client";

import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
} from "react";
import * as d3 from "d3";

import {
  EDGE_COLORS,
  EDGE_DASH,
  edgeTypeLabel,
  getNodeLabel,
  nodeRadius,
} from "@/lib/graph-utils";
import type { GraphEdge, GraphNode } from "@/types/api";

type ForceGraphNode = GraphNode & d3.SimulationNodeDatum;

type ForceGraphLink = d3.SimulationLinkDatum<ForceGraphNode> & {
  source: string | ForceGraphNode;
  target: string | ForceGraphNode;
  type: string;
  id?: string;
};

export interface HoverPayload {
  node: GraphNode;
  x: number;
  y: number;
}

export interface ForceGraphHandle {
  zoomIn: () => void;
  zoomOut: () => void;
  resetView: () => void;
  fitToScreen: () => void;
  exportSvgString: () => string | null;
  pauseSimulation: () => void;
  resumeSimulation: () => void;
  isPaused: () => boolean;
}

interface ForceGraphProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  width: number;
  height: number;
  selectedNodeId: string | null;
  searchTerm: string;
  searchMatches: Set<string>;
  degreeCentrality: Record<string, number>;
  onNodeClick?: (nodeId: string | null) => void;
  onNodeHover?: (payload: HoverPayload | null) => void;
  onSimulationAlphaChange?: (alpha: number) => void;
}

function markerId(edgeType: string): string {
  return `arrow-${edgeType}`;
}

export const ForceGraph = forwardRef<ForceGraphHandle, ForceGraphProps>(function ForceGraph(
  {
    nodes,
    edges,
    width,
    height,
    selectedNodeId,
    searchTerm,
    searchMatches,
    degreeCentrality,
    onNodeClick,
    onNodeHover,
    onSimulationAlphaChange,
  },
  ref,
) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const simulationRef = useRef<d3.Simulation<ForceGraphNode, ForceGraphLink> | null>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const rootGroupRef = useRef<d3.Selection<SVGGElement, unknown, null, undefined> | null>(null);
  const pausedRef = useRef(false);

  const hasSearch = searchTerm.trim().length > 0;

  const degreeByNode = useMemo(() => {
    const degree: Record<string, number> = {};
    for (const node of nodes) {
      degree[node.id] = 0;
    }
    for (const edge of edges) {
      if (edge.source in degree) {
        degree[edge.source] += 1;
      }
      if (edge.target in degree) {
        degree[edge.target] += 1;
      }
    }
    return degree;
  }, [edges, nodes]);

  useEffect(() => {
    const svgNode = svgRef.current;
    if (!svgNode || width <= 0 || height <= 0) {
      return;
    }

    const svg = d3.select(svgNode);
    svg.selectAll("*").remove();

    svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("width", width)
      .attr("height", height)
      .attr("role", "img")
      .attr("aria-label", "Dependency graph visualization")
      .style("cursor", "grab")
      .style("touch-action", "none");

    const defs = svg.append("defs");
    for (const edgeType of ["calls", "imports", "inherits"]) {
      defs
        .append("marker")
        .attr("id", markerId(edgeType))
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 12)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .attr("fill", EDGE_COLORS[edgeType]);
    }

    const root = svg.append("g").attr("class", "graph-root");
    rootGroupRef.current = root;

    const linksLayer = root.append("g").attr("class", "graph-links");
    const nodesLayer = root.append("g").attr("class", "graph-nodes");
    const labelsLayer = root.append("g").attr("class", "graph-labels");

    const simNodes: ForceGraphNode[] = nodes.map((node) => ({ ...node }));
    const simLinks: ForceGraphLink[] = edges.map((edge) => ({ ...edge }));

    const linkSelection = linksLayer
      .selectAll<SVGLineElement, ForceGraphLink>("line")
      .data(simLinks, (d) => d.id ?? `${String(d.source)}-${String(d.target)}-${d.type}`)
      .join("line")
      .attr("class", "graph-link")
      .attr("stroke", (d) => EDGE_COLORS[edgeTypeLabel(d.type)] ?? EDGE_COLORS.calls)
      .attr("stroke-opacity", 0.5)
      .attr("stroke-width", 1.2)
      .attr("marker-end", (d) => `url(#${markerId(edgeTypeLabel(d.type))})`)
      .attr("stroke-dasharray", (d) => EDGE_DASH[edgeTypeLabel(d.type)] ?? null);

    const nodeSelection = nodesLayer
      .selectAll<SVGCircleElement, ForceGraphNode>("circle")
      .data(simNodes, (d) => d.id)
      .join("circle")
      .attr("class", "graph-node")
      .attr("r", (d) => nodeRadius(d.id, degreeCentrality))
      .attr("fill", "#3b82f6")
      .attr("stroke", "#0f172a")
      .attr("stroke-width", 1.5)
      .style("cursor", "pointer")
      .on("click", (event, d) => {
        event.stopPropagation();
        onNodeClick?.(d.id);
      })
      .on("mouseenter", (event, d) => {
        onNodeHover?.({ node: d, x: event.clientX, y: event.clientY });
      })
      .on("mousemove", (event, d) => {
        onNodeHover?.({ node: d, x: event.clientX, y: event.clientY });
      })
      .on("mouseleave", () => {
        onNodeHover?.(null);
      })
      .on("dblclick", (event, d) => {
        event.stopPropagation();
        d.fx = null;
        d.fy = null;
        simulationRef.current?.alpha(0.15).restart();
      });

    const labelSelection = labelsLayer
      .selectAll<SVGTextElement, ForceGraphNode>("text")
      .data(simNodes, (d) => d.id)
      .join("text")
      .attr("class", "graph-label")
      .text((d) => getNodeLabel(d))
      .attr("font-size", 11)
      .attr("fill", "#cbd5e1")
      .attr("pointer-events", "none")
      .style("user-select", "none");

    const drag = d3
      .drag<SVGCircleElement, ForceGraphNode>()
      .on("start", (event, d) => {
        if (!event.active) {
          simulationRef.current?.alphaTarget(0.16).restart();
        }
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", (event, d) => {
        if (!event.active) {
          simulationRef.current?.alphaTarget(0);
        }
        d.fx = event.x;
        d.fy = event.y;
      });

    nodeSelection.call(drag);

    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.05, 8])
      .on("zoom", (event) => {
        root.attr("transform", event.transform.toString());
      });

    zoomRef.current = zoom;
    svg.call(zoom);

    svg.on("click", () => {
      onNodeClick?.(null);
      onNodeHover?.(null);
    });

    const simulation = d3
      .forceSimulation<ForceGraphNode>(simNodes)
      .force(
        "link",
        d3
          .forceLink<ForceGraphNode, ForceGraphLink>(simLinks)
          .id((d) => d.id)
          .distance((link) => {
            const sourceId = typeof link.source === "string" ? link.source : link.source.id;
            const targetId = typeof link.target === "string" ? link.target : link.target.id;
            const sourceDegree = degreeByNode[sourceId] ?? 1;
            const targetDegree = degreeByNode[targetId] ?? 1;
            const avg = (sourceDegree + targetDegree) / 2;
            return 40 + Math.min(120, Math.sqrt(avg) * 18);
          })
          .strength((link) => {
            const sourceId = typeof link.source === "string" ? link.source : link.source.id;
            const targetId = typeof link.target === "string" ? link.target : link.target.id;
            const sourceDegree = degreeByNode[sourceId] ?? 1;
            const targetDegree = degreeByNode[targetId] ?? 1;
            return 1 / Math.max(1, Math.sqrt((sourceDegree + targetDegree) / 2));
          }),
      )
      .force("charge", d3.forceManyBody().strength(-200))
      .force("center", d3.forceCenter(width / 2, height / 2).strength(0.05))
      .force("collide", d3.forceCollide<ForceGraphNode>((d) => nodeRadius(d.id, degreeCentrality) + 4))
      .on("tick", () => {
        linkSelection
          .attr("x1", (d) => (typeof d.source === "string" ? 0 : d.source.x ?? 0))
          .attr("y1", (d) => (typeof d.source === "string" ? 0 : d.source.y ?? 0))
          .attr("x2", (d) => (typeof d.target === "string" ? 0 : d.target.x ?? 0))
          .attr("y2", (d) => (typeof d.target === "string" ? 0 : d.target.y ?? 0));

        nodeSelection.attr("cx", (d) => d.x ?? 0).attr("cy", (d) => d.y ?? 0);

        labelSelection
          .attr("x", (d) => (d.x ?? 0) + nodeRadius(d.id, degreeCentrality) + 4)
          .attr("y", (d) => (d.y ?? 0) + 3);

        onSimulationAlphaChange?.(simulation.alpha());
      })
      .on("end", () => {
        onSimulationAlphaChange?.(simulation.alpha());
      });

    simulationRef.current = simulation;
    pausedRef.current = false;

    return () => {
      simulation.stop();
      simulationRef.current = null;
      zoomRef.current = null;
      rootGroupRef.current = null;
      onNodeHover?.(null);
    };
  }, [degreeByNode, degreeCentrality, edges, height, nodes, onNodeClick, onNodeHover, onSimulationAlphaChange, width]);

  useEffect(() => {
    const svgNode = svgRef.current;
    if (!svgNode) {
      return;
    }

    const svg = d3.select(svgNode);

    svg
      .selectAll<SVGCircleElement, ForceGraphNode>("circle.graph-node")
      .attr("stroke", (d) => (d.id === selectedNodeId ? "#f59e0b" : "#0f172a"))
      .attr("stroke-width", (d) => (d.id === selectedNodeId ? 3 : 1.5))
      .attr("opacity", (d) => {
        if (!hasSearch) {
          return 1;
        }
        if (!searchMatches.size) {
          return 0.15;
        }
        return searchMatches.has(d.id) ? 1 : 0.1;
      });

    svg
      .selectAll<SVGTextElement, ForceGraphNode>("text.graph-label")
      .attr("fill", (d) => {
        if (d.id === selectedNodeId) {
          return "#fbbf24";
        }
        if (hasSearch && searchMatches.size && searchMatches.has(d.id)) {
          return "#fb923c";
        }
        return "#cbd5e1";
      })
      .attr("opacity", (d) => {
        if (!hasSearch) {
          return 0.95;
        }
        if (!searchMatches.size) {
          return 0.15;
        }
        return searchMatches.has(d.id) ? 1 : 0.1;
      });

    svg
      .selectAll<SVGLineElement, ForceGraphLink>("line.graph-link")
      .attr("opacity", (d) => {
        const sourceId = typeof d.source === "string" ? d.source : d.source.id;
        const targetId = typeof d.target === "string" ? d.target : d.target.id;

        if (selectedNodeId && (sourceId === selectedNodeId || targetId === selectedNodeId)) {
          return 0.9;
        }

        if (!hasSearch) {
          return 0.5;
        }
        if (!searchMatches.size) {
          return 0.08;
        }
        return searchMatches.has(sourceId) || searchMatches.has(targetId) ? 0.6 : 0.08;
      })
      .attr("stroke-width", (d) => {
        const sourceId = typeof d.source === "string" ? d.source : d.source.id;
        const targetId = typeof d.target === "string" ? d.target : d.target.id;
        return selectedNodeId && (sourceId === selectedNodeId || targetId === selectedNodeId) ? 2.2 : 1.2;
      });
  }, [hasSearch, searchMatches, selectedNodeId]);

  useImperativeHandle(ref, () => ({
    zoomIn: () => {
      const svgNode = svgRef.current;
      const zoom = zoomRef.current;
      if (!svgNode || !zoom) {
        return;
      }
      d3.select(svgNode).transition().duration(160).call(zoom.scaleBy, 1.2);
    },
    zoomOut: () => {
      const svgNode = svgRef.current;
      const zoom = zoomRef.current;
      if (!svgNode || !zoom) {
        return;
      }
      d3.select(svgNode).transition().duration(160).call(zoom.scaleBy, 0.85);
    },
    resetView: () => {
      const svgNode = svgRef.current;
      const zoom = zoomRef.current;
      if (!svgNode || !zoom) {
        return;
      }
      d3.select(svgNode)
        .transition()
        .duration(220)
        .call(zoom.transform, d3.zoomIdentity);
    },
    fitToScreen: () => {
      const svgNode = svgRef.current;
      const zoom = zoomRef.current;
      const root = rootGroupRef.current;
      if (!svgNode || !zoom || !root) {
        return;
      }

      const bounds = (root.node() as SVGGElement | null)?.getBBox();
      if (!bounds || bounds.width <= 0 || bounds.height <= 0) {
        return;
      }

      const fullWidth = svgNode.clientWidth || width;
      const fullHeight = svgNode.clientHeight || height;
      const scale = Math.max(0.05, Math.min(8, 0.9 / Math.max(bounds.width / fullWidth, bounds.height / fullHeight)));
      const translateX = fullWidth / 2 - scale * (bounds.x + bounds.width / 2);
      const translateY = fullHeight / 2 - scale * (bounds.y + bounds.height / 2);
      const transform = d3.zoomIdentity.translate(translateX, translateY).scale(scale);

      d3.select(svgNode).transition().duration(240).call(zoom.transform, transform);
    },
    exportSvgString: () => {
      const svgNode = svgRef.current;
      if (!svgNode) {
        return null;
      }
      const serializer = new XMLSerializer();
      return serializer.serializeToString(svgNode);
    },
    pauseSimulation: () => {
      simulationRef.current?.stop();
      pausedRef.current = true;
    },
    resumeSimulation: () => {
      simulationRef.current?.alpha(0.16).restart();
      pausedRef.current = false;
    },
    isPaused: () => pausedRef.current,
  }), [height, width]);

  return (
    <svg
      className="h-full w-full rounded-2xl border border-surface-border bg-[#0b1220]"
      ref={svgRef}
    />
  );
});
