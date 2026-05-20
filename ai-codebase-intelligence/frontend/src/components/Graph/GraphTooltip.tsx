"use client";

import { useEffect, useMemo, useState } from "react";

import { getNodeFilePath, getNodeLabel, normalizeGraphNodeType } from "@/lib/graph-utils";
import type { GraphNode } from "@/types/api";

interface GraphTooltipProps {
  node: GraphNode | null;
  x: number;
  y: number;
  callerCount: number;
  calleeCount: number;
}

export function GraphTooltip({ node, x, y, callerCount, calleeCount }: GraphTooltipProps) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (!node) {
      setVisible(false);
      return;
    }

    const timer = window.setTimeout(() => {
      setVisible(true);
    }, 200);

    return () => {
      window.clearTimeout(timer);
    };
  }, [node]);

  const position = useMemo(() => {
    const margin = 16;
    const width = 280;
    const height = 130;
    const viewportW = typeof window === "undefined" ? 1280 : window.innerWidth;
    const viewportH = typeof window === "undefined" ? 720 : window.innerHeight;

    let left = x + 14;
    let top = y + 14;

    if (left + width > viewportW - margin) {
      left = x - width - 14;
    }
    if (top + height > viewportH - margin) {
      top = y - height - 14;
    }

    return {
      left: Math.max(margin, left),
      top: Math.max(margin, top),
    };
  }, [x, y]);

  if (!node || !visible) {
    return null;
  }

  return (
    <div
      className="pointer-events-none fixed z-40 w-[280px] rounded-xl border border-surface-border bg-surface/95 px-3 py-2 text-xs text-slate-200 shadow-xl backdrop-blur"
      style={position}
    >
      <p className="truncate font-semibold text-white">{getNodeLabel(node)}</p>
      <p className="mt-0.5 capitalize text-brand-200">{normalizeGraphNodeType(node)}</p>
      <p className="mt-1 truncate text-surface-muted">{getNodeFilePath(node) || "No file path"}</p>
      <div className="mt-2 flex items-center gap-3 text-[11px] text-surface-muted">
        <span>callers: {callerCount}</span>
        <span>callees: {calleeCount}</span>
      </div>
    </div>
  );
}
