"use client";

import { useEffect, useState } from "react";

import { Button } from "@/components/ui/Button";
import type { GraphFilterState, GraphNodeType } from "@/types/api";

const TYPE_LABELS: Array<{ type: GraphNodeType; label: string }> = [
  { type: "function", label: "Function" },
  { type: "class", label: "Class" },
  { type: "module", label: "Module" },
  { type: "file", label: "File" },
];

interface GraphFilterPanelProps {
  filters: GraphFilterState;
  typeCounts: Record<GraphNodeType, number>;
  onToggleNodeType: (type: GraphNodeType, checked: boolean) => void;
  onSearchChange: (value: string) => void;
  onShowIsolatedChange: (value: boolean) => void;
  onDepthChange: (value: number) => void;
  onMinDegreeChange: (value: number) => void;
  onReset: () => void;
}

export function GraphFilterPanel({
  filters,
  typeCounts,
  onToggleNodeType,
  onSearchChange,
  onShowIsolatedChange,
  onDepthChange,
  onMinDegreeChange,
  onReset,
}: GraphFilterPanelProps) {
  const [localSearch, setLocalSearch] = useState(filters.search);

  useEffect(() => {
    setLocalSearch(filters.search);
  }, [filters.search]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      onSearchChange(localSearch);
    }, 300);

    return () => {
      window.clearTimeout(timer);
    };
  }, [localSearch, onSearchChange]);

  return (
    <div className="space-y-4 rounded-2xl border border-surface-border bg-surface-card p-4">
      <div>
        <p className="text-xs uppercase tracking-[0.1em] text-surface-muted">Node types</p>
        <div className="mt-2 space-y-2">
          {TYPE_LABELS.map((entry) => (
            <label className="flex items-center justify-between gap-2 text-sm text-slate-200" key={entry.type}>
              <span className="flex items-center gap-2">
                <input
                  checked={filters.nodeTypes[entry.type]}
                  onChange={(event) => onToggleNodeType(entry.type, event.target.checked)}
                  type="checkbox"
                />
                {entry.label}
              </span>
              <span className="rounded-full border border-surface-border px-2 py-0.5 text-xs text-surface-muted">
                {typeCounts[entry.type]}
              </span>
            </label>
          ))}
        </div>
      </div>

      <div>
        <p className="text-xs uppercase tracking-[0.1em] text-surface-muted">Node search</p>
        <input
          className="mt-2 w-full rounded-lg border border-surface-border bg-surface px-3 py-2 text-sm text-slate-200 outline-none transition focus:border-brand-500/50"
          onChange={(event) => setLocalSearch(event.target.value)}
          placeholder="Search node name..."
          value={localSearch}
        />
      </div>

      <label className="flex items-center justify-between gap-2 text-sm text-slate-200">
        <span>Show isolated nodes</span>
        <input
          checked={filters.showIsolated}
          onChange={(event) => onShowIsolatedChange(event.target.checked)}
          type="checkbox"
        />
      </label>

      <div>
        <div className="flex items-center justify-between text-sm text-slate-200">
          <span>Depth</span>
          <span className="font-mono text-xs text-surface-muted">{filters.depth}</span>
        </div>
        <input
          className="mt-2 w-full"
          max={6}
          min={1}
          onChange={(event) => onDepthChange(Number(event.target.value))}
          type="range"
          value={filters.depth}
        />
      </div>

      <div>
        <div className="flex items-center justify-between text-sm text-slate-200">
          <span>Min degree</span>
          <span className="font-mono text-xs text-surface-muted">{filters.minDegree}</span>
        </div>
        <input
          className="mt-2 w-full"
          max={10}
          min={0}
          onChange={(event) => onMinDegreeChange(Number(event.target.value))}
          type="range"
          value={filters.minDegree}
        />
      </div>

      <Button className="w-full" onClick={onReset} size="sm" variant="secondary">
        Reset Filters
      </Button>
    </div>
  );
}
