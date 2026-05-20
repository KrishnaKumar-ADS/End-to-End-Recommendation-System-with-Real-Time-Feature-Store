"use client";

import { EDGE_COLORS, EDGE_DASH, NODE_COLORS } from "@/lib/graph-utils";

export function GraphLegend() {
  return (
    <div className="pointer-events-none absolute bottom-3 left-3 z-20 rounded-xl border border-surface-border bg-surface/90 p-3 text-xs text-slate-300 shadow-lg backdrop-blur">
      <p className="mb-2 font-semibold uppercase tracking-[0.1em] text-surface-muted">Legend</p>

      <div className="space-y-1.5">
        <p className="text-[10px] uppercase tracking-[0.08em] text-surface-muted">Node types</p>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: NODE_COLORS.function }} />
            function
          </div>
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: NODE_COLORS.class }} />
            class
          </div>
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: NODE_COLORS.module }} />
            module
          </div>
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: NODE_COLORS.file }} />
            file
          </div>
        </div>
      </div>

      <div className="mt-3 space-y-1.5">
        <p className="text-[10px] uppercase tracking-[0.08em] text-surface-muted">Edge types</p>
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="h-px w-6 bg-slate-100" />
            calls
          </div>
          <div className="flex items-center gap-2">
            <span
              className="h-px w-6"
              style={{
                borderTop: `1px dashed ${EDGE_COLORS.imports}`,
                borderImage: "initial",
              }}
            />
            imports
          </div>
          <div className="flex items-center gap-2">
            <span
              className="h-px w-6"
              style={{
                borderTop: `1px dashed ${EDGE_COLORS.inherits}`,
                strokeDasharray: EDGE_DASH.inherits,
              }}
            />
            inherits
          </div>
        </div>
      </div>

      <p className="mt-3 text-[10px] text-surface-muted">Larger circle = more connections.</p>
    </div>
  );
}
