"use client";

import { useState } from "react";

import { cn } from "@/lib/utils";

interface TooltipProps {
  content: string;
  children: React.ReactNode;
  className?: string;
}

export function Tooltip({ content, children, className }: TooltipProps) {
  const [open, setOpen] = useState(false);

  return (
    <span
      className={cn("relative inline-flex", className)}
      onBlur={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      {children}
      {open ? (
        <span className="absolute left-1/2 top-full z-40 mt-2 -translate-x-1/2 rounded-lg border border-surface-border bg-surface-card px-2 py-1 text-xs text-slate-100 shadow-card">
          {content}
        </span>
      ) : null}
    </span>
  );
}
