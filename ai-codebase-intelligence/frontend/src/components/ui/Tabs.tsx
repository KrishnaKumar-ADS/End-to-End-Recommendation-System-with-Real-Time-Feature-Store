"use client";

import { useMemo, useState } from "react";

import { cn } from "@/lib/utils";

export interface TabItem {
  id: string;
  label: string;
  content: React.ReactNode;
}

interface TabsProps {
  items: TabItem[];
  defaultValue?: string;
}

export function Tabs({ items, defaultValue }: TabsProps) {
  const fallback = useMemo(() => defaultValue ?? items[0]?.id ?? "", [defaultValue, items]);
  const [activeTab, setActiveTab] = useState(fallback);

  if (!items.length) {
    return null;
  }

  const activeItem = items.find((item) => item.id === activeTab) ?? items[0];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2" role="tablist">
        {items.map((item) => {
          const selected = item.id === activeItem.id;
          return (
            <button
              key={item.id}
              aria-selected={selected}
              className={cn(
                "rounded-full border px-3 py-1.5 text-sm transition",
                selected
                  ? "border-brand-500/40 bg-brand-500/15 text-brand-100"
                  : "border-surface-border bg-white/5 text-slate-300 hover:bg-surface-hover",
              )}
              onClick={() => setActiveTab(item.id)}
              role="tab"
              type="button"
            >
              {item.label}
            </button>
          );
        })}
      </div>
      <div role="tabpanel">{activeItem.content}</div>
    </div>
  );
}
