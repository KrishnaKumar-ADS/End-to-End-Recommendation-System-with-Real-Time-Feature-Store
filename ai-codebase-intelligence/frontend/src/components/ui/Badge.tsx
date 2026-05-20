import type { HTMLAttributes } from "react";

import { getProviderLabel, getProviderTone, getStatusLabel, getStatusTone, cn } from "@/lib/utils";
import type { IngestionStatus } from "@/types/api";

type BadgeTone =
  | "neutral"
  | "info"
  | "warning"
  | "success"
  | "danger"
  | "qwen"
  | "gemini"
  | "deepseek";

const toneClasses: Record<BadgeTone, string> = {
  neutral: "border-surface-border bg-white/5 text-slate-200",
  info: "border-brand-500/30 bg-brand-500/10 text-brand-200",
  warning: "border-amber-500/30 bg-amber-500/10 text-amber-200",
  success: "border-emerald-500/30 bg-emerald-500/10 text-emerald-200",
  danger: "border-red-500/30 bg-red-500/10 text-red-200",
  qwen: "border-provider-qwen/30 bg-provider-qwen/15 text-violet-200",
  gemini: "border-provider-gemini/30 bg-provider-gemini/15 text-sky-200",
  deepseek: "border-provider-deepseek/30 bg-provider-deepseek/15 text-cyan-200",
};

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: BadgeTone;
}

export function Badge({ className, tone = "neutral", ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-medium uppercase tracking-[0.08em]",
        toneClasses[tone],
        className,
      )}
      {...props}
    />
  );
}

export function ProviderBadge({
  provider,
  model,
  className,
}: {
  provider?: string;
  model?: string;
  className?: string;
}) {
  return (
    <Badge className={className} tone={getProviderTone(provider, model)}>
      {getProviderLabel(provider, model)}
    </Badge>
  );
}

export function StatusBadge({
  status,
  className,
}: {
  status: IngestionStatus | string;
  className?: string;
}) {
  return (
    <Badge className={className} tone={getStatusTone(status)}>
      {getStatusLabel(status)}
    </Badge>
  );
}
