import { cn } from "@/lib/utils";

interface ProgressBarProps {
  value: number;
  label?: string;
  indeterminate?: boolean;
  className?: string;
}

export function ProgressBar({
  value,
  label,
  indeterminate = false,
  className,
}: ProgressBarProps) {
  const safeValue = Math.max(0, Math.min(100, Math.round(value)));

  return (
    <div className={cn("space-y-2", className)}>
      {label ? (
        <div className="flex items-center justify-between text-xs uppercase tracking-[0.08em] text-surface-muted">
          <span>{label}</span>
          <span>{safeValue}%</span>
        </div>
      ) : null}
      <div className="h-3 overflow-hidden rounded-full border border-surface-border bg-white/5">
        <div
          aria-label={label ?? "Progress"}
          aria-valuemax={100}
          aria-valuemin={0}
          aria-valuenow={safeValue}
          className={cn(
            "h-full rounded-full bg-gradient-to-r from-brand-500 via-cyan-400 to-brand-600 transition-[width] duration-500 ease-out",
            indeterminate ? "animate-progress-bar bg-[length:200%_100%]" : "",
          )}
          role="progressbar"
          style={{ width: `${safeValue}%` }}
        />
      </div>
    </div>
  );
}
