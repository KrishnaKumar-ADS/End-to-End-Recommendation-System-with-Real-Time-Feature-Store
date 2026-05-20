import { forwardRef, useId } from "react";
import type { SelectHTMLAttributes } from "react";

import { cn } from "@/lib/utils";

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  hint?: string;
  error?: string;
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(function Select(
  { className, label, hint, error, id, children, ...props },
  ref,
) {
  const generatedId = useId();
  const selectId = id ?? generatedId;
  const describedBy = error ? `${selectId}-error` : hint ? `${selectId}-hint` : undefined;

  return (
    <label className="flex w-full flex-col gap-2 text-sm text-slate-200" htmlFor={selectId}>
      {label ? <span className="font-medium text-slate-100">{label}</span> : null}
      <select
        ref={ref}
        id={selectId}
        aria-describedby={describedBy}
        aria-invalid={error ? "true" : "false"}
        className={cn(
          "h-11 rounded-xl border border-surface-border bg-surface-input px-3 text-sm text-white outline-none transition focus:border-brand-400 focus:ring-2 focus:ring-brand-500/30 disabled:cursor-not-allowed disabled:opacity-60",
          error ? "border-red-500/70 focus:border-red-400 focus:ring-red-500/30" : "",
          className,
        )}
        {...props}
      >
        {children}
      </select>
      {error ? (
        <span id={`${selectId}-error`} role="alert" className="text-xs text-red-300">
          {error}
        </span>
      ) : hint ? (
        <span id={`${selectId}-hint`} className="text-xs text-surface-muted">
          {hint}
        </span>
      ) : null}
    </label>
  );
});
