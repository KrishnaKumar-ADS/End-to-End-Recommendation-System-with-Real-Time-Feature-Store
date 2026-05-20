import { forwardRef, useId } from "react";
import type { InputHTMLAttributes } from "react";

import { cn } from "@/lib/utils";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  hint?: string;
  error?: string;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input(
  { className, label, hint, error, id, ...props },
  ref,
) {
  const generatedId = useId();
  const inputId = id ?? generatedId;
  const describedBy = error ? `${inputId}-error` : hint ? `${inputId}-hint` : undefined;

  return (
    <label className="flex w-full flex-col gap-2 text-sm text-slate-200" htmlFor={inputId}>
      {label ? <span className="font-medium text-slate-100">{label}</span> : null}
      <input
        ref={ref}
        id={inputId}
        aria-describedby={describedBy}
        aria-invalid={error ? "true" : "false"}
        className={cn(
          "h-11 rounded-xl border border-surface-border bg-surface-input px-3 text-sm text-white outline-none transition placeholder:text-surface-muted focus:border-brand-400 focus:ring-2 focus:ring-brand-500/30 disabled:cursor-not-allowed disabled:opacity-60",
          error ? "border-red-500/70 focus:border-red-400 focus:ring-red-500/30" : "",
          className,
        )}
        {...props}
      />
      {error ? (
        <span id={`${inputId}-error`} role="alert" className="text-xs text-red-300">
          {error}
        </span>
      ) : hint ? (
        <span id={`${inputId}-hint`} className="text-xs text-surface-muted">
          {hint}
        </span>
      ) : null}
    </label>
  );
});
