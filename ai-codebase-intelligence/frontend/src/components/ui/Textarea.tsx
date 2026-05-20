import { forwardRef, useId } from "react";
import type { TextareaHTMLAttributes } from "react";

import { cn } from "@/lib/utils";

interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  hint?: string;
  error?: string;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(function Textarea(
  { className, label, hint, error, id, ...props },
  ref,
) {
  const generatedId = useId();
  const textareaId = id ?? generatedId;
  const describedBy = error ? `${textareaId}-error` : hint ? `${textareaId}-hint` : undefined;

  return (
    <label className="flex w-full flex-col gap-2 text-sm text-slate-200" htmlFor={textareaId}>
      {label ? <span className="font-medium text-slate-100">{label}</span> : null}
      <textarea
        ref={ref}
        id={textareaId}
        aria-describedby={describedBy}
        aria-invalid={error ? "true" : "false"}
        className={cn(
          "min-h-[120px] rounded-xl border border-surface-border bg-surface-input px-3 py-3 text-sm text-white outline-none transition placeholder:text-surface-muted focus:border-brand-400 focus:ring-2 focus:ring-brand-500/30 disabled:cursor-not-allowed disabled:opacity-60",
          error ? "border-red-500/70 focus:border-red-400 focus:ring-red-500/30" : "",
          className,
        )}
        {...props}
      />
      {error ? (
        <span id={`${textareaId}-error`} role="alert" className="text-xs text-red-300">
          {error}
        </span>
      ) : hint ? (
        <span id={`${textareaId}-hint`} className="text-xs text-surface-muted">
          {hint}
        </span>
      ) : null}
    </label>
  );
});
