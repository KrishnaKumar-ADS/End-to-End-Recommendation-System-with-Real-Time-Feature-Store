import { cn } from "@/lib/utils";

export function CodeBlock({
  code,
  language,
  className,
}: {
  code: string;
  language?: string;
  className?: string;
}) {
  return (
    <div className={cn("overflow-hidden rounded-2xl border border-surface-border bg-slate-950/70", className)}>
      <div className="border-b border-surface-border px-4 py-2 text-xs uppercase tracking-[0.12em] text-surface-muted">
        {language || "code"}
      </div>
      <pre className="overflow-x-auto p-4 text-sm text-slate-100">
        <code>{code}</code>
      </pre>
    </div>
  );
}
