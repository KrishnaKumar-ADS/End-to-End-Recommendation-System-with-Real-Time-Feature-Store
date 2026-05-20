"use client";

import { CodeBlock } from "@/components/Chat/CodeBlock";
import { parseAnswer } from "@/lib/markdown";
import { cn } from "@/lib/utils";

interface AnswerRendererProps {
  content: string;
  isStreaming?: boolean;
  className?: string;
}

export function AnswerRenderer({
  content,
  isStreaming = false,
  className,
}: AnswerRendererProps) {
  if (!content && isStreaming) {
    return (
      <div className={cn("text-sm leading-relaxed text-slate-200", className)}>
        <span className="ml-0.5 inline-block h-4 w-0.5 animate-pulse bg-brand-400 align-middle" />
      </div>
    );
  }

  if (!content) {
    return null;
  }

  const segments = parseAnswer(content);

  return (
    <div className={cn("space-y-1 text-sm leading-relaxed text-slate-200", className)}>
      {segments.map((segment, segmentIndex) => {
        if (segment.kind === "code") {
          return (
            <CodeBlock
              key={`${segment.kind}-${segmentIndex}`}
              code={segment.content}
              language={segment.language}
            />
          );
        }

        const lines = segment.content.split("\n");
        const isLastSegment = segmentIndex === segments.length - 1;

        return (
          <div key={`${segment.kind}-${segmentIndex}`}>
            {lines.map((line, lineIndex) => {
              const isLastLine = isLastSegment && lineIndex === lines.length - 1;

              return (
                <span key={`${line}-${lineIndex}`}>
                  {line}
                  {isLastLine && isStreaming ? (
                    <span
                      aria-hidden="true"
                      className="ml-0.5 inline-block h-[1em] w-0.5 animate-pulse bg-brand-400 align-text-bottom"
                    />
                  ) : null}
                  {lineIndex < lines.length - 1 ? <br /> : null}
                </span>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}
