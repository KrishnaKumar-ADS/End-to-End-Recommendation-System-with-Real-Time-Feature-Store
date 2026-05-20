"use client";

import { type KeyboardEvent, useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/utils";

interface ChatInputProps {
  onSubmit: (question: string) => void;
  isStreaming: boolean;
  onCancel: () => void;
  disabled?: boolean;
  placeholder?: string;
  value?: string;
  onValueChange?: (value: string) => void;
}

export function ChatInput({
  onSubmit,
  isStreaming,
  onCancel,
  disabled = false,
  placeholder = "Ask anything about this repository...",
  value,
  onValueChange,
}: ChatInputProps) {
  const [internalValue, setInternalValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isControlled = typeof value === "string";
  const inputValue = isControlled ? value : internalValue;

  const setValue = useCallback(
    (nextValue: string) => {
      if (!isControlled) {
        setInternalValue(nextValue);
      }
      onValueChange?.(nextValue);
    },
    [isControlled, onValueChange],
  );

  const autoResize = useCallback(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }

    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 192)}px`;
  }, []);

  useEffect(() => {
    autoResize();
  }, [autoResize, inputValue]);

  const handleSubmit = useCallback(() => {
    const trimmed = inputValue.trim();
    if (!trimmed || disabled || isStreaming) {
      return;
    }

    onSubmit(trimmed);
    setValue("");

    requestAnimationFrame(() => {
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto";
      }
    });
  }, [disabled, inputValue, isStreaming, onSubmit, setValue]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key !== "Enter") {
        return;
      }

      if (event.shiftKey) {
        return;
      }

      if (event.ctrlKey || event.metaKey || !event.shiftKey) {
        event.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  const canSubmit = !disabled && Boolean(inputValue.trim());

  return (
    <div className="shrink-0 border-t border-surface-border bg-surface px-4 py-3">
      <div
        className={cn(
          "flex items-end gap-2 rounded-xl border px-3 py-2",
          "bg-surface-input transition-colors duration-150",
          disabled
            ? "cursor-not-allowed border-surface-border opacity-60"
            : "border-surface-border focus-within:border-brand-500/50",
        )}
      >
        <textarea
          aria-label="Chat input"
          aria-multiline="true"
          className={cn(
            "min-h-[1.5rem] max-h-48 flex-1 resize-none bg-transparent py-0.5",
            "text-sm leading-relaxed text-slate-200 placeholder:text-surface-muted",
            "focus:outline-none",
          )}
          disabled={disabled}
          onChange={(event) => {
            setValue(event.target.value);
            autoResize();
          }}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          ref={textareaRef}
          rows={1}
          value={inputValue}
        />

        {isStreaming ? (
          <Button
            aria-label="Stop generating"
            className="shrink-0 self-end"
            onClick={onCancel}
            size="sm"
            variant="danger"
          >
            Stop
          </Button>
        ) : (
          <Button
            aria-label="Send question"
            className="shrink-0 self-end"
            disabled={!canSubmit}
            onClick={handleSubmit}
            size="sm"
            variant="primary"
          >
            Send
          </Button>
        )}
      </div>

      <p className="mt-1.5 text-center text-[10px] text-surface-muted">
        Enter to send · Shift+Enter for new line
      </p>
    </div>
  );
}
