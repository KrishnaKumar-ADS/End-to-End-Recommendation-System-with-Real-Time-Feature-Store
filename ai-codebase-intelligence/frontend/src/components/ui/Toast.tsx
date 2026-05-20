"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
} from "react";

import { cn } from "@/lib/utils";
import type { ToastItem } from "@/types/api";

interface ToastContextValue {
  toasts: ToastItem[];
  toast: (item: Omit<ToastItem, "id">) => void;
  dismiss: (id: string) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

function ToastCard({
  toast,
  onDismiss,
}: {
  toast: ToastItem;
  onDismiss: (id: string) => void;
}) {
  const variantClasses = {
    info: "border-brand-500/25 bg-brand-500/12",
    success: "border-emerald-500/25 bg-emerald-500/12",
    warning: "border-amber-500/25 bg-amber-500/12",
    error: "border-red-500/25 bg-red-500/12",
  };

  return (
    <div
      className={cn(
        "animate-slide-up rounded-2xl border px-4 py-3 text-sm text-slate-100 shadow-card",
        variantClasses[toast.variant ?? "info"],
      )}
      role="status"
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="font-medium">{toast.title}</p>
          {toast.description ? (
            <p className="mt-1 text-xs text-slate-300">{toast.description}</p>
          ) : null}
        </div>
        <button
          className="text-xs uppercase tracking-[0.08em] text-slate-300 transition hover:text-white"
          onClick={() => onDismiss(toast.id)}
          type="button"
        >
          Dismiss
        </button>
      </div>
    </div>
  );
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const timers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  const dismiss = useCallback((id: string) => {
    const timer = timers.current.get(id);
    if (timer) {
      clearTimeout(timer);
      timers.current.delete(id);
    }
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const toast = useCallback(
    (item: Omit<ToastItem, "id">) => {
      const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      setToasts((current) => [...current, { ...item, id }]);

      const timer = setTimeout(() => {
        dismiss(id);
      }, 4000);
      timers.current.set(id, timer);
    },
    [dismiss],
  );

  const value = useMemo(
    () => ({
      toasts,
      toast,
      dismiss,
    }),
    [dismiss, toast, toasts],
  );

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="pointer-events-none fixed right-4 top-4 z-[60] flex w-full max-w-sm flex-col gap-3">
        {toasts.map((item) => (
          <div key={item.id} className="pointer-events-auto">
            <ToastCard onDismiss={dismiss} toast={item} />
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error("useToast must be used inside a <ToastProvider>");
  }
  return context;
}
