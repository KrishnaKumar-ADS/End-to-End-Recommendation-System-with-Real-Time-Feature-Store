import { Skeleton } from "@/components/ui/Skeleton";

export default function Loading() {
  return (
    <div className="flex h-[calc(100vh-73px)] flex-col">
      <div className="flex items-center justify-between gap-3 border-b border-surface-border bg-surface-card px-4 py-3">
        <Skeleton className="h-5 w-48" />
        <div className="flex items-center gap-2">
          <Skeleton className="h-9 w-9 rounded-xl" />
          <Skeleton className="h-9 w-28 rounded-xl" />
          <Skeleton className="h-9 w-20 rounded-xl" />
        </div>
      </div>

      <div className="flex-1 space-y-6 overflow-hidden px-4 py-6">
        <div className="flex justify-end">
          <Skeleton className="h-16 w-[70%] rounded-2xl" />
        </div>
        <div className="flex justify-start">
          <Skeleton className="h-40 w-[82%] rounded-2xl" />
        </div>
      </div>

      <div className="border-t border-surface-border bg-surface px-4 py-3">
        <Skeleton className="h-14 w-full rounded-xl" />
      </div>
    </div>
  );
}
