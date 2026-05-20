"use client";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <Card className="space-y-4" padding="lg">
      <p className="text-xs uppercase tracking-[0.14em] text-red-200">Something went wrong</p>
      <h1 className="text-2xl font-semibold text-white">The frontend hit a recoverable error.</h1>
      <p className="text-sm text-surface-muted">{error.message}</p>
      <div className="flex gap-3">
        <Button onClick={reset}>Try again</Button>
      </div>
    </Card>
  );
}
