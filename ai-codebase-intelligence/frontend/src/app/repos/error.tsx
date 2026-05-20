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
      <p className="text-xs uppercase tracking-[0.14em] text-red-200">Repositories route error</p>
      <h1 className="text-2xl font-semibold text-white">The repository list could not be loaded.</h1>
      <p className="text-sm text-surface-muted">{error.message}</p>
      <Button onClick={reset}>Retry</Button>
    </Card>
  );
}
