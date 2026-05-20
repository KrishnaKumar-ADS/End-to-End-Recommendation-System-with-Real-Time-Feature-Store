import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";

export default function Loading() {
  return (
    <div className="grid gap-6 lg:grid-cols-[320px,1fr]">
      <Card padding="lg">
        <Skeleton className="h-6 w-32" />
        <Skeleton className="mt-4 h-96 w-full" />
      </Card>
      <div className="space-y-6">
        <Card padding="lg">
          <Skeleton className="h-8 w-56" />
          <Skeleton className="mt-4 h-16 w-full" />
        </Card>
        <Card padding="lg">
          <Skeleton className="h-8 w-72" />
          <Skeleton className="mt-4 h-40 w-full" />
        </Card>
      </div>
    </div>
  );
}
