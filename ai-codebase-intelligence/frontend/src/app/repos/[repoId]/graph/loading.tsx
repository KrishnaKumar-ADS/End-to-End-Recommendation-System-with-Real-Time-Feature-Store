import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";

export default function Loading() {
  return (
    <div className="space-y-3">
      <Card padding="sm">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="mt-3 h-10 w-full" />
      </Card>
      <div className="grid gap-3 lg:grid-cols-[280px,1fr,320px]">
        <Card className="hidden lg:block" padding="lg">
          <Skeleton className="h-72 w-full" />
        </Card>
        <Card padding="lg">
          <Skeleton className="h-[68vh] w-full" />
        </Card>
        <Card className="hidden lg:block" padding="lg">
          <Skeleton className="h-72 w-full" />
        </Card>
      </div>
    </div>
  );
}
