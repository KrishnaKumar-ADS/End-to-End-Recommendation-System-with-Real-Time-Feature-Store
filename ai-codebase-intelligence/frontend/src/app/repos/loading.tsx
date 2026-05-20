import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";

export default function Loading() {
  return (
    <div className="space-y-6">
      <Card padding="lg">
        <Skeleton className="h-8 w-56" />
        <Skeleton className="mt-4 h-16 w-full" />
      </Card>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        {Array.from({ length: 6 }).map((_, index) => (
          <Card key={index} padding="lg">
            <Skeleton className="h-6 w-32" />
            <Skeleton className="mt-4 h-28 w-full" />
          </Card>
        ))}
      </div>
    </div>
  );
}
