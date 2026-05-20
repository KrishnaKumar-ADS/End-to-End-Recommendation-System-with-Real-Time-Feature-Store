import { Card } from "@/components/ui/Card";

export function Sidebar({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <Card className="sticky top-24" padding="lg">
      <div className="space-y-4">
        <div className="space-y-1">
          <h2 className="text-base font-semibold text-white">{title}</h2>
          {description ? <p className="text-sm text-surface-muted">{description}</p> : null}
        </div>
        {children}
      </div>
    </Card>
  );
}
