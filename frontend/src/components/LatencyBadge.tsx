import { useHealth } from "@/hooks/useHealth";

/**
 * Polls /health every 10 seconds and shows live cache hit ratio + system status.
 * Placed in Navbar for constant visibility.
 */
export default function LatencyBadge() {
  const { data, isLoading, isError } = useHealth();

  if (isLoading) {
    return (
      <div className="badge-muted animate-pulse-slow">
        ⟳ Connecting...
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="badge bg-red-500/20 text-red-400 border border-red-500/30">
        ⚠ API Down
      </div>
    );
  }

  const statusColor =
    data.status === "healthy" ? "badge-success" :
    data.status === "degraded" ? "badge bg-yellow-500/20 text-yellow-400 border border-yellow-500/30" :
    "badge bg-red-500/20 text-red-400 border border-red-500/30";

  const statusIcon =
    data.status === "healthy" ? "●" :
    data.status === "degraded" ? "◐" : "○";

  return (
    <div className="flex items-center gap-2">
      <div className={statusColor}>
        <span className="animate-pulse">{statusIcon}</span>
        {data.status}
      </div>
      <div className="badge-muted">
        Cache: {(data.metrics.cache_hit_ratio * 100).toFixed(0)}%
      </div>
    </div>
  );
}