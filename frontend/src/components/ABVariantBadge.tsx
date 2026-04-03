interface Props {
  variant: string;
  latencyMs: number;
  cacheHit: boolean;
}

const VARIANT_CONFIG: Record<string, { label: string; description: string; color: string }> = {
  mf: {
    label: "Matrix Factorization",
    description: "Collaborative filtering — classic embedding dot product",
    color: "badge bg-blue-500/20 text-blue-400 border border-blue-500/30",
  },
  sasrec: {
    label: "SASRec Transformer",
    description: "Sequential attention model — captures watch history patterns",
    color: "badge bg-purple-500/20 text-purple-400 border border-purple-500/30",
  },
  item_seed: {
    label: "Movie Similarity (FAISS)",
    description: "Item-seeded retrieval using embedding nearest neighbors",
    color: "badge bg-indigo-500/20 text-indigo-300 border border-indigo-500/30",
  },
  unknown: {
    label: "Unknown Model",
    description: "Variant information unavailable",
    color: "badge-muted",
  },
};

export default function ABVariantBadge({ variant, latencyMs, cacheHit }: Props) {
  const config = VARIANT_CONFIG[variant] ?? VARIANT_CONFIG.unknown;

  return (
    <div className="flex flex-wrap items-center gap-2">
      {/* Model variant badge */}
      <div className={config.color} title={config.description}>
        🧪 {config.label}
      </div>

      {/* Latency badge */}
      <div className={`badge ${latencyMs < 50 ? "badge-success" : latencyMs < 200 ? "badge bg-yellow-500/20 text-yellow-400 border border-yellow-500/30" : "badge bg-red-500/20 text-red-400 border border-red-500/30"}`}>
        ⚡ {latencyMs}ms
      </div>

      {/* Cache indicator */}
      {cacheHit && (
        <div className="badge bg-teal-500/20 text-teal-400 border border-teal-500/30">
          💾 Cache Hit
        </div>
      )}
    </div>
  );
}