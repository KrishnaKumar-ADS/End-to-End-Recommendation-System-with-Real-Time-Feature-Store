import { useABReport } from "@/hooks/useABReport";
import { useHealth } from "@/hooks/useHealth";

const VARIANT_LABELS: Record<string, string> = {
  mf: "Matrix Factorization",
  sasrec: "SASRec Transformer",
  lgbm_only: "Two-Tower + LightGBM",
  lgbm_bandit: "Two-Tower + LightGBM + Bandit",
};

const STATUS_BADGE_CLASS: Record<string, string> = {
  healthy: "badge-success",
  ok: "badge-success",
  disabled: "badge-muted",
  degraded: "badge bg-yellow-500/20 text-yellow-400 border border-yellow-500/30",
  down: "badge bg-red-500/20 text-red-400 border border-red-500/30",
  unavailable: "badge-muted",
};

const ARCHITECTURE_OVERVIEW = [
  {
    title: "Retrieval Layer",
    stack: "Two-Tower User Encoder + FAISS ANN",
    detail: "User interaction sequence is encoded, then nearest candidates are retrieved from FAISS.",
  },
  {
    title: "Ranking Layer",
    stack: "LightGBM LambdaRank",
    detail: "Candidates are reranked with feature-rich scoring to produce the final top-k list.",
  },
  {
    title: "Online Learning",
    stack: "Thompson Sampling Bandit",
    detail: "Feedback updates online preference signals and can rerank outputs in real time.",
  },
  {
    title: "Experimentation",
    stack: "A/B Router + Statistical Analyzer",
    detail: "Traffic split, exposure/conversion logging, and significance testing run continuously.",
  },
  {
    title: "Monitoring",
    stack: "Health API + Metrics + Dashboard",
    detail: "System status, cache ratio, latency and experiment outcomes are surfaced for demos.",
  },
  {
    title: "User Experience",
    stack: "React + React Query",
    detail: "Live recommendation UI, feedback capture, and A/B insights are visible in one app.",
  },
];

const PROJECT_HIGHLIGHTS = [
  "Robust ranking inference handling for non-numeric feature columns.",
  "Accurate p-value display for very small significance values.",
  "Architecture-aligned A/B variant labels for clearer storytelling.",
  "Live health status integration for shared pipeline visibility.",
];

function getVariantLabel(variant: string | undefined): string {
  if (!variant) return "Unknown Variant";
  return VARIANT_LABELS[variant] ?? variant;
}

function getStatusMeta(status: string | undefined): { label: string; className: string } {
  const normalized = (status ?? "unavailable").toLowerCase();
  const className = STATUS_BADGE_CLASS[normalized] ?? STATUS_BADGE_CLASS.unavailable;
  const label = normalized === "ok"
    ? "OK"
    : normalized.charAt(0).toUpperCase() + normalized.slice(1);
  return { label, className };
}

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="card text-center">
      <div className="text-2xl font-bold text-cinema-accent">{value}</div>
      <div className="text-cinema-text font-medium text-sm mt-1">{label}</div>
      {sub && <div className="text-cinema-muted text-xs mt-0.5">{sub}</div>}
    </div>
  );
}

function CTRBar({ ctr, maxCtr, label, color }: {
  ctr: number; maxCtr: number; label: string; color: string;
}) {
  const pct = maxCtr > 0 ? (ctr / maxCtr) * 100 : 0;
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="font-medium">{label}</span>
        <span className={color}>{(ctr * 100).toFixed(2)}% CTR</span>
      </div>
      <div className="h-3 bg-cinema-border rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-1000 ${color.includes("blue") ? "bg-blue-500" : "bg-purple-500"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function formatPValue(pValue: number): string {
  if (!Number.isFinite(pValue)) return "N/A";
  if (pValue === 0) return "< 1e-16";
  if (pValue < 1e-4) return pValue.toExponential(2);
  return pValue.toFixed(4);
}

export default function ABDashboard() {
  const { data, isLoading, isError, error } = useABReport();
  const { data: health, isLoading: isHealthLoading, isError: isHealthError } = useHealth();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64 text-cinema-muted">
        <div className="text-center">
          <div className="animate-spin text-4xl mb-3">⟳</div>
          <p>Loading A/B results...</p>
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="card border-cinema-accent/50 text-center py-12">
        <div className="text-4xl mb-3">⚠</div>
        <p className="text-cinema-accent font-medium">Failed to load A/B report</p>
        <p className="text-cinema-muted text-sm mt-1">{error?.message}</p>
      </div>
    );
  }

  if (!data) return null;

  const maxCtr = Math.max(...data.variants.map((v) => v.ctr));
  const sharedComponents = health
    ? [
        { title: "Core Models", ...health.components.models },
        { title: "Redis Cache", ...health.components.redis },
        { title: "Feature Layer", ...health.components.feast },
      ]
    : [];

  return (
    <div className="space-y-8 animate-fade-in">

      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">A/B Test Dashboard</h1>
        <p className="text-cinema-muted text-sm mt-1">
          Experiment: <span className="text-cinema-text">{data.experiment_name}</span>
          {" · "}Started: {new Date(data.start_time).toLocaleDateString()}
        </p>
      </div>

      {/* Significance banner */}
      <div className={`card border-2 text-center py-4
        ${data.is_significant
          ? "border-green-500/50 bg-green-500/5"
          : "border-cinema-border"
        }`}>
        {data.is_significant ? (
          <div>
            <div className="text-green-400 font-bold text-lg">
              🏆 Statistically Significant Result
            </div>
            <div className="text-cinema-muted text-sm mt-1">
              Winner: <span className="text-green-400 font-medium">{getVariantLabel(data.winner)}</span>
              {" · "}p-value: {formatPValue(data.p_value)}
              {" · "}z-score: {data.z_score.toFixed(2)}
            </div>
          </div>
        ) : (
          <div>
            <div className="text-cinema-muted font-medium">
              🔬 Not yet statistically significant
            </div>
            <div className="text-cinema-muted text-sm mt-1">
              p-value: {formatPValue(data.p_value)} · Need more data to conclude
            </div>
          </div>
        )}
      </div>

      {/* Per-variant stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {data.variants.map((variant, i) => {
          const colors = ["text-blue-400", "text-purple-400"];
          return (
            <div key={variant.variant} className="card space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-semibold">{getVariantLabel(variant.variant)}</h3>
                  <p className="text-cinema-muted text-xs mt-0.5">ID: {variant.variant}</p>
                </div>
                {data.winner === variant.variant && (
                  <span className="badge-success">👑 Winner</span>
                )}
              </div>

              <div className="grid grid-cols-3 gap-3">
                <StatCard
                  label="Exposures"
                  value={variant.n_exposures.toLocaleString()}
                />
                <StatCard
                  label="Clicks"
                  value={variant.n_conversions.toLocaleString()}
                />
                <StatCard
                  label="CTR"
                  value={`${(variant.ctr * 100).toFixed(2)}%`}
                  sub={`CI: [${(variant.confidence_interval[0]*100).toFixed(1)}%, ${(variant.confidence_interval[1]*100).toFixed(1)}%]`}
                />
              </div>

              <CTRBar
                ctr={variant.ctr}
                maxCtr={maxCtr}
                label="Click-through Rate"
                color={colors[i % colors.length]}
              />
            </div>
          );
        })}
      </div>

      {/* Shared pipeline status */}
      <div className="card bg-cinema-accent2/10 border-cinema-accent2/30 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <h4 className="font-semibold text-sm">🏗 Shared Pipeline (Live from /health)</h4>
            <p className="text-cinema-muted text-xs mt-1">
              This serving stack is shared across current A/B variants.
            </p>
          </div>

          {isHealthLoading ? (
            <span className="badge-muted animate-pulse-slow">⟳ Checking health...</span>
          ) : isHealthError || !health ? (
            <span className="badge bg-red-500/20 text-red-400 border border-red-500/30">⚠ Health unavailable</span>
          ) : (
            <span className={getStatusMeta(health.status).className}>
              System {getStatusMeta(health.status).label}
            </span>
          )}
        </div>

        {health && !isHealthError && (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <StatCard label="Overall" value={getStatusMeta(health.status).label} />
              <StatCard label="Uptime" value={`${health.metrics.uptime_hours.toFixed(2)}h`} />
              <StatCard label="Cache Hit Ratio" value={`${(health.metrics.cache_hit_ratio * 100).toFixed(1)}%`} />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {sharedComponents.map((component) => {
                const statusMeta = getStatusMeta(component.status);
                return (
                  <div key={component.title} className="rounded-xl border border-cinema-border bg-cinema-bg/40 p-3 space-y-1">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-sm font-medium">{component.title}</p>
                      <span className={statusMeta.className}>{statusMeta.label}</span>
                    </div>
                    <p className="text-cinema-muted text-xs">{component.detail ?? "No detail reported"}</p>
                    {typeof component.latency_ms === "number" && (
                      <p className="text-cinema-muted text-xs">Latency: {component.latency_ms.toFixed(2)}ms</p>
                    )}
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>

      {/* Architecture snapshot */}
      <div className="card space-y-4">
        <div>
          <h4 className="font-semibold text-sm">🧭 End-to-End Architecture Snapshot</h4>
          <p className="text-cinema-muted text-xs mt-1">
            Use this section to explain the full project flow during demos.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {ARCHITECTURE_OVERVIEW.map((block) => (
            <div key={block.title} className="rounded-xl border border-cinema-border bg-cinema-bg/40 p-3">
              <p className="text-sm font-semibold">{block.title}</p>
              <p className="text-cinema-text text-xs mt-1">{block.stack}</p>
              <p className="text-cinema-muted text-xs mt-1.5">{block.detail}</p>
            </div>
          ))}
        </div>

        <div className="rounded-xl border border-cinema-border bg-cinema-bg/40 p-3">
          <p className="text-sm font-semibold">✅ Project Highlights Implemented</p>
          <ul className="text-cinema-muted text-xs mt-2 space-y-1.5">
            {PROJECT_HIGHLIGHTS.map((item) => (
              <li key={item}>• {item}</li>
            ))}
          </ul>
        </div>
      </div>

      {/* How to interpret */}
      <div className="card bg-cinema-accent2/10 border-cinema-accent2/30">
        <h4 className="font-semibold mb-2 text-sm">📚 How to Interpret These Results</h4>
        <ul className="text-cinema-muted text-xs space-y-1.5">
          <li>• <strong className="text-cinema-text">CTR (Click-Through Rate)</strong>: fraction of shown recommendations that users clicked. Higher = better recommendations.</li>
          <li>• <strong className="text-cinema-text">Confidence Interval</strong>: 95% CI on the true CTR. Narrower = more data collected.</li>
          <li>• <strong className="text-cinema-text">p-value</strong>: probability the CTR difference is due to chance. p &lt; 0.05 = significant.</li>
          <li>• <strong className="text-cinema-text">z-score</strong>: number of standard deviations the difference is from zero. |z| &gt; 1.96 = significant at 95%.</li>
          <li>• <strong className="text-cinema-text">Not significant yet?</strong>: Run more traffic. Need ~1,500 exposures per variant for a 5% CTR difference.</li>
        </ul>
      </div>
    </div>
  );
}