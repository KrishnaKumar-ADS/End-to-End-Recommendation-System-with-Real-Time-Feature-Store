export interface MovieRecommendation {
  item_id: number;
  movie_id?: number;        // original MovieLens movie_id
  title: string;
  genres: string[];
  score: number;            // LightGBM ranking score [0, 1]
  bandit_score?: number;    // Thompson Sampling posterior mean
  similarity_score?: number; // Two-Tower cosine similarity
  year?: number;            // extracted from title "(YYYY)"
  poster_url?: string;      // optional; backend may return null
}

export interface RecommendResponse {
  user_id: number;
  top_k?: number;
  recommendations: MovieRecommendation[];
  variant: string;          // "mf" | "sasrec"
  source_type?: "user" | "movie" | string;
  seed_title?: string;
  seed_item_id?: number;
  latency_ms: number;
  cache_hit?: boolean;
  cached?: boolean;
  request_id?: string;
}

export interface MovieSuggestionResponse {
  query: string;
  suggestions: string[];
}

export interface FeedbackRequest {
  user_id: number;
  item_id: number;
  reward: 0 | 1;            // 0 = no click, 1 = click
  variant: string;
}

export interface FeedbackResponse {
  status: "ok" | "accepted";
  message: string;
}

export interface ComponentStatus {
  status: "ok" | "disabled" | "degraded" | "down";
  latency_ms?: number;
  detail?: string;
}

export interface HealthResponse {
  status: "healthy" | "degraded" | "unhealthy";
  uptime_seconds: number;
  version: string;
  components: {
    redis: ComponentStatus;
    feast: ComponentStatus;
    models: ComponentStatus;
  };
  metrics: {
    cache_hit_ratio: number;
    uptime_hours: number;
  };
}

export interface ABExperimentStats {
  variant: string;
  n_exposures: number;
  n_conversions: number;
  ctr: number;              // click-through rate [0, 1]
  confidence_interval: [number, number];
}

export interface ABReport {
  experiment_name: string;
  start_time: string;
  variants: ABExperimentStats[];
  winner?: string;
  is_significant: boolean;
  p_value: number;
  z_score: number;
}