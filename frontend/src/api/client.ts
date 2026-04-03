import axios from "axios";

function normalizeApiErrorMessage(data: unknown): string | null {
  if (!data) {
    return null;
  }

  if (typeof data === "string") {
    const trimmed = data.trim();
    return trimmed.length > 0 ? trimmed : null;
  }

  if (typeof data !== "object") {
    return null;
  }

  const payload = data as {
    detail?: unknown;
    message?: unknown;
    error?: unknown;
    suggestions?: unknown;
  };

  if (typeof payload.message === "string" && payload.message.trim().length > 0) {
    return payload.message;
  }

  let detailPayload = payload.detail;
  if (!detailPayload && (payload.error || payload.suggestions)) {
    detailPayload = payload;
  }

  if (typeof detailPayload === "string" && detailPayload.trim().length > 0) {
    return detailPayload;
  }

  if (detailPayload && typeof detailPayload === "object") {
    const detail = detailPayload as {
      detail?: unknown;
      error?: unknown;
      suggestions?: unknown;
    };

    const detailText = typeof detail.detail === "string" ? detail.detail.trim() : "";
    const suggestions = Array.isArray(detail.suggestions)
      ? detail.suggestions.filter((value): value is string => typeof value === "string")
      : [];

    if (suggestions.length > 0) {
      const suggestionText = suggestions.slice(0, 4).join(", ");
      const prefix = detailText || "No exact title found.";
      return `${prefix} Try: ${suggestionText}`;
    }

    if (detailText.length > 0) {
      return detailText;
    }

    if (typeof detail.error === "string" && detail.error.trim().length > 0) {
      return detail.error;
    }
  }

  return null;
}

/**
 * Configured Axios instance for DS19 API.
 *
 * baseURL: empty string → uses Vite proxy in dev, Nginx in prod.
 * withCredentials: false → no cookies (stateless API).
 * timeout: 10s → UI should never hang longer than this.
 */
const apiClient = axios.create({
  baseURL: "",             // Vite proxy handles routing in dev
  timeout: 10_000,
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
});

// ── REQUEST INTERCEPTOR ──────────────────────────────────────────────
// Add request timestamp for client-side latency tracking.
apiClient.interceptors.request.use((config) => {
  config.metadata = { startTime: performance.now() };
  return config;
});

// ── RESPONSE INTERCEPTOR ──────────────────────────────────────────────
// Attach client-side latency to every response.
apiClient.interceptors.response.use(
  (response) => {
    const elapsed = performance.now() - response.config.metadata?.startTime;
    response.data._client_latency_ms = Math.round(elapsed);
    return response;
  },
  (error) => {
    // Normalize error message
    const message =
      normalizeApiErrorMessage(error.response?.data) ||
      error.message ||
      "Unknown API error";

    console.error("[DS19 API Error]", {
      status: error.response?.status,
      url: error.config?.url,
      message,
    });

    return Promise.reject(new Error(message));
  }
);

// TypeScript: extend AxiosRequestConfig to accept metadata
declare module "axios" {
  interface InternalAxiosRequestConfig {
    metadata?: { startTime: number };
  }
}

export default apiClient;