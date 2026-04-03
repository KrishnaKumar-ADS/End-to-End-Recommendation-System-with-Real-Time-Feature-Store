import apiClient from "./client";
import type {
  RecommendResponse,
  FeedbackRequest,
  FeedbackResponse,
  ABReport,
  MovieSuggestionResponse,
} from "@/types/api";

/**
 * Fetch recommendations for a given user.
 *
 * @param userId  - The integer user ID (from MovieLens dataset)
 * @param topK    - Number of results to request (default: 10)
 */
export async function fetchRecommendations(
  userId: number,
  topK: number = 10
): Promise<RecommendResponse> {
  const response = await apiClient.get<RecommendResponse>("/recommend", {
    params: { user_id: userId, top_k: topK },
  });
  return response.data;
}

/**
 * Fetch recommendations seeded by a movie title.
 */
export async function fetchMovieRecommendations(
  movieTitle: string,
  topK: number = 10
): Promise<RecommendResponse> {
  const response = await apiClient.get<RecommendResponse>("/recommend/by-movie", {
    params: { movie_title: movieTitle, top_k: topK },
  });
  return response.data;
}

/**
 * Fetch movie title suggestions for keyword search autocomplete.
 */
export async function fetchMovieSuggestions(
  query: string,
  limit: number = 8
): Promise<MovieSuggestionResponse> {
  const response = await apiClient.get<MovieSuggestionResponse>("/recommend/movie-suggestions", {
    params: { query, limit },
  });
  return response.data;
}

/**
 * Submit user feedback (click / no-click).
 *
 * Call this when:
 *   reward=1 → user clicked / liked the movie
 *   reward=0 → user explicitly disliked (thumbs down)
 *
 * The backend updates both the Thompson Sampling bandit state
 * AND the A/B testing conversion logger with this signal.
 */
export async function submitFeedback(
  data: FeedbackRequest
): Promise<FeedbackResponse> {
  const response = await apiClient.post<FeedbackResponse>("/feedback", data);
  return response.data;
}

/**
 * Fetch the current A/B test report.
 * Returns statistical significance + per-variant CTR.
 */
export async function fetchABReport(): Promise<ABReport> {
  const response = await apiClient.get<ABReport>("/ab/report");
  return response.data;
}