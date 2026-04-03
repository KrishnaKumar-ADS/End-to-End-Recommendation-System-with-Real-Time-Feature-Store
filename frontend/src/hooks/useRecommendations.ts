import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchMovieRecommendations,
  fetchMovieSuggestions,
  fetchRecommendations,
  submitFeedback,
} from "@/api/recommendations";
import type {
  RecommendResponse,
  FeedbackRequest,
  MovieSuggestionResponse,
} from "@/types/api";

/**
 * Fetches recommendations for a given userId.
 * Returns null query if userId is not yet set.
 *
 * Caching behavior:
 *   - staleTime: 30s (re-use cached recs if user hasn't changed)
 *   - gcTime: 5min (keep in React Query cache after component unmounts)
 */
export function useRecommendations(userId: number | null) {
  return useQuery<RecommendResponse, Error>({
    queryKey: ["recommendations", userId],
    queryFn: () => fetchRecommendations(userId!),
    enabled: userId !== null && userId > 0,
    staleTime: 30_000,
  });
}

/**
 * Fetches recommendations seeded by a movie title.
 */
export function useMovieRecommendations(movieTitle: string | null) {
  const normalizedTitle = movieTitle?.trim() ?? "";

  return useQuery<RecommendResponse, Error>({
    queryKey: ["movie-recommendations", normalizedTitle],
    queryFn: () => fetchMovieRecommendations(normalizedTitle),
    enabled: normalizedTitle.length > 0,
    staleTime: 30_000,
  });
}

/**
 * Fetches keyword-based title suggestions for movie search.
 */
export function useMovieTitleSuggestions(query: string | null) {
  const normalizedQuery = query?.trim() ?? "";

  return useQuery<MovieSuggestionResponse, Error>({
    queryKey: ["movie-suggestions", normalizedQuery],
    queryFn: () => fetchMovieSuggestions(normalizedQuery, 8),
    enabled: normalizedQuery.length >= 2,
    staleTime: 60_000,
  });
}

/**
 * Submits feedback. Optimistically updates UI.
 * On success: invalidates recommendation cache (forces fresh fetch).
 */
export function useFeedback() {
  const queryClient = useQueryClient();

  return useMutation<void, Error, FeedbackRequest>({
    mutationFn: async (data) => {
      await submitFeedback(data);
    },
    onSuccess: (_, variables) => {
      // Invalidate this user's recommendations so next fetch is fresh
      queryClient.invalidateQueries({
        queryKey: ["recommendations", variables.user_id],
      });
    },
  });
}