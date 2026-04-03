import { useState } from "react";
import type { RecommendResponse } from "@/types/api";
import MovieCard from "./MovieCard";
import ABVariantBadge from "./ABVariantBadge";
import { useFeedback } from "@/hooks/useRecommendations";

interface Props {
  data: RecommendResponse;
}

export default function RecommendationGrid({ data }: Props) {
  const [feedbackSent, setFeedbackSent] = useState<Set<number>>(new Set());
  const feedbackMutation = useFeedback();
  const isMovieSeeded = data.source_type === "movie";
  const feedbackEnabled = !isMovieSeeded && data.user_id > 0;

  const handleFeedback = async (itemId: number, reward: 0 | 1) => {
    if (!feedbackEnabled) return;
    if (feedbackSent.has(itemId)) return;

    await feedbackMutation.mutateAsync({
      user_id: data.user_id,
      item_id: itemId,
      reward,
      variant: data.variant,
    });

    setFeedbackSent((prev) => new Set([...prev, itemId]));
  };

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Results header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h3 className="font-semibold">
            {isMovieSeeded
              ? `Top ${data.recommendations.length} Similar Movies`
              : `Top ${data.recommendations.length} Recommendations`}
          </h3>
          <p className="text-cinema-muted text-sm mt-0.5">
            {isMovieSeeded
              ? `Based on: ${data.seed_title ?? "Selected movie"}`
              : `for User #${data.user_id}`}
          </p>
        </div>
        <ABVariantBadge
          variant={data.variant}
          latencyMs={data.latency_ms}
          cacheHit={Boolean(data.cache_hit ?? data.cached)}
        />
      </div>

      {/* Movie cards grid */}
      <div className="grid grid-cols-1 gap-3">
        {data.recommendations.map((movie, index) => (
          <MovieCard
            key={movie.item_id}
            movie={movie}
            rank={index + 1}
            userId={data.user_id}
            variant={data.variant}
            onFeedback={feedbackEnabled ? handleFeedback : undefined}
            feedbackSent={feedbackEnabled && feedbackSent.has(movie.item_id)}
            enableFeedback={feedbackEnabled}
          />
        ))}
      </div>

      {/* Footer note */}
      <p className="text-cinema-muted text-xs text-center pt-2">
        {isMovieSeeded
          ? "Showing embedding-nearest movies from the Two-Tower + FAISS retrieval layer."
          : "Click any card to see detailed scores. Hover to rate movies. Your feedback trains the bandit engine in real-time."}
      </p>
    </div>
  );
}