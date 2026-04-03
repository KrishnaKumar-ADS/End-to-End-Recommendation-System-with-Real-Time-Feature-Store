import { useState } from "react";
import type { MovieRecommendation } from "@/types/api";
import FeedbackButton from "./FeedbackButton";

interface Props {
  movie: MovieRecommendation;
  rank: number;
  userId?: number;
  variant?: string;
  onFeedback?: (itemId: number, reward: 0 | 1) => void;
  feedbackSent?: boolean;
  enableFeedback?: boolean;
}

// Color map for genre badges
const GENRE_COLORS: Record<string, string> = {
  Action: "bg-red-500/20 text-red-400",
  Comedy: "bg-yellow-500/20 text-yellow-400",
  Drama: "bg-blue-500/20 text-blue-400",
  Thriller: "bg-orange-500/20 text-orange-400",
  "Sci-Fi": "bg-cyan-500/20 text-cyan-400",
  Horror: "bg-purple-500/20 text-purple-400",
  Romance: "bg-pink-500/20 text-pink-400",
  Animation: "bg-lime-500/20 text-lime-400",
  Documentary: "bg-teal-500/20 text-teal-400",
};

export default function MovieCard({
  movie,
  rank,
  userId,
  variant,
  onFeedback,
  feedbackSent = false,
  enableFeedback = true,
}: Props) {
  const [isExpanded, setIsExpanded] = useState(false);
  const canSendFeedback = enableFeedback && typeof onFeedback === "function";
  const banditScore = movie.bandit_score ?? movie.score;
  const similarityScore = movie.similarity_score ?? movie.score;

  // Extract year from title like "Toy Story (1995)"
  const yearMatch = movie.title.match(/\((\d{4})\)$/);
  const year = yearMatch ? yearMatch[1] : movie.year?.toString();
  const cleanTitle = movie.title.replace(/\s*\(\d{4}\)$/, "");

  // Score bar width (score is 0–1 from LightGBM)
  const scorePercent = Math.round(movie.score * 100);

  return (
    <div className={`card group cursor-pointer animate-slide-up
      ${feedbackSent ? "opacity-60" : ""}
    `}
      style={{ animationDelay: `${rank * 50}ms` }}
      onClick={() => setIsExpanded(!isExpanded)}
    >
      {/* Rank + Title Row */}
      <div className="flex items-start gap-3">
        {/* Rank number */}
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-cinema-accent/10
          border border-cinema-accent/30 flex items-center justify-center">
          <span className="text-cinema-accent text-xs font-bold">{rank}</span>
        </div>

        {/* Title + Year + Genres */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="font-semibold text-cinema-text text-sm leading-tight truncate">
              {cleanTitle}
            </h3>
            {year && (
              <span className="text-cinema-muted text-xs flex-shrink-0">({year})</span>
            )}
          </div>

          {/* Genre tags */}
          <div className="flex flex-wrap gap-1 mt-1.5">
            {movie.genres.slice(0, 4).map((genre) => (
              <span
                key={genre}
                className={`text-xs px-2 py-0.5 rounded-full
                  ${GENRE_COLORS[genre] ?? "bg-cinema-border text-cinema-muted"}`}
              >
                {genre}
              </span>
            ))}
          </div>
        </div>

        {/* Score */}
        <div className="flex-shrink-0 text-right">
          <div className="text-cinema-accent font-bold text-sm">
            {scorePercent}%
          </div>
          <div className="text-cinema-muted text-xs">match</div>
        </div>
      </div>

      {/* Score bar */}
      <div className="mt-3 h-1 bg-cinema-border rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-cinema-accent2 to-cinema-accent
            rounded-full transition-all duration-700 ease-out"
          style={{ width: `${scorePercent}%` }}
        />
      </div>

      {/* Expanded details */}
      {isExpanded && (
        <div className="mt-3 pt-3 border-t border-cinema-border animate-fade-in">
          <div className="grid grid-cols-3 gap-3 text-xs">
            <div>
              <div className="text-cinema-muted mb-0.5">LightGBM Score</div>
              <div className="font-medium">{movie.score.toFixed(4)}</div>
            </div>
            <div>
              <div className="text-cinema-muted mb-0.5">Bandit Score</div>
              <div className="font-medium">{banditScore.toFixed(4)}</div>
            </div>
            <div>
              <div className="text-cinema-muted mb-0.5">Similarity</div>
              <div className="font-medium">{similarityScore.toFixed(4)}</div>
            </div>
          </div>
        </div>
      )}

      {/* Feedback buttons */}
      {canSendFeedback && !feedbackSent && (
        <div
          className="mt-3 flex gap-2 justify-end opacity-0 group-hover:opacity-100
            transition-opacity duration-200"
          onClick={(e) => e.stopPropagation()}
        >
          <FeedbackButton
            label="👍 Like"
            onClick={() => onFeedback?.(movie.item_id, 1)}
            variant="positive"
          />
          <FeedbackButton
            label="👎 Dislike"
            onClick={() => onFeedback?.(movie.item_id, 0)}
            variant="negative"
          />
        </div>
      )}

      {canSendFeedback && feedbackSent && (
        <div className="mt-2 text-xs text-cinema-muted text-right">
          ✓ Feedback recorded
        </div>
      )}
    </div>
  );
}