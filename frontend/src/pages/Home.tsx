import { useState } from "react";
import UserInputPanel from "@/components/UserInputPanel";
import MovieSearchPanel from "@/components/MovieSearchPanel";
import RecommendationGrid from "@/components/RecommendationGrid";
import { useMovieRecommendations, useRecommendations } from "@/hooks/useRecommendations";

export default function Home() {
  const [userId, setUserId] = useState<number | null>(null);
  const [movieTitle, setMovieTitle] = useState<string | null>(null);

  const userQuery = useRecommendations(userId);
  const movieQuery = useMovieRecommendations(movieTitle);

  const movieMode = movieTitle !== null;
  const { data, isLoading, isError, error } = movieMode ? movieQuery : userQuery;

  const handleUserSubmit = (nextUserId: number) => {
    setMovieTitle(null);
    setUserId(nextUserId);
  };

  const handleMovieSubmit = (nextMovieTitle: string) => {
    setUserId(null);
    setMovieTitle(nextMovieTitle);
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">

      {/* Hero header */}
      <div className="text-center py-6">
        <h1 className="text-3xl font-bold">
          Movie Recommendations
        </h1>
        <p className="text-cinema-muted mt-2">
          Powered by SASRec · Two-Tower FAISS · LightGBM · Thompson Sampling
        </p>
      </div>

      {/* Input panel */}
      <UserInputPanel
        onSubmit={handleUserSubmit}
        isLoading={isLoading}
        currentUserId={userId}
      />

      <MovieSearchPanel
        onSubmit={handleMovieSubmit}
        isLoading={isLoading}
        currentMovieTitle={movieTitle}
      />

      {/* Results */}
      {isLoading && (userId || movieTitle) && (
        <div className="card flex items-center justify-center py-12 text-cinema-muted">
          <div className="text-center">
            <div className="text-4xl animate-spin mb-3">⟳</div>
            {movieMode ? (
              <>
                <p>Finding movies similar to "{movieTitle}"...</p>
                <p className="text-xs mt-1">Title match → Two-Tower embedding → FAISS neighbors</p>
              </>
            ) : (
              <>
                <p>Running recommendation pipeline for User #{userId}...</p>
                <p className="text-xs mt-1">Feature store → FAISS → LightGBM → Bandit</p>
              </>
            )}
          </div>
        </div>
      )}

      {isError && (
        <div className="card border-cinema-accent/50 py-8 text-center">
          <div className="text-4xl mb-3">⚠</div>
          <p className="text-cinema-accent font-medium">Failed to load recommendations</p>
          <p className="text-cinema-muted text-sm mt-1">{error?.message}</p>
          <p className="text-cinema-muted text-xs mt-2">
            Is the FastAPI server running? Check: <code className="text-cinema-text">uvicorn backend.app.main:app</code>
          </p>
        </div>
      )}

      {data && !isLoading && (
        <RecommendationGrid data={data} />
      )}

      {!userId && !movieTitle && !isLoading && !data && (
        <div className="text-center py-12 text-cinema-muted">
          <div className="text-5xl mb-4">🎬</div>
          <p>Enter a user ID or search by movie title to get recommendations.</p>
        </div>
      )}
    </div>
  );
}