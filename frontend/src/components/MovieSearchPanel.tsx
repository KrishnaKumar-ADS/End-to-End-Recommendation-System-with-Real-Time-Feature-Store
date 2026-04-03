import { useEffect, useState } from "react";
import { useMovieTitleSuggestions } from "@/hooks/useRecommendations";

interface Props {
  onSubmit: (movieTitle: string) => void;
  isLoading: boolean;
  currentMovieTitle: string | null;
}

const SAMPLE_MOVIES = [
  "Toy Story",
  "The Matrix",
  "Inception",
  "Titanic",
  "Fight Club",
  "The Dark Knight",
];

export default function MovieSearchPanel({ onSubmit, isLoading, currentMovieTitle }: Props) {
  const [inputValue, setInputValue] = useState<string>("");
  const [debouncedInput, setDebouncedInput] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [showSuggestions, setShowSuggestions] = useState<boolean>(false);

  useEffect(() => {
    const handle = window.setTimeout(() => {
      setDebouncedInput(inputValue.trim());
    }, 220);

    return () => window.clearTimeout(handle);
  }, [inputValue]);

  const suggestionsQuery = useMovieTitleSuggestions(debouncedInput);
  const suggestions = suggestionsQuery.data?.suggestions ?? [];
  const shouldShowSuggestionBox = showSuggestions && inputValue.trim().length >= 2;

  const handleSubmit = () => {
    const title = inputValue.trim();

    if (title.length < 2) {
      setError("Please enter at least 2 characters for movie title");
      return;
    }

    setError("");
    setShowSuggestions(false);
    onSubmit(title);
  };

  const handleSuggestionClick = (title: string) => {
    setInputValue(title);
    setError("");
    setShowSuggestions(false);
    onSubmit(title);
  };

  return (
    <div className="card p-6 animate-fade-in">
      <div className="flex flex-col gap-4">
        <div>
          <h2 className="text-xl font-bold">Recommend by Movie</h2>
          <p className="text-cinema-muted text-sm mt-1">
            Enter a movie title and get similar movies instantly.
          </p>
        </div>

        <div className="flex gap-3 items-start">
          <div className="flex-1 relative">
            <input
              type="text"
              className="input-field"
              placeholder="Search a movie title (e.g., Toy Story)"
              value={inputValue}
              onChange={(e) => {
                setInputValue(e.target.value);
                setError("");
                setShowSuggestions(true);
              }}
              onFocus={() => setShowSuggestions(true)}
              onBlur={() => {
                window.setTimeout(() => setShowSuggestions(false), 120);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  handleSubmit();
                }
              }}
            />
            {error && <p className="text-cinema-accent text-xs mt-1">{error}</p>}

            {shouldShowSuggestionBox && (
              <div className="mt-2 max-h-56 overflow-y-auto rounded-lg border border-cinema-border bg-cinema-surface/95 backdrop-blur-sm shadow-lg">
                {suggestionsQuery.isFetching ? (
                  <p className="px-3 py-2 text-xs text-cinema-muted">Searching titles...</p>
                ) : suggestions.length > 0 ? (
                  suggestions.map((title) => (
                    <button
                      key={title}
                      type="button"
                      className="w-full px-3 py-2 text-left text-sm hover:bg-cinema-border/40 transition-colors"
                      onMouseDown={(e) => e.preventDefault()}
                      onClick={() => handleSuggestionClick(title)}
                    >
                      {title}
                    </button>
                  ))
                ) : (
                  <p className="px-3 py-2 text-xs text-cinema-muted">
                    No close matches yet. Keep typing to refine.
                  </p>
                )}
              </div>
            )}
          </div>

          <button
            className="btn-secondary whitespace-nowrap"
            onClick={handleSubmit}
            disabled={isLoading || !inputValue.trim()}
          >
            {isLoading ? "Searching..." : "Find Similar"}
          </button>
        </div>

        <div>
          <p className="text-cinema-muted text-xs mb-2">Try a sample movie:</p>
          <div className="flex flex-wrap gap-2">
            {SAMPLE_MOVIES.map((title) => (
              <button
                key={title}
                className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${
                  currentMovieTitle?.toLowerCase() === title.toLowerCase()
                    ? "bg-cinema-accent text-white"
                    : "bg-cinema-border text-cinema-muted hover:text-cinema-text hover:bg-cinema-accent/20"
                }`}
                onClick={() => {
                  setInputValue(title);
                  setError("");
                  setShowSuggestions(false);
                  onSubmit(title);
                }}
              >
                {title}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
