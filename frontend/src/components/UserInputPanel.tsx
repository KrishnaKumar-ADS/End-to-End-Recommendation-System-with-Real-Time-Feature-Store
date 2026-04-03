import { useState } from "react";

interface Props {
  onSubmit: (userId: number) => void;
  isLoading: boolean;
  currentUserId: number | null;
}

// Quick-access sample user IDs from MovieLens
const SAMPLE_USER_IDS = [1, 42, 100, 500, 1000, 5000, 10000];

export default function UserInputPanel({ onSubmit, isLoading, currentUserId }: Props) {
  const [inputValue, setInputValue] = useState<string>("");
  const [error, setError] = useState<string>("");

  const handleSubmit = () => {
    const parsed = parseInt(inputValue.trim(), 10);

    if (isNaN(parsed) || parsed <= 0) {
      setError("Please enter a valid positive user ID");
      return;
    }
    if (parsed > 162541) {   // MovieLens 25M max user ID
      setError("User ID must be between 1 and 162,541 (MovieLens 25M range)");
      return;
    }

    setError("");
    onSubmit(parsed);
  };

  return (
    <div className="card p-6 animate-fade-in">
      <div className="flex flex-col gap-4">

        {/* Header */}
        <div>
          <h2 className="text-xl font-bold">Get Recommendations</h2>
          <p className="text-cinema-muted text-sm mt-1">
            Enter a MovieLens user ID to see personalized movie recommendations.
          </p>
        </div>

        {/* Input Row */}
        <div className="flex gap-3 items-start">
          <div className="flex-1">
            <input
              type="number"
              className="input-field"
              placeholder="Enter User ID (1 – 162,541)"
              value={inputValue}
              onChange={(e) => {
                setInputValue(e.target.value);
                setError("");
              }}
              onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
              min={1}
              max={162541}
            />
            {error && (
              <p className="text-cinema-accent text-xs mt-1">{error}</p>
            )}
          </div>
          <button
            className="btn-primary whitespace-nowrap"
            onClick={handleSubmit}
            disabled={isLoading || !inputValue}
          >
            {isLoading ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                </svg>
                Loading...
              </span>
            ) : "Get Recommendations"}
          </button>
        </div>

        {/* Sample User IDs */}
        <div>
          <p className="text-cinema-muted text-xs mb-2">Try a sample user:</p>
          <div className="flex flex-wrap gap-2">
            {SAMPLE_USER_IDS.map((id) => (
              <button
                key={id}
                className={`px-3 py-1 rounded-full text-xs font-medium transition-all
                  ${currentUserId === id
                    ? "bg-cinema-accent text-white"
                    : "bg-cinema-border text-cinema-muted hover:text-cinema-text hover:bg-cinema-accent/20"
                  }`}
                onClick={() => {
                  setInputValue(String(id));
                  setError("");
                  onSubmit(id);
                }}
              >
                User {id}
              </button>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}