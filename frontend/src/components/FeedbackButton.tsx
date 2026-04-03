import { useState } from "react";

interface Props {
  label: string;
  onClick: () => void | Promise<void>;
  variant: "positive" | "negative";
}

export default function FeedbackButton({ label, onClick, variant }: Props) {
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleClick = async () => {
    setIsSubmitting(true);
    try {
      await onClick();
    } finally {
      setIsSubmitting(false);
    }
  };

  const baseClass =
    variant === "positive"
      ? "px-3 py-1 text-xs rounded-lg border transition-all hover:bg-green-500/20 border-green-500/30 text-green-400"
      : "px-3 py-1 text-xs rounded-lg border transition-all hover:bg-red-500/20 border-red-500/30 text-red-400";

  return (
    <button
      className={`${baseClass} ${isSubmitting ? "opacity-50 cursor-not-allowed" : ""}`}
      onClick={handleClick}
      disabled={isSubmitting}
    >
      {isSubmitting ? "..." : label}
    </button>
  );
}