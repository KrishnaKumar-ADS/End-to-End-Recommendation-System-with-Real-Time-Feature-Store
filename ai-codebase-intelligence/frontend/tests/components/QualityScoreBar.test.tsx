import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { QualityScorePanel } from "@/components/Chat/QualityScoreBar";
import type { QualityScore } from "@/types/api";

const qualityScore: QualityScore = {
  faithfulness: 0.9,
  relevance: 0.75,
  completeness: 0.6,
  overall: 0.75,
  critique: "Answer misses a specific edge case.",
};

describe("QualityScorePanel", () => {
  it("renders title and overall percentage", () => {
    render(<QualityScorePanel qualityScore={qualityScore} />);

    expect(screen.getByText("Answer quality")).toBeInTheDocument();
    expect(screen.getByText("Overall: 75%")).toBeInTheDocument();
  });

  it("renders progress bars with expected values", () => {
    render(<QualityScorePanel qualityScore={qualityScore} />);

    expect(screen.getByLabelText("Faithfulness: 90%")).toHaveAttribute("aria-valuenow", "90");
    expect(screen.getByLabelText("Relevance: 75%")).toHaveAttribute("aria-valuenow", "75");
    expect(screen.getByLabelText("Completeness: 60%")).toHaveAttribute("aria-valuenow", "60");
  });

  it("shows and hides critique details", async () => {
    const user = userEvent.setup();
    render(<QualityScorePanel qualityScore={qualityScore} />);

    expect(screen.queryByText('"Answer misses a specific edge case."')).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /See critique/i }));
    expect(screen.getByText('"Answer misses a specific edge case."')).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Hide critique/i }));
    expect(screen.queryByText('"Answer misses a specific edge case."')).not.toBeInTheDocument();
  });

  it("renders skipped state when evaluation was skipped", () => {
    render(
      <QualityScorePanel
        qualityScore={{
          ...qualityScore,
          skipped: true,
          skip_reason: "Token budget exceeded",
        }}
      />,
    );

    expect(screen.getByText("Quality scoring skipped - Token budget exceeded.")).toBeInTheDocument();
  });

  it("renders score status labels", () => {
    render(<QualityScorePanel qualityScore={qualityScore} />);

    expect(screen.getByText("OK")).toBeInTheDocument();
    expect(screen.getAllByText("WARN").length).toBeGreaterThan(0);
  });
});
