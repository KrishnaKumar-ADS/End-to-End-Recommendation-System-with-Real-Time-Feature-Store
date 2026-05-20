import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { IngestForm } from "@/components/ingest/IngestForm";

describe("IngestForm", () => {
  it("validates GitHub URL before submitting", async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);

    render(<IngestForm error={null} isLoading={false} onSubmit={onSubmit} />);

    await userEvent.type(
      screen.getByPlaceholderText("https://github.com/tiangolo/fastapi"),
      "not-a-github-url",
    );
    await userEvent.click(screen.getByRole("button", { name: "Index Repository" }));

    expect(screen.getByRole("alert")).toHaveTextContent("Enter a valid GitHub repository URL.");
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("submits normalized payload and defaults empty branch to main", async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);

    render(<IngestForm error={null} isLoading={false} onSubmit={onSubmit} />);

    await userEvent.type(
      screen.getByPlaceholderText("https://github.com/tiangolo/fastapi"),
      "  https://github.com/tiangolo/fastapi  ",
    );
    await userEvent.clear(screen.getByDisplayValue("main"));
    await userEvent.click(screen.getByRole("button", { name: "Index Repository" }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit).toHaveBeenCalledWith({
      github_url: "https://github.com/tiangolo/fastapi",
      branch: "main",
    });
  });

  it("shows backend error message when provided", () => {
    render(
      <IngestForm
        error="Repository already exists"
        isLoading={false}
        onSubmit={vi.fn().mockResolvedValue(undefined)}
      />,
    );

    expect(screen.getByText("Repository already exists")).toBeInTheDocument();
  });
});
