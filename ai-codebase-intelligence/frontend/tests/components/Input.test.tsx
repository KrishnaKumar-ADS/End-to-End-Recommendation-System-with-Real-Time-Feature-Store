import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { Input } from "@/components/ui/Input";

describe("Input", () => {
  it("renders with label", () => {
    render(<Input label="GitHub URL" />);
    expect(screen.getByLabelText("GitHub URL")).toBeInTheDocument();
  });

  it("shows error message and sets aria-invalid", () => {
    render(<Input error="This field is required" label="URL" />);
    expect(screen.getByRole("alert")).toHaveTextContent("This field is required");
    expect(screen.getByRole("textbox")).toHaveAttribute("aria-invalid", "true");
  });

  it("shows hint when no error is present", () => {
    render(<Input hint="Enter a GitHub URL" label="URL" />);
    expect(screen.getByText("Enter a GitHub URL")).toBeInTheDocument();
  });

  it("does not show hint when error is present", () => {
    render(<Input error="Bad URL" hint="Enter a GitHub URL" label="URL" />);
    expect(screen.queryByText("Enter a GitHub URL")).not.toBeInTheDocument();
  });

  it("calls onChange when typing", async () => {
    const handler = vi.fn();
    render(<Input label="URL" onChange={handler} />);
    await userEvent.type(screen.getByRole("textbox"), "hello");
    expect(handler).toHaveBeenCalled();
  });

  it("is disabled when disabled prop is set", () => {
    render(<Input disabled label="URL" />);
    expect(screen.getByRole("textbox")).toBeDisabled();
  });
});
