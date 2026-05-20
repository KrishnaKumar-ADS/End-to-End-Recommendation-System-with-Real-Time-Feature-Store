import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ToastProvider, useToast } from "@/components/ui/Toast";

function Trigger() {
  const { toast } = useToast();
  return (
    <button
      onClick={() =>
        toast({
          title: "Saved",
          description: "The design changes were stored.",
          variant: "success",
        })
      }
      type="button"
    >
      Trigger
    </button>
  );
}

describe("Toast", () => {
  it("useToast throws outside the provider", () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => render(<Trigger />)).toThrow("useToast must be used inside a <ToastProvider>");
    consoleError.mockRestore();
  });

  it("renders a toast inside the provider", async () => {
    render(
      <ToastProvider>
        <Trigger />
      </ToastProvider>,
    );

    await userEvent.click(screen.getByRole("button", { name: "Trigger" }));
    expect(screen.getByText("Saved")).toBeInTheDocument();
    expect(screen.getByText("The design changes were stored.")).toBeInTheDocument();
  });
});
