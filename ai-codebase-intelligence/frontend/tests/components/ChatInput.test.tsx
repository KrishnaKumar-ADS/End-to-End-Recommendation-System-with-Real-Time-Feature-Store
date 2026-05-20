import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ChatInput } from "@/components/Chat/ChatInput";

describe("ChatInput", () => {
  it("submits from the send button and clears input", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();

    render(<ChatInput isStreaming={false} onCancel={vi.fn()} onSubmit={onSubmit} />);

    const input = screen.getByLabelText("Chat input");
    await user.type(input, "Explain auth flow");
    await user.click(screen.getByRole("button", { name: "Send question" }));

    expect(onSubmit).toHaveBeenCalledWith("Explain auth flow");
    expect(input).toHaveValue("");
  });

  it("submits on Enter", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();

    render(<ChatInput isStreaming={false} onCancel={vi.fn()} onSubmit={onSubmit} />);

    const input = screen.getByLabelText("Chat input");
    await user.type(input, "What is middleware?{enter}");

    expect(onSubmit).toHaveBeenCalledWith("What is middleware?");
  });

  it("submits on Ctrl+Enter", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();

    render(<ChatInput isStreaming={false} onCancel={vi.fn()} onSubmit={onSubmit} />);

    const input = screen.getByLabelText("Chat input");
    await user.type(input, "Explain this module{Control>}{Enter}{/Control}");

    expect(onSubmit).toHaveBeenCalledWith("Explain this module");
  });

  it("allows Shift+Enter without submitting", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();

    render(<ChatInput isStreaming={false} onCancel={vi.fn()} onSubmit={onSubmit} />);

    const input = screen.getByLabelText("Chat input");
    await user.type(input, "Line 1{Shift>}{Enter}{/Shift}Line 2");

    expect(onSubmit).not.toHaveBeenCalled();
    expect(input).toHaveValue("Line 1\nLine 2");
  });

  it("does not submit empty or whitespace-only text", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();

    render(<ChatInput isStreaming={false} onCancel={vi.fn()} onSubmit={onSubmit} />);

    const input = screen.getByLabelText("Chat input");
    await user.type(input, "   ");
    await user.keyboard("{enter}");

    expect(onSubmit).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Send question" })).toBeDisabled();
  });

  it("respects disabled state", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();

    render(<ChatInput disabled isStreaming={false} onCancel={vi.fn()} onSubmit={onSubmit} />);

    const input = screen.getByLabelText("Chat input");
    expect(input).toBeDisabled();

    await user.type(input, "Should not type");
    expect(input).toHaveValue("");
  });

  it("shows stop button while streaming and triggers cancel", async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();

    render(<ChatInput isStreaming onCancel={onCancel} onSubmit={vi.fn()} />);

    const stopButton = screen.getByRole("button", { name: "Stop generating" });
    expect(stopButton).toBeInTheDocument();

    await user.click(stopButton);
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("disables submit while streaming", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();

    render(<ChatInput isStreaming onCancel={vi.fn()} onSubmit={onSubmit} />);

    const input = screen.getByLabelText("Chat input");
    await user.type(input, "Question during stream{enter}");

    expect(onSubmit).not.toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: "Send question" })).not.toBeInTheDocument();
  });

  it("supports controlled value for prefilled questions", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    const onValueChange = vi.fn();

    render(
      <ChatInput
        isStreaming={false}
        onCancel={vi.fn()}
        onSubmit={onSubmit}
        onValueChange={onValueChange}
        value="Explain login in app/auth.py"
      />,
    );

    const input = screen.getByLabelText("Chat input");
    expect(input).toHaveValue("Explain login in app/auth.py");

    await user.click(screen.getByRole("button", { name: "Send question" }));

    expect(onSubmit).toHaveBeenCalledWith("Explain login in app/auth.py");
    expect(onValueChange).toHaveBeenCalledWith("");
  });
});
