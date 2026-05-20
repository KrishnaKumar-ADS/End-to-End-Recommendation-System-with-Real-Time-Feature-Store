import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { ChatMessage } from "@/components/Chat/ChatMessage";
import type { AssistantMessage, ChatMessage as ChatMessageType, UserMessage } from "@/types/api";

function makeUserMessage(content = "How does auth work?"): UserMessage {
  return {
    id: "user-1",
    role: "user",
    content,
    timestamp: Date.now(),
  };
}

function makeAssistantMessage(partial?: Partial<AssistantMessage>): AssistantMessage {
  return {
    id: "assistant-1",
    role: "assistant",
    content: "Auth is handled in app/auth.py",
    sources: [],
    qualityScore: null,
    graphPath: [],
    providerUsed: "openrouter",
    modelUsed: "qwen",
    cached: false,
    totalMs: 1300,
    isStreaming: false,
    timestamp: Date.now(),
    ...partial,
  };
}

describe("ChatMessage", () => {
  it("renders user messages as user bubbles", () => {
    const message: ChatMessageType = makeUserMessage("Explain repository structure");
    render(<ChatMessage message={message} repoId="repo-1" />);

    expect(screen.getByText("Explain repository structure")).toBeInTheDocument();
  });

  it("shows streaming indicator for assistant messages without content", () => {
    const message: ChatMessageType = makeAssistantMessage({
      content: "",
      isStreaming: true,
      modelUsed: "qwen/qwen-2.5-coder-32b-instruct",
    });

    render(<ChatMessage message={message} repoId="repo-1" />);

    expect(screen.getByLabelText("AI is generating a response")).toBeInTheDocument();
    expect(screen.getByText("thinking...")).toBeInTheDocument();
  });

  it("renders assistant content and metadata", () => {
    const message: ChatMessageType = makeAssistantMessage({
      content: "The login flow starts in app/main.py",
      totalMs: 2100,
    });

    render(<ChatMessage message={message} repoId="repo-1" />);

    expect(screen.getByText("The login flow starts in app/main.py")).toBeInTheDocument();
    expect(screen.getByText("qwen via openrouter")).toBeInTheDocument();
    expect(screen.getByText("2.1s")).toBeInTheDocument();
  });

  it("renders source citation links for completed answers", () => {
    const message: ChatMessageType = makeAssistantMessage({
      sources: [
        {
          file: "app/main.py",
          function: "login",
          lines: "10-22",
          snippet: "def login():\n    return True",
        },
      ],
    });

    render(<ChatMessage message={message} repoId="repo-1" />);

    const link = screen.getByRole("link", { name: "app/main.py" });
    expect(link).toHaveAttribute("href", "/repos/repo-1?file=app%2Fmain.py");
    expect(screen.getByText("login()")).toBeInTheDocument();
  });

  it("expands source snippets on demand", async () => {
    const user = userEvent.setup();
    const message: ChatMessageType = makeAssistantMessage({
      sources: [
        {
          file: "app/main.py",
          function: "login",
          lines: "10-22",
          snippet: "def login():\n    return True",
        },
      ],
    });

    render(<ChatMessage message={message} repoId="repo-1" />);

    await user.click(screen.getByRole("button", { name: "Show code snippet" }));

    expect(screen.getByLabelText("Python code")).toHaveTextContent("def login():");
  });

  it("renders quality and graph path sections", () => {
    const message: ChatMessageType = makeAssistantMessage({
      qualityScore: {
        faithfulness: 0.91,
        relevance: 0.83,
        completeness: 0.78,
        overall: 0.84,
        critique: "Add one concrete code reference.",
      },
      graphPath: ["api.login", "service.authenticate", "db.users"],
    });

    render(<ChatMessage message={message} repoId="repo-1" />);

    expect(screen.getByText("Answer quality")).toBeInTheDocument();
    expect(screen.getByText("Call path:")).toBeInTheDocument();
    expect(screen.getByText("api.login")).toBeInTheDocument();
    expect(screen.getByText("service.authenticate")).toBeInTheDocument();
    expect(screen.getByText("db.users")).toBeInTheDocument();
  });
});
