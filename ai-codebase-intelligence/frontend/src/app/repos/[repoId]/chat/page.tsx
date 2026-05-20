"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { ChatInput } from "@/components/Chat/ChatInput";
import { ChatMessageList } from "@/components/Chat/ChatMessageList";
import { WelcomeScreen } from "@/components/Chat/WelcomeScreen";
import { FileTree } from "@/components/CodeViewer/FileTree";
import { SearchPanel } from "@/components/search/SearchPanel";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Spinner } from "@/components/ui/Spinner";
import { useChat } from "@/hooks/useChat";
import { useRepo } from "@/hooks/useRepo";
import { buildFileTree, repoNameFromUrl } from "@/lib/utils";

export default function RepoChatPage({
  params,
}: {
  params: { repoId: string };
}) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const initialQuestion = searchParams.get("question") ?? "";
  const { repo, isLoading: isRepoLoading } = useRepo(params.repoId);

  const [searchOpen, setSearchOpen] = useState(false);
  const [filesDrawerOpen, setFilesDrawerOpen] = useState(false);
  const [draftQuestion, setDraftQuestion] = useState("");
  const autoSubmitted = useRef(false);

  const {
    messages,
    isStreaming,
    sessionId,
    error,
    stepMessage,
    ask,
    cancelStream,
    newSession,
    clearError,
  } = useChat(params.repoId);

  const repoName = repo ? repoNameFromUrl(repo.github_url) : params.repoId;
  const tree = useMemo(() => buildFileTree(repo?.files ?? []), [repo?.files]);

  useEffect(() => {
    if (!initialQuestion || autoSubmitted.current) {
      return;
    }

    autoSubmitted.current = true;
    const timer = window.setTimeout(() => {
      void ask(initialQuestion);
      router.replace(`/repos/${params.repoId}/chat`, { scroll: false });
    }, 300);

    return () => {
      window.clearTimeout(timer);
    };
  }, [ask, initialQuestion, params.repoId, router]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setSearchOpen((current) => !current);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const handleSearchSelect = (question: string) => {
    setSearchOpen(false);
    setDraftQuestion(question);
  };

  const handleSubmit = (question: string) => {
    setDraftQuestion("");
    void ask(question);
  };

  const handleFileExplain = (filePath: string) => {
    const question = `Explain the file ${filePath}`;
    setDraftQuestion("");
    void ask(question);
    setFilesDrawerOpen(false);
  };

  return (
    <>
      <div className="flex h-[calc(100vh-73px)] min-h-0 overflow-hidden rounded-2xl border border-surface-border">
        <aside className="hidden h-full w-72 shrink-0 border-r border-surface-border bg-surface-card lg:flex lg:flex-col">
          <div className="border-b border-surface-border px-3 py-2 text-xs uppercase tracking-[0.1em] text-surface-muted">
            Files
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto px-2 py-2">
            <FileTree
              nodes={tree}
              onFileClick={(file) => handleFileExplain(file.file_path)}
              onFileGraphClick={(file) =>
                router.push(`/repos/${params.repoId}/graph?file=${encodeURIComponent(file.file_path)}`)
              }
            />
          </div>
        </aside>

        <div className="flex min-w-0 flex-1 flex-col">
          <div className="flex shrink-0 items-center justify-between gap-3 border-b border-surface-border bg-surface-card px-4 py-3">
            <div className="flex min-w-0 items-center gap-2">
              <span aria-hidden="true" className="text-lg">
                💬
              </span>
              {isRepoLoading ? (
                <Spinner size="sm" />
              ) : (
                <span className="truncate font-mono text-sm text-slate-200">{repoName}</span>
              )}
              {sessionId ? (
                <Badge className="shrink-0 font-mono text-[10px] normal-case">session active</Badge>
              ) : null}
            </div>

            <div className="flex shrink-0 items-center gap-2">
              <Button
                className="lg:hidden"
                onClick={() => setFilesDrawerOpen(true)}
                size="sm"
                title="Browse files"
                variant="ghost"
              >
                Files
              </Button>

              <Button
                aria-label="Open code search panel"
                onClick={() => setSearchOpen(true)}
                size="sm"
                title="Search code (Ctrl/Cmd+K)"
                variant="ghost"
              >
                🔍
              </Button>

              {(sessionId || messages.length > 0) ? (
                <Button
                  onClick={newSession}
                  size="sm"
                  title="Clear conversation and start a new session"
                  variant="ghost"
                >
                  ↺ New Session
                </Button>
              ) : null}

              <Button onClick={() => router.push(`/repos/${params.repoId}/graph`)} size="sm" variant="ghost">
                ⬡ Graph
              </Button>
              <Button onClick={() => router.push(`/repos/${params.repoId}`)} size="sm" variant="ghost">
                ← Repo
              </Button>
            </div>
          </div>

          {error ? (
            <div
              className="flex shrink-0 items-center justify-between gap-3 border-b border-red-500/30 bg-red-500/10 px-4 py-2.5"
              role="alert"
            >
              <p className="text-sm text-red-300">{error}</p>
              <Button onClick={clearError} size="sm" variant="ghost">
                ✕
              </Button>
            </div>
          ) : null}

          {messages.length === 0 && !isStreaming ? (
            <WelcomeScreen onQuestion={(question) => void ask(question)} repoName={repoName} />
          ) : (
            <ChatMessageList isStreaming={isStreaming} messages={messages} repoId={params.repoId} />
          )}

          {stepMessage && isStreaming ? (
            <div className="shrink-0 border-t border-surface-border bg-surface-card px-4 py-2 text-xs text-surface-muted">
              {stepMessage}
            </div>
          ) : null}

          <div className="sticky bottom-0 z-10">
            <ChatInput
              disabled={!params.repoId}
              isStreaming={isStreaming}
              onCancel={cancelStream}
              onSubmit={handleSubmit}
              onValueChange={setDraftQuestion}
              placeholder={params.repoId ? "Ask anything about this codebase..." : "Loading repository..."}
              value={draftQuestion}
            />
          </div>
        </div>
      </div>

      {searchOpen ? (
        <SearchPanel onClose={() => setSearchOpen(false)} onSelect={handleSearchSelect} repoId={params.repoId} />
      ) : null}

      {filesDrawerOpen ? (
        <div className="fixed inset-0 z-50 flex items-end bg-black/50 p-3 lg:hidden" onClick={() => setFilesDrawerOpen(false)}>
          <div className="max-h-[80vh] w-full overflow-hidden rounded-2xl border border-surface-border bg-surface-card" onClick={(event) => event.stopPropagation()}>
            <div className="flex items-center justify-between border-b border-surface-border px-3 py-2">
              <p className="text-sm text-slate-200">Repository files</p>
              <Button onClick={() => setFilesDrawerOpen(false)} size="sm" variant="ghost">
                Close
              </Button>
            </div>
            <div className="max-h-[68vh] overflow-y-auto px-2 py-2">
              <FileTree
                nodes={tree}
                onFileClick={(file) => handleFileExplain(file.file_path)}
                onFileGraphClick={(file) => {
                  setFilesDrawerOpen(false);
                  router.push(`/repos/${params.repoId}/graph?file=${encodeURIComponent(file.file_path)}`);
                }}
              />
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
