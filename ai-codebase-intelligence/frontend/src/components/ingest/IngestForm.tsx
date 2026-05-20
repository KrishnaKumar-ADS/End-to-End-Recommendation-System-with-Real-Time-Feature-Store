"use client";

import { useState } from "react";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import { isGithubUrl } from "@/lib/utils";
import type { IngestRequest } from "@/types/api";

export function IngestForm({
  isLoading,
  error,
  onSubmit,
}: {
  isLoading: boolean;
  error: string | null;
  onSubmit: (payload: IngestRequest) => Promise<unknown>;
}) {
  const [githubUrl, setGithubUrl] = useState("");
  const [branch, setBranch] = useState("main");
  const [validationError, setValidationError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!isGithubUrl(githubUrl)) {
      setValidationError("Enter a valid GitHub repository URL.");
      return;
    }

    setValidationError(null);
    await onSubmit({
      github_url: githubUrl.trim(),
      branch: branch.trim() || "main",
    });
  };

  return (
    <Card className="space-y-5" padding="lg">
      <div className="space-y-2">
        <p className="text-xs uppercase tracking-[0.14em] text-brand-200">Step 1</p>
        <h2 className="text-2xl font-semibold text-white">Index a GitHub repository</h2>
        <p className="max-w-2xl text-sm text-surface-muted">
          Paste a public repository URL, choose an optional branch, and the backend will start cloning,
          parsing, and embedding the codebase into the platform.
        </p>
      </div>

      <form className="grid gap-4 md:grid-cols-[minmax(0,1fr),220px] md:items-end" onSubmit={handleSubmit}>
        <Input
          label="GitHub URL"
          value={githubUrl}
          placeholder="https://github.com/tiangolo/fastapi"
          error={validationError ?? undefined}
          hint="Only GitHub repository URLs are supported."
          onChange={(event) => setGithubUrl(event.target.value)}
        />
        <Input
          label="Branch"
          value={branch}
          hint="Defaults to main when empty."
          onChange={(event) => setBranch(event.target.value)}
        />

        <div className="md:col-span-2 flex flex-wrap items-center gap-3">
          <Button isLoading={isLoading} type="submit">
            Index Repository
          </Button>
          {error ? <p className="text-sm text-red-300">{error}</p> : null}
        </div>
      </form>
    </Card>
  );
}
