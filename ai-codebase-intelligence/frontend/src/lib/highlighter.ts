type ShikiHighlighter = {
  codeToHtml: (code: string, options: { lang: string; theme: "dark-plus" | "light-plus" }) => string;
};

let highlighterPromise: Promise<ShikiHighlighter> | null = null;

export function getHighlighter(): Promise<ShikiHighlighter> {
  if (!highlighterPromise) {
    highlighterPromise = import("shiki").then(async ({ createHighlighter }) => {
      const highlighter = await createHighlighter({
        themes: ["dark-plus", "light-plus"],
        langs: [
          "python",
          "typescript",
          "javascript",
          "tsx",
          "jsx",
          "json",
          "yaml",
          "bash",
          "shell",
          "sql",
          "go",
          "rust",
          "markdown",
          "html",
          "css",
          "text",
        ],
      });

      return highlighter as ShikiHighlighter;
    });
  }

  return highlighterPromise;
}

function normaliseLanguage(lang: string): string {
  const map: Record<string, string> = {
    py: "python",
    ts: "typescript",
    js: "javascript",
    sh: "bash",
    shell: "bash",
    yml: "yaml",
    plaintext: "text",
    "": "text",
  };

  const normalized = lang.trim().toLowerCase();
  return map[normalized] ?? normalized;
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export async function highlightCode(
  code: string,
  lang: string,
  theme: "dark-plus" | "light-plus" = "dark-plus",
): Promise<string> {
  const language = normaliseLanguage(lang);

  try {
    const highlighter = await getHighlighter();
    return highlighter.codeToHtml(code, { lang: language, theme });
  } catch (error) {
    if (process.env.NODE_ENV === "development") {
      // eslint-disable-next-line no-console
      console.warn(`[highlighter] failed to highlight ${language}`, error);
    }

    return `<pre class=\"shiki-fallback\"><code>${escapeHtml(code)}</code></pre>`;
  }
}
