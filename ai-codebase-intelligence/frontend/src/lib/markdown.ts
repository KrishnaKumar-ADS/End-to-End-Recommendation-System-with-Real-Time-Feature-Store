export interface TextSegment {
  kind: "text";
  content: string;
}

export interface CodeSegment {
  kind: "code";
  language: string;
  content: string;
}

export type AnswerSegment = TextSegment | CodeSegment;

const CODE_FENCE_REGEX = /```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```/g;

export function parseAnswer(answer: string): AnswerSegment[] {
  if (!answer) {
    return [];
  }

  const segments: AnswerSegment[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = CODE_FENCE_REGEX.exec(answer)) !== null) {
    const [fullMatch, language, code] = match;
    const matchStart = match.index;

    if (matchStart > lastIndex) {
      const textContent = answer.slice(lastIndex, matchStart);
      if (textContent) {
        segments.push({ kind: "text", content: textContent });
      }
    }

    segments.push({
      kind: "code",
      language: language.trim().toLowerCase() || "text",
      content: code,
    });

    lastIndex = matchStart + fullMatch.length;
  }

  if (lastIndex < answer.length) {
    const remaining = answer.slice(lastIndex);
    if (remaining) {
      segments.push({ kind: "text", content: remaining });
    }
  }

  if (!segments.length) {
    segments.push({ kind: "text", content: answer });
  }

  return segments;
}

export function extractCodeBlocks(answer: string): CodeSegment[] {
  return parseAnswer(answer).filter((segment): segment is CodeSegment => segment.kind === "code");
}

export function stripCodeBlocks(answer: string): string {
  return answer.replace(/```[\s\S]*?```/g, "[code block]").trim();
}

export function languageDisplayName(lang: string): string {
  const map: Record<string, string> = {
    python: "Python",
    py: "Python",
    typescript: "TypeScript",
    ts: "TypeScript",
    javascript: "JavaScript",
    js: "JavaScript",
    tsx: "TSX",
    jsx: "JSX",
    bash: "Bash",
    sh: "Shell",
    shell: "Shell",
    json: "JSON",
    yaml: "YAML",
    yml: "YAML",
    sql: "SQL",
    html: "HTML",
    css: "CSS",
    go: "Go",
    rust: "Rust",
    java: "Java",
    cpp: "C++",
    c: "C",
    text: "Text",
    plaintext: "Text",
    "": "Text",
  };

  return map[lang.toLowerCase()] ?? lang.toUpperCase();
}
