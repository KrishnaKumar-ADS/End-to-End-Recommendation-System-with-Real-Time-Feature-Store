import { describe, expect, it } from "vitest";

import {
  extractCodeBlocks,
  languageDisplayName,
  parseAnswer,
  stripCodeBlocks,
} from "@/lib/markdown";

describe("markdown", () => {
  it("returns an empty array for empty answers", () => {
    expect(parseAnswer("")).toEqual([]);
  });

  it("returns a text segment for plain text answers", () => {
    expect(parseAnswer("Hello world")).toEqual([
      {
        kind: "text",
        content: "Hello world",
      },
    ]);
  });

  it("parses a single fenced code block with language", () => {
    const answer = "Before\n```python\ndef hello():\n    return 1\n```\nAfter";
    const segments = parseAnswer(answer);

    expect(segments).toEqual([
      { kind: "text", content: "Before\n" },
      { kind: "code", language: "python", content: "def hello():\n    return 1\n" },
      { kind: "text", content: "\nAfter" },
    ]);
  });

  it("defaults code block language to text when omitted", () => {
    const answer = "```\nSELECT 1;\n```";
    const segments = parseAnswer(answer);

    expect(segments).toEqual([
      { kind: "code", language: "text", content: "SELECT 1;\n" },
    ]);
  });

  it("parses multiple code blocks", () => {
    const answer = [
      "A",
      "```ts",
      "const a = 1;",
      "```",
      "B",
      "```bash",
      "echo hi",
      "```",
    ].join("\n");

    const segments = parseAnswer(answer);

    expect(segments.filter((segment) => segment.kind === "code")).toHaveLength(2);
  });

  it("extractCodeBlocks only returns code segments", () => {
    const blocks = extractCodeBlocks("text\n```js\nconsole.log(1)\n```\nmore");

    expect(blocks).toEqual([
      { kind: "code", language: "js", content: "console.log(1)\n" },
    ]);
  });

  it("stripCodeBlocks replaces fenced code with placeholder", () => {
    const stripped = stripCodeBlocks("A\n```ts\nconst x = 1\n```\nB");

    expect(stripped).toBe("A\n[code block]\nB");
  });

  it("maps known language display names", () => {
    expect(languageDisplayName("py")).toBe("Python");
    expect(languageDisplayName("typescript")).toBe("TypeScript");
    expect(languageDisplayName("unknownlang")).toBe("UNKNOWNLANG");
  });
});
