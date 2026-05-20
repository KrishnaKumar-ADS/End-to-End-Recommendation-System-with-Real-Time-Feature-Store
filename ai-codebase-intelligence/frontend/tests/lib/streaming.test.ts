import { describe, expect, it } from "vitest";

import {
  collectSSEStream,
  makeChunkedTestStream,
  makeTestStream,
  readSSEStream,
} from "@/lib/streaming";
import type { ChatEvent } from "@/types/api";

function streamFromString(payload: string): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoder.encode(payload));
      controller.close();
    },
  });
}

async function readAll(
  stream: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): Promise<ChatEvent[]> {
  const events: ChatEvent[] = [];
  for await (const event of readSSEStream(stream, signal)) {
    events.push(event);
  }
  return events;
}

describe("streaming", () => {
  it("parses token events that use text field", async () => {
    const events = await readAll(streamFromString('event: token\ndata: {"text":"hello"}\n\n'));

    expect(events).toEqual([{ type: "token", content: "hello" }]);
  });

  it("parses payload-driven token events without event header", async () => {
    const events = await readAll(streamFromString('data: {"type":"token","content":"world"}\n\n'));

    expect(events).toEqual([{ type: "token", content: "world" }]);
  });

  it("normalizes delta events to token events", async () => {
    const events = await readAll(streamFromString('event: delta\ndata: {"text":"chunk"}\n\n'));

    expect(events).toEqual([{ type: "token", content: "chunk" }]);
  });

  it("parses step and error events", async () => {
    const payload = [
      'event: step\ndata: {"stage":"searching","message":"Searching code context..."}\n\n',
      'event: error\ndata: {"error":"LLM failed"}\n\n',
    ].join("");

    const events = await readAll(streamFromString(payload));

    expect(events).toEqual([
      { type: "step", stage: "searching", message: "Searching code context..." },
      { type: "error", message: "LLM failed" },
    ]);
  });

  it("parses sources events from backend source shape", async () => {
    const payload =
      'event: sources\ndata: {"sources":[{"file_path":"app/auth.py","function_name":"login","start_line":10,"end_line":22,"snippet":"def login(): pass"}]}\n\n';

    const events = await readAll(streamFromString(payload));

    expect(events).toEqual([
      {
        type: "sources",
        sources: [
          {
            file: "app/auth.py",
            function: "login",
            lines: "10-22",
            snippet: "def login(): pass",
          },
        ],
      },
    ]);
  });

  it("parses done events and timing metadata", async () => {
    const payload =
      'event: done\ndata: {"session_id":"s1","quality_score":null,"graph_path":["A","B"],"provider_used":"openrouter","model_used":"qwen","timing":{"search_ms":10,"graph_ms":5,"context_ms":2,"total_ms":17}}\n\n';

    const events = await readAll(streamFromString(payload));

    expect(events).toEqual([
      {
        type: "done",
        session_id: "s1",
        quality_score: null,
        graph_path: ["A", "B"],
        provider_used: "openrouter",
        model_used: "qwen",
        provider: undefined,
        model: undefined,
        cached: undefined,
        total_ms: 17,
        timing: {
          search_ms: 10,
          graph_ms: 5,
          context_ms: 2,
          total_ms: 17,
        },
      },
    ]);
  });

  it("ignores malformed payloads and done markers", async () => {
    const payload = [
      "event: token\ndata: not-json\n\n",
      "event: token\ndata: {\"text\":\"ok\"}\n\n",
      "event: done\ndata: [DONE]\n\n",
    ].join("");

    const events = await readAll(streamFromString(payload));

    expect(events).toEqual([{ type: "token", content: "ok" }]);
  });

  it("supports abort signals while reading", async () => {
    const events = [
      { type: "token", content: "part1" },
      { type: "token", content: "part2" },
      { type: "done", session_id: null, quality_score: null, graph_path: [] },
    ] satisfies ChatEvent[];

    const abortController = new AbortController();
    abortController.abort();

    const parsed = await readAll(makeChunkedTestStream(events, 3), abortController.signal);

    expect(parsed).toEqual([]);
  });

  it("collects complete streams from utility builders", async () => {
    const events = [
      { type: "token", content: "hello" },
      {
        type: "done",
        session_id: "s1",
        quality_score: null,
        graph_path: ["nodeA"],
        provider_used: "openrouter",
        model_used: "qwen",
      },
    ] satisfies ChatEvent[];

    const parsed = await collectSSEStream(makeTestStream(events));

    expect(parsed).toHaveLength(2);
    expect(parsed[0]).toEqual({ type: "token", content: "hello" });
    expect(parsed[1]).toMatchObject({ type: "done", session_id: "s1" });
  });
});
