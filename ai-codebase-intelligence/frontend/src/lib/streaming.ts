import type {
  ChatEvent,
  DoneEvent,
  ErrorEvent,
  SourcesEvent,
  StepEvent,
  TokenEvent,
} from "@/types/api";

function toTokenEvent(payload: Record<string, unknown>): TokenEvent | null {
  const content = typeof payload.content === "string"
    ? payload.content
    : typeof payload.text === "string"
      ? payload.text
      : "";

  if (!content) {
    return null;
  }

  return {
    type: "token",
    content,
  };
}

function toSourcesEvent(payload: Record<string, unknown>): SourcesEvent {
  const sources = Array.isArray(payload.sources) ? payload.sources : [];

  const normalizedSources = sources
    .map((source): SourcesEvent["sources"][number] | null => {
      if (!source || typeof source !== "object") {
        return null;
      }
      const record = source as Record<string, unknown>;
      const file = String(record.file ?? record.file_path ?? "").trim();
      const fn = String(record.function ?? record.function_name ?? "").trim();
      const startLine = Number(record.start_line ?? 0);
      const endLine = Number(record.end_line ?? 0);
      const explicitLines = String(record.lines ?? "").trim();
      const lines = explicitLines || `${Math.max(0, startLine)}-${Math.max(0, endLine)}`;
      const snippet = typeof record.snippet === "string" ? record.snippet : undefined;

      if (snippet) {
        return {
          file,
          function: fn,
          lines,
          snippet,
        };
      }

      return {
        file,
        function: fn,
        lines,
      };
    })
    .filter((source): source is SourcesEvent["sources"][number] => source !== null);

  return {
    type: "sources",
    sources: normalizedSources,
  };
}

function toDoneEvent(payload: Record<string, unknown>): DoneEvent {
  const timing = payload.timing;
  const totalMsFromTiming =
    timing && typeof timing === "object" && Number.isFinite(Number((timing as Record<string, unknown>).total_ms))
      ? Number((timing as Record<string, unknown>).total_ms)
      : undefined;

  return {
    type: "done",
    session_id:
      typeof payload.session_id === "string"
        ? payload.session_id
        : payload.session_id === null
          ? null
          : null,
    quality_score:
      payload.quality_score && typeof payload.quality_score === "object"
        ? (payload.quality_score as DoneEvent["quality_score"])
        : null,
    graph_path: Array.isArray(payload.graph_path)
      ? payload.graph_path.filter((item): item is string => typeof item === "string")
      : [],
    model_used:
      typeof payload.model_used === "string"
        ? payload.model_used
        : typeof payload.model === "string"
          ? payload.model
          : undefined,
    provider_used:
      typeof payload.provider_used === "string"
        ? payload.provider_used
        : typeof payload.provider === "string"
          ? payload.provider
          : undefined,
    model: typeof payload.model === "string" ? payload.model : undefined,
    provider: typeof payload.provider === "string" ? payload.provider : undefined,
    cached: typeof payload.cached === "boolean" ? payload.cached : undefined,
    total_ms:
      Number.isFinite(Number(payload.total_ms))
        ? Number(payload.total_ms)
        : totalMsFromTiming,
    timing:
      timing && typeof timing === "object"
        ? {
            search_ms: Number((timing as Record<string, unknown>).search_ms ?? 0),
            graph_ms: Number((timing as Record<string, unknown>).graph_ms ?? 0),
            context_ms: Number((timing as Record<string, unknown>).context_ms ?? 0),
            total_ms: Number((timing as Record<string, unknown>).total_ms ?? 0),
          }
        : undefined,
  };
}

function toErrorEvent(payload: Record<string, unknown>): ErrorEvent {
  return {
    type: "error",
    message:
      typeof payload.message === "string"
        ? payload.message
        : typeof payload.error === "string"
          ? payload.error
          : "Streaming failed.",
  };
}

function toStepEvent(payload: Record<string, unknown>): StepEvent {
  return {
    type: "step",
    stage: typeof payload.stage === "string" ? payload.stage : "unknown",
    message: typeof payload.message === "string" ? payload.message : "",
  };
}

function normalizeEvent(eventName: string | null, payload: unknown): ChatEvent | null {
  if (!payload || typeof payload !== "object") {
    return null;
  }

  const record = payload as Record<string, unknown>;
  const sourceType = typeof record.type === "string" ? record.type : eventName;

  if (!sourceType) {
    return null;
  }

  switch (sourceType) {
    case "token":
      return toTokenEvent(record);
    case "delta":
      return toTokenEvent(record);
    case "sources":
      return toSourcesEvent(record);
    case "done":
      return toDoneEvent(record);
    case "error":
      return toErrorEvent(record);
    case "step":
      return toStepEvent(record);
    default:
      return null;
  }
}

function parseEventBlock(block: string): ChatEvent | null {
  const lines = block.split(/\r?\n/);
  let eventName: string | null = null;
  const dataLines: string[] = [];

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || line.startsWith(":")) {
      continue;
    }

    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
      continue;
    }

    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (!dataLines.length) {
    return null;
  }

  const data = dataLines.join("\n").trim();
  if (!data || data === "[DONE]") {
    return null;
  }

  try {
    const parsed = JSON.parse(data) as unknown;
    return normalizeEvent(eventName, parsed);
  } catch {
    if (process.env.NODE_ENV === "development") {
      // Keep malformed lines from crashing the whole stream.
      // eslint-disable-next-line no-console
      console.warn("[streaming] failed to parse SSE payload", data);
    }
    return null;
  }
}

function firstSeparator(input: string): { index: number; length: number } | null {
  const match = input.match(/\r?\n\r?\n/);
  if (!match || typeof match.index !== "number") {
    return null;
  }

  return {
    index: match.index,
    length: match[0].length,
  };
}

export async function* readSSEStream(
  stream: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const reader = stream.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      if (signal?.aborted) {
        break;
      }

      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });

      while (true) {
        const separator = firstSeparator(buffer);
        if (!separator) {
          break;
        }

        const block = buffer.slice(0, separator.index);
        buffer = buffer.slice(separator.index + separator.length);

        const event = parseEventBlock(block);
        if (event) {
          yield event;
        }
      }
    }

    if (buffer.trim()) {
      const trailingEvent = parseEventBlock(buffer);
      if (trailingEvent) {
        yield trailingEvent;
      }
    }
  } finally {
    try {
      await reader.cancel();
    } catch {
      // Ignore reader cancellation failures.
    }
    reader.releaseLock();
  }
}

export async function collectSSEStream(
  stream: ReadableStream<Uint8Array>,
): Promise<ChatEvent[]> {
  const events: ChatEvent[] = [];
  for await (const event of readSSEStream(stream)) {
    events.push(event);
  }
  return events;
}

function eventToPayload(event: ChatEvent): Record<string, unknown> {
  switch (event.type) {
    case "token":
      return { text: event.content };
    case "sources":
      return { sources: event.sources };
    case "done":
      return {
        session_id: event.session_id,
        quality_score: event.quality_score,
        graph_path: event.graph_path,
        model_used: event.model_used,
        provider_used: event.provider_used,
        cached: event.cached,
        total_ms: event.total_ms,
        timing: event.timing,
      };
    case "error":
      return { error: event.message };
    case "step":
      return { stage: event.stage, message: event.message };
    default:
      return {};
  }
}

export function makeTestStream(events: ChatEvent[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const data = events
    .map((event) => `event: ${event.type}\ndata: ${JSON.stringify(eventToPayload(event))}\n\n`)
    .join("");

  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoder.encode(data));
      controller.close();
    },
  });
}

export function makeChunkedTestStream(
  events: ChatEvent[],
  chunkSize = 10,
): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const payload = events
    .map((event) => `event: ${event.type}\ndata: ${JSON.stringify(eventToPayload(event))}\n\n`)
    .join("");
  const bytes = encoder.encode(payload);

  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (let index = 0; index < bytes.length; index += chunkSize) {
        controller.enqueue(bytes.slice(index, index + chunkSize));
      }
      controller.close();
    },
  });
}
