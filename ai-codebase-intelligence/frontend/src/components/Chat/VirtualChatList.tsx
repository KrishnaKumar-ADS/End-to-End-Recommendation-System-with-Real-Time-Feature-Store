"use client";

import {
  memo,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { VariableSizeList as List, type ListChildComponentProps } from "react-window";

import { ChatMessage } from "@/components/Chat/ChatMessage";
import type { ChatMessage as ChatMessageType } from "@/types/api";

interface VirtualChatListProps {
  messages: ChatMessageType[];
  repoId: string;
  isStreaming: boolean;
}

interface RowData {
  messages: ChatMessageType[];
  repoId: string;
  setSize: (index: number, size: number) => void;
  isStreaming: boolean;
}

const MessageRow = memo(function MessageRow({ index, style, data }: ListChildComponentProps<RowData>) {
  const rowRef = useRef<HTMLDivElement | null>(null);
  const message = data.messages[index];

  useLayoutEffect(() => {
    const node = rowRef.current;
    if (!node) {
      return;
    }

    const nextSize = node.getBoundingClientRect().height + 12;
    data.setSize(index, nextSize);
  }, [data, index, message, data.isStreaming]);

  if (!message) {
    return null;
  }

  return (
    <div style={style}>
      <div ref={rowRef}>
        <ChatMessage message={message} repoId={data.repoId} />
      </div>
    </div>
  );
});

export function VirtualChatList({ messages, repoId, isStreaming }: VirtualChatListProps) {
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const outerRef = useRef<HTMLDivElement | null>(null);
  const listRef = useRef<List | null>(null);
  const sizeMapRef = useRef<Record<number, number>>({});
  const userScrolledUpRef = useRef(false);

  const [showJumpToLatest, setShowJumpToLatest] = useState(false);
  const [height, setHeight] = useState(480);

  useEffect(() => {
    const node = wrapperRef.current;
    if (!node) {
      return;
    }

    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (rect) {
        setHeight(Math.max(240, Math.floor(rect.height)));
      }
    });

    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const setSize = (index: number, size: number) => {
    if (sizeMapRef.current[index] === size) {
      return;
    }
    sizeMapRef.current[index] = size;
    listRef.current?.resetAfterIndex(index);
  };

  const getSize = (index: number) => sizeMapRef.current[index] ?? 132;

  const rowData = useMemo<RowData>(
    () => ({
      messages,
      repoId,
      setSize,
      isStreaming,
    }),
    [isStreaming, messages, repoId],
  );

  useEffect(() => {
    const outer = outerRef.current;
    if (!outer) {
      return;
    }

    const onScroll = () => {
      const distanceFromBottom = outer.scrollHeight - outer.scrollTop - outer.clientHeight;
      const nearBottom = distanceFromBottom < 100;
      userScrolledUpRef.current = !nearBottom;
      setShowJumpToLatest(!nearBottom && messages.length > 0);
    };

    outer.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => outer.removeEventListener("scroll", onScroll);
  }, [messages.length]);

  useEffect(() => {
    if (!messages.length) {
      return;
    }

    const lastIndex = messages.length - 1;
    if (isStreaming) {
      listRef.current?.resetAfterIndex(lastIndex);
    }

    const shouldAutoScroll = isStreaming || !userScrolledUpRef.current;
    if (shouldAutoScroll) {
      listRef.current?.scrollToItem(lastIndex, "end");
      setShowJumpToLatest(false);
    }
  }, [isStreaming, messages]);

  return (
    <div className="relative min-h-0 flex-1" ref={wrapperRef}>
      <List
        className="px-4 py-6"
        height={height}
        itemCount={messages.length}
        itemData={rowData}
        itemSize={getSize}
        overscanCount={6}
        ref={listRef}
        width="100%"
        outerRef={outerRef}
      >
        {MessageRow}
      </List>

      {showJumpToLatest ? (
        <button
          className="absolute bottom-4 right-4 rounded-full border border-brand-500/40 bg-brand-500/10 px-3 py-1.5 text-xs text-brand-100 shadow-lg transition hover:bg-brand-500/20"
          onClick={() => {
            const lastIndex = messages.length - 1;
            if (lastIndex >= 0) {
              listRef.current?.scrollToItem(lastIndex, "end");
            }
          }}
          type="button"
        >
          Jump to latest ↓
        </button>
      ) : null}
    </div>
  );
}
