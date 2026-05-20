"use client";

import { useEffect, useState } from "react";

import { cn } from "@/lib/utils";
import type { FileTreeNode, RepositoryFile } from "@/types/api";

function TreeNode({
  node,
  depth,
  selectedPath,
  onFileClick,
  onFileContextMenu,
}: {
  node: FileTreeNode;
  depth: number;
  selectedPath?: string;
  onFileClick?: (file: RepositoryFile) => void;
  onFileContextMenu?: (event: React.MouseEvent<HTMLButtonElement>, file: RepositoryFile) => void;
}) {
  const [open, setOpen] = useState(depth < 1);
  const isSelected = selectedPath === node.path;

  if (node.type === "dir") {
    return (
      <div className="space-y-1">
        <button
          className="flex w-full items-center gap-2 rounded-xl px-2 py-1.5 text-left text-sm text-slate-200 transition hover:bg-white/6"
          onClick={() => setOpen((current) => !current)}
          style={{ paddingLeft: `${depth * 14 + 8}px` }}
          type="button"
        >
          <span className="text-xs text-surface-muted">{open ? "▾" : "▸"}</span>
          <span className="font-medium">{node.name}</span>
        </button>
        {open ? (
          <div className="space-y-1">
            {node.children?.map((child) => (
              <TreeNode
                key={child.path}
                depth={depth + 1}
                node={child}
                onFileClick={onFileClick}
                onFileContextMenu={onFileContextMenu}
                selectedPath={selectedPath}
              />
            ))}
          </div>
        ) : null}
      </div>
    );
  }

  return (
    <button
      className={cn(
        "flex w-full items-center gap-2 rounded-xl px-2 py-1.5 text-left text-sm transition",
        isSelected
          ? "bg-brand-500/15 text-brand-100"
          : "text-slate-300 hover:bg-white/6 hover:text-white",
      )}
      onClick={() => {
        if (node.file && onFileClick) {
          onFileClick(node.file);
        }
      }}
      onContextMenu={(event) => {
        if (node.file && onFileContextMenu) {
          event.preventDefault();
          onFileContextMenu(event, node.file);
        }
      }}
      style={{ paddingLeft: `${depth * 14 + 20}px` }}
      type="button"
    >
      <span className="text-xs text-surface-muted">•</span>
      <span>{node.name}</span>
    </button>
  );
}

export function FileTree({
  nodes,
  selectedPath,
  onFileClick,
  onFileGraphClick,
  emptyLabel = "No files indexed yet.",
}: {
  nodes: FileTreeNode[];
  selectedPath?: string;
  onFileClick?: (file: RepositoryFile) => void;
  onFileGraphClick?: (file: RepositoryFile) => void;
  emptyLabel?: string;
}) {
  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
    file: RepositoryFile;
  } | null>(null);

  useEffect(() => {
    const close = () => setContextMenu(null);
    window.addEventListener("click", close);
    window.addEventListener("scroll", close, true);
    window.addEventListener("resize", close);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("resize", close);
    };
  }, []);

  if (!nodes.length) {
    return (
      <p className="rounded-2xl border border-dashed border-surface-border p-4 text-sm text-surface-muted">
        {emptyLabel}
      </p>
    );
  }

  return (
    <div className="relative space-y-1">
      {nodes.map((node) => (
        <TreeNode
          key={node.path}
          depth={0}
          node={node}
          onFileClick={onFileClick}
          onFileContextMenu={(event, file) => {
            if (!onFileGraphClick) {
              return;
            }
            setContextMenu({ x: event.clientX, y: event.clientY, file });
          }}
          selectedPath={selectedPath}
        />
      ))}

      {contextMenu && onFileGraphClick ? (
        <div
          className="fixed z-50 min-w-[170px] rounded-lg border border-surface-border bg-surface px-1 py-1 shadow-xl"
          style={{ left: contextMenu.x, top: contextMenu.y }}
        >
          <button
            className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-xs text-slate-200 transition hover:bg-surface-hover"
            onClick={() => {
              onFileGraphClick(contextMenu.file);
              setContextMenu(null);
            }}
            type="button"
          >
            <span aria-hidden="true">⬡</span>
            Show in graph
          </button>
        </div>
      ) : null}
    </div>
  );
}
