"use client";

import { useState } from "react";

import type { ForceGraphHandle } from "@/components/Graph/ForceGraph";
import { Button } from "@/components/ui/Button";
import type { GraphEdge, GraphNode } from "@/types/api";

function downloadBlob(blob: Blob, filename: string) {
	const url = URL.createObjectURL(blob);
	const anchor = document.createElement("a");
	anchor.href = url;
	anchor.download = filename;
	document.body.appendChild(anchor);
	anchor.click();
	anchor.remove();
	URL.revokeObjectURL(url);
}

interface GraphControlsProps {
	repoId: string;
	nodes: GraphNode[];
	edges: GraphEdge[];
	graphRef: React.RefObject<ForceGraphHandle>;
	simulationPaused: boolean;
	onToggleSimulation: () => void;
}

export function GraphControls({
	repoId,
	nodes,
	edges,
	graphRef,
	simulationPaused,
	onToggleSimulation,
}: GraphControlsProps) {
	const [isExporting, setIsExporting] = useState(false);

	const exportJson = () => {
		const payload = {
			repo_id: repoId,
			node_count: nodes.length,
			edge_count: edges.length,
			nodes,
			edges,
			exported_at: new Date().toISOString(),
		};

		downloadBlob(
			new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }),
			`${repoId}-graph.json`,
		);
	};

	const exportPng = async () => {
		const svgString = graphRef.current?.exportSvgString();
		if (!svgString) {
			return;
		}

		setIsExporting(true);
		try {
			const svgBlob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
			const svgUrl = URL.createObjectURL(svgBlob);
			const img = new Image();

			await new Promise<void>((resolve, reject) => {
				img.onload = () => resolve();
				img.onerror = () => reject(new Error("Failed to render SVG for PNG export."));
				img.src = svgUrl;
			});

			const canvas = document.createElement("canvas");
			canvas.width = Math.max(1200, img.width || 1200);
			canvas.height = Math.max(800, img.height || 800);
			const context = canvas.getContext("2d");
			if (!context) {
				throw new Error("Canvas context unavailable.");
			}

			context.fillStyle = "#0b1220";
			context.fillRect(0, 0, canvas.width, canvas.height);
			context.drawImage(img, 0, 0, canvas.width, canvas.height);

			URL.revokeObjectURL(svgUrl);

			const pngBlob = await new Promise<Blob | null>((resolve) => {
				canvas.toBlob((blob) => resolve(blob), "image/png");
			});

			if (pngBlob) {
				downloadBlob(pngBlob, `${repoId}-graph.png`);
			}
		} finally {
			setIsExporting(false);
		}
	};

	return (
		<div className="flex flex-wrap items-center gap-2">
			<Button onClick={() => graphRef.current?.zoomIn()} size="sm" variant="secondary">
				Zoom +
			</Button>
			<Button onClick={() => graphRef.current?.zoomOut()} size="sm" variant="secondary">
				Zoom -
			</Button>
			<Button onClick={() => graphRef.current?.resetView()} size="sm" variant="secondary">
				Reset
			</Button>
			<Button onClick={() => graphRef.current?.fitToScreen()} size="sm" variant="secondary">
				Fit
			</Button>
			<Button onClick={onToggleSimulation} size="sm" variant="ghost">
				{simulationPaused ? "Resume" : "Pause"}
			</Button>
			<Button disabled={isExporting} onClick={() => void exportPng()} size="sm" variant="secondary">
				Export PNG
			</Button>
			<Button onClick={exportJson} size="sm" variant="secondary">
				Export JSON
			</Button>
		</div>
	);
}
