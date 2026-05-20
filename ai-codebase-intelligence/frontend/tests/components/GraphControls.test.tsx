import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { RefObject } from "react";
import { afterAll, beforeEach, describe, expect, it, vi } from "vitest";

import { GraphControls } from "@/components/Graph/GraphControls";
import type { ForceGraphHandle } from "@/components/Graph/ForceGraph";

const originalCreateObjectURL = URL.createObjectURL;
const originalRevokeObjectURL = URL.revokeObjectURL;

function makeGraphHandle(overrides?: Partial<ForceGraphHandle>): ForceGraphHandle {
  return {
    zoomIn: vi.fn(),
    zoomOut: vi.fn(),
    resetView: vi.fn(),
    fitToScreen: vi.fn(),
    exportSvgString: vi.fn(() => null),
    pauseSimulation: vi.fn(),
    resumeSimulation: vi.fn(),
    isPaused: vi.fn(() => false),
    ...overrides,
  };
}

function renderControls(options?: {
  simulationPaused?: boolean;
  handle?: ForceGraphHandle;
  onToggleSimulation?: () => void;
}) {
  const handle = options?.handle ?? makeGraphHandle();
  const graphRef = { current: handle } as RefObject<ForceGraphHandle>;
  const onToggleSimulation = options?.onToggleSimulation ?? vi.fn();

  return {
    handle,
    onToggleSimulation,
    ...render(
      <GraphControls
        repoId="repo-1"
        nodes={[
          { id: "a", _type: "function", display_name: "a" },
          { id: "b", _type: "class", display_name: "b" },
        ]}
        edges={[{ source: "a", target: "b", type: "CALLS" }]}
        graphRef={graphRef}
        simulationPaused={options?.simulationPaused ?? false}
        onToggleSimulation={onToggleSimulation}
      />,
    ),
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  Object.defineProperty(URL, "createObjectURL", {
    writable: true,
    value: vi.fn(() => "blob:mock"),
  });
  Object.defineProperty(URL, "revokeObjectURL", {
    writable: true,
    value: vi.fn(),
  });
});

afterAll(() => {
  Object.defineProperty(URL, "createObjectURL", {
    writable: true,
    value: originalCreateObjectURL,
  });
  Object.defineProperty(URL, "revokeObjectURL", {
    writable: true,
    value: originalRevokeObjectURL,
  });
});

describe("GraphControls", () => {
  it("renders zoom/view controls and calls graph handle methods", async () => {
    const user = userEvent.setup();
    const { handle } = renderControls();

    await user.click(screen.getByRole("button", { name: "Zoom +" }));
    await user.click(screen.getByRole("button", { name: "Zoom -" }));
    await user.click(screen.getByRole("button", { name: "Reset" }));
    await user.click(screen.getByRole("button", { name: "Fit" }));

    expect(handle.zoomIn).toHaveBeenCalledTimes(1);
    expect(handle.zoomOut).toHaveBeenCalledTimes(1);
    expect(handle.resetView).toHaveBeenCalledTimes(1);
    expect(handle.fitToScreen).toHaveBeenCalledTimes(1);
  });

  it("supports pause and resume button states", async () => {
    const user = userEvent.setup();
    const onToggleSimulation = vi.fn();

    const { rerender } = renderControls({ simulationPaused: false, onToggleSimulation });

    await user.click(screen.getByRole("button", { name: "Pause" }));
    expect(onToggleSimulation).toHaveBeenCalledTimes(1);

    rerender(
      <GraphControls
        repoId="repo-1"
        nodes={[{ id: "a", _type: "function", display_name: "a" }]}
        edges={[]}
        graphRef={{ current: makeGraphHandle() } as RefObject<ForceGraphHandle>}
        simulationPaused
        onToggleSimulation={onToggleSimulation}
      />,
    );

    expect(screen.getByRole("button", { name: "Resume" })).toBeInTheDocument();
  });

  it("exports graph JSON", async () => {
    const user = userEvent.setup();
    const anchorClickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    renderControls();

    await user.click(screen.getByRole("button", { name: "Export JSON" }));

    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    expect(anchorClickSpy).toHaveBeenCalledTimes(1);
    expect(URL.revokeObjectURL).toHaveBeenCalledTimes(1);

    anchorClickSpy.mockRestore();
  });

  it("exports PNG from SVG graph markup", async () => {
    const user = userEvent.setup();
    const anchorClickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    const handle = makeGraphHandle({
      exportSvgString: vi.fn(() => "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400\" height=\"300\"></svg>"),
    });

    const OriginalImage = global.Image;
    class MockImage {
      onload: ((event: Event) => void) | null = null;
      onerror: ((event: Event) => void) | null = null;
      width = 400;
      height = 300;

      set src(_value: string) {
        if (this.onload) {
          this.onload(new Event("load"));
        }
      }
    }

    const fillRect = vi.fn();
    const drawImage = vi.fn();
    const getContextSpy = vi
      .spyOn(HTMLCanvasElement.prototype, "getContext")
      .mockReturnValue({ fillRect, drawImage } as unknown as CanvasRenderingContext2D);
    const toBlobSpy = vi
      .spyOn(HTMLCanvasElement.prototype, "toBlob")
      .mockImplementation(function toBlob(callback) {
        callback(new Blob(["png"], { type: "image/png" }));
      });

    Object.defineProperty(global, "Image", {
      writable: true,
      value: MockImage,
    });

    renderControls({ handle });

    await user.click(screen.getByRole("button", { name: "Export PNG" }));

    await waitFor(() => {
      expect(handle.exportSvgString).toHaveBeenCalledTimes(1);
      expect(URL.createObjectURL).toHaveBeenCalledTimes(2);
      expect(anchorClickSpy).toHaveBeenCalledTimes(1);
    });

    getContextSpy.mockRestore();
    toBlobSpy.mockRestore();
    anchorClickSpy.mockRestore();
    Object.defineProperty(global, "Image", {
      writable: true,
      value: OriginalImage,
    });
  });
});
