import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { GraphFilterPanel } from "@/components/Graph/GraphFilterPanel";
import type { GraphFilterState } from "@/types/api";

const defaultFilters: GraphFilterState = {
  nodeTypes: {
    function: true,
    class: true,
    module: true,
    file: true,
  },
  search: "",
  showIsolated: true,
  minDegree: 0,
  depth: 2,
};

function renderPanel(overrides?: Partial<ComponentProps<typeof GraphFilterPanel>>) {
  const props: ComponentProps<typeof GraphFilterPanel> = {
    filters: defaultFilters,
    typeCounts: {
      function: 5,
      class: 2,
      module: 1,
      file: 3,
    },
    onToggleNodeType: vi.fn(),
    onSearchChange: vi.fn(),
    onShowIsolatedChange: vi.fn(),
    onDepthChange: vi.fn(),
    onMinDegreeChange: vi.fn(),
    onReset: vi.fn(),
    ...overrides,
  };

  return {
    ...render(<GraphFilterPanel {...props} />),
    props,
  };
}

afterEach(() => {
  vi.useRealTimers();
});

describe("GraphFilterPanel", () => {
  it("renders node type controls with counts", () => {
    renderPanel();
    const checkboxes = screen.getAllByRole("checkbox");

    expect(screen.getByText("Node types")).toBeInTheDocument();
    expect(checkboxes[0]).toBeChecked();
    expect(checkboxes[1]).toBeChecked();
    expect(screen.getByText("5")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
  });

  it("updates checkbox state through callback", async () => {
    const user = userEvent.setup();
    const onToggleNodeType = vi.fn();
    renderPanel({ onToggleNodeType });

    await user.click(screen.getByText("Function"));

    expect(onToggleNodeType).toHaveBeenCalledWith("function", false);
  });

  it("debounces search input changes", () => {
    vi.useFakeTimers();
    const onSearchChange = vi.fn();

    renderPanel({ onSearchChange });

    vi.advanceTimersByTime(300);
    onSearchChange.mockClear();

    fireEvent.change(screen.getByPlaceholderText("Search node name..."), {
      target: { value: "auth" },
    });

    expect(onSearchChange).not.toHaveBeenCalled();

    vi.advanceTimersByTime(299);
    expect(onSearchChange).not.toHaveBeenCalled();

    vi.advanceTimersByTime(1);
    expect(onSearchChange).toHaveBeenCalledWith("auth");
  });

  it("updates depth and min degree sliders", () => {
    const onDepthChange = vi.fn();
    const onMinDegreeChange = vi.fn();
    renderPanel({ onDepthChange, onMinDegreeChange });

    const sliders = screen.getAllByRole("slider");
    fireEvent.change(sliders[0], { target: { value: "4" } });
    fireEvent.change(sliders[1], { target: { value: "6" } });

    expect(onDepthChange).toHaveBeenCalledWith(4);
    expect(onMinDegreeChange).toHaveBeenCalledWith(6);
  });

  it("resets filters", async () => {
    const user = userEvent.setup();
    const onReset = vi.fn();
    renderPanel({ onReset });

    await user.click(screen.getByRole("button", { name: "Reset Filters" }));

    expect(onReset).toHaveBeenCalledTimes(1);
  });
});
