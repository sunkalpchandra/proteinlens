import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SequenceViewer } from "@/components/sequence-viewer";

describe("SequenceViewer", () => {
  it("renders one selectable cell per residue", () => {
    render(<SequenceViewer sequence="MKTVH" />);
    const cells = screen.getAllByRole("button");
    expect(cells).toHaveLength(5);
    expect(cells[0]).toHaveTextContent("M");
    expect(cells[4]).toHaveTextContent("H");
  });

  it("reports 1-based positions on click", () => {
    const onSelect = vi.fn();
    render(<SequenceViewer sequence="MKTVH" onSelect={onSelect} />);
    fireEvent.click(screen.getAllByRole("button")[2]);
    expect(onSelect).toHaveBeenCalledWith(3);
  });

  it("marks the selected residue", () => {
    render(<SequenceViewer sequence="MKTVH" selected={2} />);
    const cell = screen.getAllByRole("button")[1];
    expect(cell.className).toContain("outline");
  });

  it("titles cells with position and category", () => {
    render(<SequenceViewer sequence="MK" />);
    expect(screen.getAllByRole("button")[1]).toHaveAttribute("title", "K2 · positive");
  });
});

describe("SequenceViewer keyboard navigation", () => {
  it("arrow keys move the selection with clamping", () => {
    const onSelect = vi.fn();
    render(<SequenceViewer sequence="MKTVH" selected={5} onSelect={onSelect} />);
    const listbox = screen.getByRole("listbox");
    fireEvent.keyDown(listbox, { key: "ArrowRight" });
    expect(onSelect).toHaveBeenLastCalledWith(5); // clamped at the end
    fireEvent.keyDown(listbox, { key: "ArrowLeft" });
    expect(onSelect).toHaveBeenLastCalledWith(4);
    fireEvent.keyDown(listbox, { key: "ArrowUp" });
    expect(onSelect).toHaveBeenLastCalledWith(1); // -10 clamps to 1
  });

  it("read-only viewers are not focusable", () => {
    render(<SequenceViewer sequence="MKTVH" />);
    expect(screen.queryByRole("listbox")).toBeNull();
  });
});
