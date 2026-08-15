import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const findProteins = vi.fn();
vi.mock("@/lib/data", () => ({
  findProteins: (...args: unknown[]) => findProteins(...args),
}));

import { ProteinFinder } from "@/components/protein-finder";

const HIT = {
  accession: "P68871",
  name: "Hemoglobin subunit beta",
  gene: "HBB",
  organism: "H. sapiens",
  length: 147,
  family: "Globin family",
  pfam: "PF00042",
  ec_class: null,
  localization: "Cytoplasm",
};

describe("ProteinFinder", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    findProteins.mockReset();
    findProteins.mockResolvedValue([HIT]);
  });

  it("debounces: no search until the delay elapses", async () => {
    render(<ProteinFinder onPick={() => {}} />);
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "hemo" } });
    expect(findProteins).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(300);
    expect(findProteins).toHaveBeenCalledWith("hemo", 12);
  });

  it("picking a hit reports the full summary", async () => {
    vi.useRealTimers();
    const onPick = vi.fn();
    render(<ProteinFinder onPick={onPick} />);
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "hemo" } });
    await waitFor(() => expect(screen.getByText(/Hemoglobin subunit beta/)).toBeInTheDocument());
    fireEvent.click(screen.getByText(/Hemoglobin subunit beta/));
    expect(onPick).toHaveBeenCalledWith(HIT);
  });

  it("Enter selects the first hit", async () => {
    vi.useRealTimers();
    const onPick = vi.fn();
    render(<ProteinFinder onPick={onPick} />);
    const input = screen.getByRole("textbox");
    fireEvent.change(input, { target: { value: "hemo" } });
    await waitFor(() => expect(screen.getByText(/Hemoglobin/)).toBeInTheDocument());
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onPick).toHaveBeenCalledWith(HIT);
  });

  it("empty query never searches", async () => {
    render(<ProteinFinder onPick={() => {}} />);
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "   " } });
    await vi.advanceTimersByTimeAsync(500);
    expect(findProteins).not.toHaveBeenCalled();
  });
});
