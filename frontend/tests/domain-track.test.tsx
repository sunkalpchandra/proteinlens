import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DomainTrack } from "@/components/domain-track";
import type { DomainFeature } from "@/lib/types";

const domains: DomainFeature[] = [
  { name: "EF-hand 1", start: 10, end: 45 },
  { name: "EF-hand 2", start: 80, end: 115 },
  { name: "Overlapping", start: 30, end: 70 }, // overlaps EF-hand 1
];

describe("DomainTrack", () => {
  it("renders one element per domain", () => {
    render(
      <DomainTrack length={150} domains={domains} selected={null} onSelect={() => {}} />,
    );
    expect(screen.getByTitle(/EF-hand 1/)).toBeInTheDocument();
    expect(screen.getByTitle(/EF-hand 2/)).toBeInTheDocument();
    expect(screen.getByTitle(/Overlapping/)).toBeInTheDocument();
  });

  it("reports the domain's coordinates on click", () => {
    const onSelect = vi.fn();
    render(
      <DomainTrack length={150} domains={domains} selected={null} onSelect={onSelect} />,
    );
    fireEvent.click(screen.getByTitle(/EF-hand 2/));
    expect(onSelect).toHaveBeenCalledWith({ start: 80, end: 115 });
  });

  it("shows sequence end labels", () => {
    render(
      <DomainTrack length={150} domains={domains} selected={null} onSelect={() => {}} />,
    );
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("150")).toBeInTheDocument();
  });

  it("renders nothing dramatic with zero domains", () => {
    const { container } = render(
      <DomainTrack length={150} domains={[]} selected={null} onSelect={() => {}} />,
    );
    expect(container.firstChild).not.toBeNull();
  });
});
