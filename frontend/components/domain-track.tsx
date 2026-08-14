"use client";

/** Horizontal domain-architecture track: curated DOMAIN features positioned
 *  proportionally along 1..length. Overlapping features fall back to a second
 *  row (greedy). Clicking a feature selects its region; the selected feature
 *  gets an accent outline. Sits above the region-search controls on the
 *  protein profile — deliberately quiet, no fill color beyond surface tones.
 */

import { useMemo } from "react";
import type { DomainFeature } from "@/lib/types";

export interface DomainTrackProps {
  length: number;
  domains: DomainFeature[];
  selected: { start: number; end: number } | null;
  onSelect: (region: { start: number; end: number }) => void;
  className?: string;
}

interface PlacedDomain extends DomainFeature {
  key: string;
  row: 0 | 1;
}

export function DomainTrack({
  length,
  domains,
  selected,
  onSelect,
  className = "",
}: DomainTrackProps) {
  // Greedy two-row layout: sweep left to right; a feature drops to the second
  // row when it overlaps the last feature placed on the first. If both rows
  // are occupied, it joins the row that frees up first.
  const placed = useMemo<PlacedDomain[]>(() => {
    const sorted = domains
      .map((d, i) => ({ ...d, key: `${d.name}-${d.start}-${d.end}-${i}` }))
      .sort((a, b) => a.start - b.start || a.end - b.end);
    const rowEnd: [number, number] = [0, 0];
    return sorted.map((d) => {
      let row: 0 | 1;
      if (d.start > rowEnd[0]) row = 0;
      else if (d.start > rowEnd[1]) row = 1;
      else row = rowEnd[0] <= rowEnd[1] ? 0 : 1;
      rowEnd[row] = Math.max(rowEnd[row], d.end);
      return { ...d, row };
    });
  }, [domains]);

  if (length <= 0) return null;

  const twoRows = placed.some((d) => d.row === 1);
  const pct = (v: number) => (v / length) * 100;

  return (
    <div className={className}>
      <div className="relative h-8">
        {placed.map((d) => {
          const isSel =
            selected !== null && selected.start === d.start && selected.end === d.end;
          return (
            <button
              key={d.key}
              type="button"
              title={`${d.name} (${d.start}–${d.end})`}
              onClick={() => onSelect({ start: d.start, end: d.end })}
              className={`absolute flex items-center overflow-hidden rounded-[3px] border border-bds bg-surface2 text-left transition-colors hover:bg-surface ${
                isSel ? "outline outline-1 outline-accent" : ""
              }`}
              style={{
                left: `${pct(d.start - 1)}%`,
                width: `${pct(d.end - d.start + 1)}%`,
                minWidth: 8,
                top: twoRows && d.row === 1 ? 17 : 1,
                height: twoRows ? 14 : 30,
              }}
            >
              <span className="block w-full truncate px-1 font-mono text-[10px] leading-none text-ink2">
                {d.name}
              </span>
            </button>
          );
        })}
      </div>
      <div className="mt-1 flex justify-between border-t border-bds pt-0.5 font-mono text-[9px] text-ink3 tabular">
        <span>1</span>
        <span>{length}</span>
      </div>
    </div>
  );
}
