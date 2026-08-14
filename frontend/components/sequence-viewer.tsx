"use client";

/** Residue-level sequence display.
 *
 *  Residues render as a wrapping grid of monospace cells colored by
 *  physicochemical category (see lib/palette.ts; the legend names every
 *  category, so identity never rides on hue alone). Cells are buttons:
 *  click selects a 1-based position. An optional intensity array (attention
 *  weights, per-residue Δ) shades a bar under each cell.
 */

import { useMemo } from "react";
import { AA_CATEGORY, AA_CATEGORY_COLOR, aaColor } from "@/lib/palette";

export interface SequenceViewerProps {
  sequence: string;
  selected?: number | null; // 1-based
  onSelect?: (position: number) => void;
  /** Optional per-residue intensity in [0, 1] drawn as an underbar. */
  intensity?: number[] | null;
  intensityLabel?: string;
  className?: string;
}

const RULER_EVERY = 10;

export function SequenceViewer({
  sequence,
  selected = null,
  onSelect,
  intensity = null,
  intensityLabel,
  className = "",
}: SequenceViewerProps) {
  const maxIntensity = useMemo(
    () => (intensity?.length ? Math.max(...intensity, 1e-9) : 1),
    [intensity],
  );

  return (
    <div className={className}>
      <div className="flex flex-wrap gap-x-4 gap-y-1 pb-2">
        {Object.entries(AA_CATEGORY_COLOR).map(([category, color]) => (
          <span key={category} className="flex items-center gap-1.5 font-mono text-[10px] text-ink3">
            <span className="inline-block h-2 w-2 rounded-sm" style={{ background: color }} />
            {category}
          </span>
        ))}
        {intensity && intensityLabel && (
          <span className="ml-auto font-mono text-[10px] text-ink3">
            underbar: {intensityLabel}
          </span>
        )}
      </div>
      <div className="flex flex-wrap leading-none">
        {sequence.split("").map((aa, i) => {
          const pos = i + 1;
          const isSelected = selected === pos;
          const barHeight = intensity
            ? Math.max(0.08, (intensity[i] ?? 0) / maxIntensity)
            : 0;
          return (
            <button
              key={pos}
              type="button"
              onClick={() => onSelect?.(pos)}
              title={`${aa}${pos} · ${AA_CATEGORY[aa] ?? "?"}`}
              className={`group relative flex h-7 w-[17px] flex-col items-center justify-start pt-0.5 font-mono text-[11px] transition-colors ${
                isSelected
                  ? "rounded-sm bg-surface2 outline outline-1 outline-white/60"
                  : "hover:rounded-sm hover:bg-surface2"
              } ${onSelect ? "cursor-pointer" : "cursor-default"}`}
            >
              <span style={{ color: aaColor(aa) }}>{aa}</span>
              {pos % RULER_EVERY === 0 ? (
                <span className="mt-0.5 text-[8px] text-ink3 tabular">{pos}</span>
              ) : (
                <span className="mt-0.5 text-[8px] text-transparent">·</span>
              )}
              {intensity && (
                <span
                  className="absolute bottom-0 left-[2px] right-[2px] rounded-t-[1px]"
                  style={{
                    height: `${Math.round(barHeight * 8)}px`,
                    background: "#3987e5",
                    opacity: 0.85,
                  }}
                />
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
