/** Compact legend overlay for the explorer map (bottom-left).
 *
 *  Categorical: swatch rows with label + count, capped and truncated so the
 *  panel never covers the viz. Sequential: a small horizontal ramp bar built
 *  from the blue-ramp steps with min/max endpoint labels. Text stays in ink
 *  tokens; only the swatches carry series color.
 */

import { blueRamp } from "@/lib/palette";
import type { LegendEntry } from "@/lib/color-modes";

export interface MapLegendProps {
  title: string;
  legend: LegendEntry[] | null;
  kind: "categorical" | "sequential" | "none";
  className?: string;
}

const MAX_ROWS = 9; // 7–8 category slots + Other / n-a

const RAMP_GRADIENT = `linear-gradient(to right, ${Array.from(
  { length: 13 },
  (_, i) => blueRamp(i / 12),
).join(", ")})`;

export function MapLegend({ title, legend, kind, className = "" }: MapLegendProps) {
  if (kind === "none" || !legend || legend.length === 0) return null;

  return (
    <div
      className={`absolute bottom-3 left-3 z-10 w-52 rounded border border-bd bg-surface/90 p-2.5 backdrop-blur ${className}`}
    >
      <div className="label-mono pb-1.5">{title}</div>
      {kind === "categorical" ? (
        <div className="space-y-1">
          {legend.slice(0, MAX_ROWS).map((row) => (
            <div key={row.label} className="flex items-center gap-2">
              <span
                className="h-2 w-2 shrink-0 rounded-sm"
                style={{ background: row.color }}
              />
              <span className="min-w-0 flex-1 truncate text-[11px] text-ink2" title={row.label}>
                {row.label}
              </span>
              {row.count !== undefined && (
                <span className="shrink-0 font-mono text-[10px] text-ink3 tabular">
                  {row.count}
                </span>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div>
          <div className="h-2 rounded-sm" style={{ background: RAMP_GRADIENT }} />
          <div className="flex justify-between pt-1 font-mono text-[10px] text-ink3 tabular">
            <span>{legend[0].label}</span>
            <span>{legend[legend.length - 1].label}</span>
          </div>
        </div>
      )}
    </div>
  );
}
