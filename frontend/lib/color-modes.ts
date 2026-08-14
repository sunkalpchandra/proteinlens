/** Color modes for the explorer map.
 *
 *  Pure functions: given the map points and a mode, produce a `colorOf`
 *  for the canvas plus the legend rows the overlay renders. Categorical
 *  modes ride on the fixed SERIES slots (top-N by frequency, rest fold to
 *  Other); magnitude modes use the single-hue blue ramp. Identity is never
 *  color-alone — the hover card names every point.
 */

import { blueRamp, categoricalScale, DIM_COLOR } from "@/lib/palette";
import type { MapPoint } from "@/lib/types";

export type ColorMode =
  | "family"
  | "organism"
  | "localization"
  | "ec"
  | "length"
  | "cluster"
  | "outlier"
  | "similarity"
  | "none";

export const COLOR_MODES: { id: ColorMode; label: string }[] = [
  { id: "family", label: "Family" },
  { id: "organism", label: "Organism" },
  { id: "localization", label: "Localization" },
  { id: "ec", label: "EC class" },
  { id: "length", label: "Length" },
  { id: "cluster", label: "Cluster" },
  { id: "outlier", label: "Outlier score" },
  { id: "similarity", label: "Similarity" },
  { id: "none", label: "None" },
];

export interface LegendEntry {
  label: string;
  color: string;
  count?: number;
}

export interface ColorScale {
  colorOf: (p: MapPoint) => string;
  legend: LegendEntry[] | null;
  kind: "categorical" | "sequential" | "none";
}

/** Neutral single-point color for the 'none' mode (mid blue-ramp step). */
const NONE_COLOR = "#6da7ec";

function categoricalMode(
  points: MapPoint[],
  get: (p: MapPoint) => string | null,
  topN = 7,
): ColorScale {
  const scale = categoricalScale(points.map(get), topN);
  const legend: LegendEntry[] = [...scale.legend];
  const missing = points.reduce((n, p) => n + (get(p) === null ? 1 : 0), 0);
  if (missing > 0) legend.push({ label: "n/a", color: DIM_COLOR, count: missing });
  return {
    colorOf: (p) => scale.colorOf(get(p)),
    legend,
    kind: "categorical",
  };
}

export function buildColorScale(
  points: MapPoint[],
  mode: ColorMode,
  neighborSims?: ReadonlyMap<string, number>,
): ColorScale {
  switch (mode) {
    case "family":
      return categoricalMode(points, (p) => p.family);
    case "organism":
      return categoricalMode(points, (p) => p.org);
    case "localization":
      return categoricalMode(points, (p) => p.loc);
    case "ec":
      return categoricalMode(points, (p) => p.ec);

    case "cluster": {
      const scale = categoricalScale(points.map((p) => String(p.cluster)), 8);
      return {
        colorOf: (p) => scale.colorOf(String(p.cluster)),
        legend: scale.legend.map(({ label, color, count }) => ({
          label: label === "Other" ? label : `cluster ${label}`,
          color,
          count,
        })),
        kind: "categorical",
      };
    }

    case "length": {
      let min = Infinity;
      let max = -Infinity;
      for (const p of points) {
        if (p.len < min) min = p.len;
        if (p.len > max) max = p.len;
      }
      if (!Number.isFinite(min)) {
        min = 0;
        max = 1;
      }
      const span = max - min || 1;
      return {
        colorOf: (p) => blueRamp((p.len - min) / span),
        legend: [
          { label: `${min} aa`, color: blueRamp(0) },
          { label: `${max} aa`, color: blueRamp(1) },
        ],
        kind: "sequential",
      };
    }

    case "outlier":
      return {
        colorOf: (p) => blueRamp(p.outlier),
        legend: [
          { label: "0.0", color: blueRamp(0) },
          { label: "1.0", color: blueRamp(1) },
        ],
        kind: "sequential",
      };

    case "similarity":
      return {
        colorOf: (p) => {
          const sim = neighborSims?.get(p.id);
          return sim === undefined ? DIM_COLOR : blueRamp(sim);
        },
        legend: [
          { label: "cos 0", color: blueRamp(0) },
          { label: "cos 1", color: blueRamp(1) },
        ],
        kind: "sequential",
      };

    case "none":
      return { colorOf: () => NONE_COLOR, legend: null, kind: "none" };
  }
}
