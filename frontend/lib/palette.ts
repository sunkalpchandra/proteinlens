/** Chart palette: validated dark-mode categorical slots (fixed order, never
 *  cycled), a single-hue blue ramp for magnitudes, and amino-acid category
 *  colors. Category values beyond the top seven fold into "Other" (gray) —
 *  identity is then recoverable from the hover card, never from hue alone. */

export const SERIES = [
  "#3987e5", // 1 blue
  "#d95926", // 2 orange
  "#199e70", // 3 aqua
  "#c98500", // 4 yellow
  "#d55181", // 5 magenta
  "#008300", // 6 green
  "#9085e9", // 7 violet
  "#e66767", // 8 red
] as const;

export const OTHER_COLOR = "#5b5a55";
export const DIM_COLOR = "#33332f";

/** Blue sequential ramp, dark→light: near-zero recedes into the dark surface. */
const BLUE_RAMP = [
  "#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf",
  "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef",
  "#9ec5f4", "#b7d3f6", "#cde2fb",
];

export function blueRamp(t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  return BLUE_RAMP[Math.round(clamped * (BLUE_RAMP.length - 1))];
}

export interface CategoricalScale {
  colorOf: (value: string | null) => string;
  legend: { label: string; color: string; count: number }[];
}

/** Top-N categories by frequency take the fixed slots; the rest fold to Other. */
export function categoricalScale(
  values: (string | null)[],
  topN = 7,
): CategoricalScale {
  const counts = new Map<string, number>();
  for (const v of values) {
    if (v !== null) counts.set(v, (counts.get(v) ?? 0) + 1);
  }
  const top = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, topN);
  const slot = new Map(top.map(([label], i) => [label, SERIES[i]]));
  const otherCount = values.filter((v) => v !== null && !slot.has(v)).length;

  const legend: CategoricalScale["legend"] = top.map(([label, count], i) => ({
    label,
    color: SERIES[i],
    count,
  }));
  if (otherCount > 0) legend.push({ label: "Other", color: OTHER_COLOR, count: otherCount });

  return {
    colorOf: (value) =>
      value === null ? DIM_COLOR : (slot.get(value) ?? OTHER_COLOR),
    legend,
  };
}

/** Amino-acid physicochemical categories (mirrors ml/sequence.py). */
export const AA_CATEGORY: Record<string, string> = {
  A: "hydrophobic", V: "hydrophobic", L: "hydrophobic", I: "hydrophobic", M: "hydrophobic",
  F: "aromatic", W: "aromatic", Y: "aromatic",
  S: "polar", T: "polar", N: "polar", Q: "polar",
  D: "negative", E: "negative",
  K: "positive", R: "positive", H: "positive",
  C: "special", G: "special", P: "special",
};

/** Fixed category→slot mapping (documented order; color follows the entity). */
export const AA_CATEGORY_COLOR: Record<string, string> = {
  hydrophobic: SERIES[0],
  aromatic: SERIES[1],
  polar: SERIES[2],
  positive: SERIES[3],
  negative: SERIES[4],
  special: SERIES[5],
};

export const AA_LIST = "ACDEFGHIKLMNPQRSTVWY".split("");

export function aaColor(aa: string): string {
  return AA_CATEGORY_COLOR[AA_CATEGORY[aa] ?? "special"] ?? OTHER_COLOR;
}
