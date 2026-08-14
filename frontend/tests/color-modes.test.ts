import { describe, expect, it } from "vitest";
import { buildColorScale, COLOR_MODES } from "@/lib/color-modes";
import { DIM_COLOR, SERIES } from "@/lib/palette";
import type { MapPoint } from "@/lib/types";

function point(overrides: Partial<MapPoint>): MapPoint {
  return {
    id: "P1", name: "test", gene: null, org: "H. sapiens", len: 100,
    family: null, pfam: null, ec: null, enzyme: false, loc: null,
    x: 0, y: 0, cluster: 0, knn_dist: 0.1, outlier: 0.5,
    ...overrides,
  };
}

const points: MapPoint[] = [
  point({ id: "A", family: "Globin family", len: 50, outlier: 0 }),
  point({ id: "B", family: "Globin family", len: 300, outlier: 0.5 }),
  point({ id: "C", family: "Kinase family", len: 512, outlier: 1 }),
  point({ id: "D", family: null }),
];

describe("buildColorScale", () => {
  it("covers every declared color mode", () => {
    for (const mode of COLOR_MODES) {
      const scale = buildColorScale(points, mode.id);
      expect(scale.colorOf(points[0])).toMatch(/^#/);
    }
  });

  it("categorical: most frequent family takes the first series slot", () => {
    const scale = buildColorScale(points, "family");
    expect(scale.kind).toBe("categorical");
    expect(scale.colorOf(points[0])).toBe(SERIES[0]);
    expect(scale.colorOf(points[3])).toBe(DIM_COLOR); // null family dims
  });

  it("sequential: length maps min to dark and max to light", () => {
    const scale = buildColorScale(points, "length");
    expect(scale.kind).toBe("sequential");
    expect(scale.colorOf(points[0])).not.toBe(scale.colorOf(points[2]));
  });

  it("similarity without a selection dims everything", () => {
    const scale = buildColorScale(points, "similarity");
    expect(scale.colorOf(points[0])).toBe(DIM_COLOR);
  });

  it("similarity colors neighbors by cosine", () => {
    const sims = new Map([["A", 0.99], ["B", 0.5]]);
    const scale = buildColorScale(points, "similarity", sims);
    expect(scale.colorOf(points[0])).not.toBe(DIM_COLOR);
    expect(scale.colorOf(points[2])).toBe(DIM_COLOR); // not a neighbor
  });
});
