import { describe, expect, it } from "vitest";
import {
  AA_CATEGORY,
  AA_LIST,
  OTHER_COLOR,
  SERIES,
  aaColor,
  blueRamp,
  categoricalScale,
} from "@/lib/palette";

describe("categoricalScale", () => {
  const values = [
    ...Array(5).fill("Globin family"),
    ...Array(3).fill("Kinase family"),
    ...Array(2).fill("Lectin family"),
    "Rare family",
    null,
  ];

  it("assigns fixed slots by frequency order", () => {
    const scale = categoricalScale(values, 2);
    expect(scale.colorOf("Globin family")).toBe(SERIES[0]);
    expect(scale.colorOf("Kinase family")).toBe(SERIES[1]);
  });

  it("folds beyond-topN values into Other", () => {
    const scale = categoricalScale(values, 2);
    expect(scale.colorOf("Lectin family")).toBe(OTHER_COLOR);
    expect(scale.colorOf("Rare family")).toBe(OTHER_COLOR);
    const other = scale.legend.find((l) => l.label === "Other");
    expect(other?.count).toBe(3);
  });

  it("never colors null like a category", () => {
    const scale = categoricalScale(values, 2);
    expect(scale.colorOf(null)).not.toBe(SERIES[0]);
    expect(scale.colorOf(null)).not.toBe(OTHER_COLOR);
  });

  it("legend is ordered by count and capped", () => {
    const scale = categoricalScale(values, 7);
    expect(scale.legend[0].label).toBe("Globin family");
    expect(scale.legend[0].count).toBe(5);
  });
});

describe("blueRamp", () => {
  it("clamps out-of-range inputs", () => {
    expect(blueRamp(-1)).toBe(blueRamp(0));
    expect(blueRamp(2)).toBe(blueRamp(1));
  });

  it("is monotone dark→light in hex lightness endpoints", () => {
    expect(blueRamp(0)).toBe("#0d366b");
    expect(blueRamp(1)).toBe("#cde2fb");
  });
});

describe("amino-acid categories", () => {
  it("covers all 20 canonical residues", () => {
    for (const aa of AA_LIST) {
      expect(AA_CATEGORY[aa], `category for ${aa}`).toBeDefined();
      expect(aaColor(aa)).toMatch(/^#/);
    }
  });
});
