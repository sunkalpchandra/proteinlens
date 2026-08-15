"use client";

/** 19-substitution landscape for one position.
 *
 *  A single-row heat strip over the 20 canonical amino acids: each cell is
 *  colored by blueRamp of the normalized perturbation for the chosen metric
 *  (cosine similarity is inverted to 1 − cos, since a lower cosine means a
 *  bigger representational move). The wild-type cell carries a dot, no value.
 *  Below the strip, the 19 substitutions rank descending by perturbation as
 *  thin horizontal bars. The metric selector lives in the parent — this
 *  component only consumes `metric`.
 */

import { useMemo, useRef, useState } from "react";
import { AA_LIST, blueRamp } from "@/lib/palette";
import type { Landscape, LandscapeEffect } from "@/lib/types";

export type LandscapeMetric = "displacement" | "cosine_similarity" | "local_delta" | "llr";

export const METRIC_LABEL: Record<LandscapeMetric, string> = {
  displacement: "displacement ‖Δz‖",
  cosine_similarity: "cosine similarity",
  local_delta: "local Δ (±8)",
  llr: "LM log-likelihood ratio",
};

export interface MutationHeatmapProps {
  landscape: Landscape;
  metric: LandscapeMetric;
  onPick?: (mutation: string) => void;
  picked?: string | null;
}

/** Magnitude of perturbation under the chosen metric (always "bigger = more"). */
function perturbationOf(e: LandscapeEffect, metric: LandscapeMetric): number {
  if (metric === "cosine_similarity") return 1 - e.cosine_similarity;
  if (metric === "llr") return -(e.llr ?? 0); // more negative LLR = more disfavored
  return e[metric];
}

/** Raw metric value for display (cosine shows the actual cosine). */
function displayOf(e: LandscapeEffect, metric: LandscapeMetric): number {
  if (metric === "llr") return e.llr ?? 0;
  return e[metric];
}

/** ≤ 5-character number for the 9px value row. */
function fmt(v: number): string {
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

interface HoverState {
  effect: LandscapeEffect;
  left: number;
  top: number;
}

const TOOLTIP_HALF = 92; // half of the w-44 tooltip, for edge clamping

export function MutationHeatmap({
  landscape,
  metric,
  onPick,
  picked = null,
}: MutationHeatmapProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<HoverState | null>(null);

  const bySub = useMemo(() => {
    const m = new Map<string, LandscapeEffect>();
    for (const e of landscape.effects) m.set(e.mutation.slice(-1), e);
    return m;
  }, [landscape]);

  const { min, max, span } = useMemo(() => {
    const values = landscape.effects.map((e) => perturbationOf(e, metric));
    if (!values.length) return { min: 0, max: 1, span: 1 };
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    return { min: lo, max: hi, span: hi - lo || 1 };
  }, [landscape, metric]);

  const ranked = useMemo(
    () =>
      [...landscape.effects].sort(
        (a, b) => perturbationOf(b, metric) - perturbationOf(a, metric),
      ),
    [landscape, metric],
  );
  const maxPert = ranked.length ? perturbationOf(ranked[0], metric) : 1;

  const showTooltip = (
    e: React.MouseEvent<HTMLButtonElement>,
    effect: LandscapeEffect,
  ) => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const cell = e.currentTarget.getBoundingClientRect();
    const box = wrap.getBoundingClientRect();
    const x = cell.left - box.left + cell.width / 2;
    setHover({
      effect,
      left: Math.max(TOOLTIP_HALF, Math.min(x, box.width - TOOLTIP_HALF)),
      top: cell.top - box.top - 6,
    });
  };

  const rampLabel = metric === "cosine_similarity" ? "1 − cos" : METRIC_LABEL[metric];

  return (
    <div ref={wrapRef} className="relative">
      {/* legend: mini ramp with the perturbation range */}
      <div className="flex items-center justify-end gap-1.5 pb-2 font-mono text-[9px] text-ink3 tabular">
        <span>{rampLabel}</span>
        <span>{fmt(min)}</span>
        <span className="flex h-2 overflow-hidden rounded-[2px]">
          {Array.from({ length: 12 }, (_, i) => (
            <span
              key={i}
              className="h-2 w-1.5"
              style={{ background: blueRamp(i / 11) }}
            />
          ))}
        </span>
        <span>{fmt(max)}</span>
      </div>

      {/* heat strip: 20 canonical AAs, wildtype dotted */}
      <div className="scroll-thin overflow-x-auto pb-1">
        <div className="flex w-max gap-[2px]">
          {AA_LIST.map((aa) => {
            if (aa === landscape.wildtype) {
              return (
                <div key={aa} className="flex w-7 flex-col items-center gap-1">
                  <span className="font-mono text-[11px] leading-none text-ink3">{aa}</span>
                  <span
                    className="flex h-7 w-7 items-center justify-center rounded-[3px] border border-bd bg-surface2"
                    title={`${aa}${landscape.position} · wild type`}
                  >
                    <span className="h-1.5 w-1.5 rounded-full bg-ink3" />
                  </span>
                  <span className="font-mono text-[9px] leading-none text-ink3">wt</span>
                </div>
              );
            }
            const effect = bySub.get(aa);
            if (!effect) {
              return (
                <div key={aa} className="flex w-7 flex-col items-center gap-1">
                  <span className="font-mono text-[11px] leading-none text-ink3">{aa}</span>
                  <span className="h-7 w-7 rounded-[3px] border border-bd" />
                  <span className="font-mono text-[9px] leading-none text-ink3">–</span>
                </div>
              );
            }
            const t = (perturbationOf(effect, metric) - min) / span;
            const isPicked = picked === effect.mutation;
            return (
              <button
                key={aa}
                type="button"
                onClick={() => onPick?.(effect.mutation)}
                onMouseEnter={(e) => showTooltip(e, effect)}
                onMouseLeave={() => setHover(null)}
                className="group flex w-7 flex-col items-center gap-1"
              >
                <span className="font-mono text-[11px] leading-none text-ink">{aa}</span>
                <span
                  className={`h-7 w-7 rounded-[3px] ${
                    isPicked
                      ? "outline outline-1 outline-white/70"
                      : "group-hover:outline group-hover:outline-1 group-hover:outline-white/40"
                  }`}
                  style={{ background: blueRamp(t) }}
                />
                <span className="font-mono text-[9px] leading-none text-ink3 tabular">
                  {fmt(displayOf(effect, metric))}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* ranked substitutions */}
      <div className="mt-4">
        <div className="label-mono pb-1.5">
          ranked by{" "}
          {metric === "cosine_similarity"
            ? "perturbation (1 − cos)"
            : METRIC_LABEL[metric]}
        </div>
        <div className="space-y-px">
          {ranked.map((e) => {
            const frac = maxPert > 0 ? perturbationOf(e, metric) / maxPert : 0;
            const isPicked = picked === e.mutation;
            return (
              <button
                key={e.mutation}
                type="button"
                onClick={() => onPick?.(e.mutation)}
                className={`grid w-full grid-cols-[56px_minmax(0,1fr)_64px] items-center gap-2 rounded px-1 py-[3px] text-left transition-colors ${
                  isPicked ? "bg-surface2" : "hover:bg-surface2"
                }`}
              >
                <span className={`font-mono text-[11px] ${isPicked ? "text-ink" : "text-ink2"}`}>
                  {e.mutation}
                </span>
                <span className="relative h-[5px] overflow-hidden rounded-[2px] bg-surface2">
                  <span
                    className="absolute inset-y-0 left-0 rounded-[2px]"
                    style={{
                      width: `${Math.max(frac * 100, 0.5)}%`,
                      background: "#3987e5",
                    }}
                  />
                </span>
                <span className="text-right font-mono text-[11px] text-ink2 tabular">
                  {fmt(displayOf(e, metric))}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* hover card: exact values of all three metrics */}
      {hover && (
        <div
          className="pointer-events-none absolute z-10 w-44 -translate-x-1/2 -translate-y-full rounded border border-bds bg-surface2 px-2.5 py-2 shadow-xl"
          style={{ left: hover.left, top: hover.top }}
        >
          <div className="pb-1 font-mono text-[11px] text-ink">{hover.effect.mutation}</div>
          <div className="space-y-0.5 font-mono text-[10px] text-ink2 tabular">
            <div className="flex justify-between gap-3">
              <span className="text-ink3">‖Δz‖</span>
              <span>{hover.effect.displacement.toPrecision(4)}</span>
            </div>
            <div className="flex justify-between gap-3">
              <span className="text-ink3">cos</span>
              <span>{hover.effect.cosine_similarity.toFixed(4)}</span>
            </div>
            <div className="flex justify-between gap-3">
              <span className="text-ink3">local Δ (±8)</span>
              <span>{hover.effect.local_delta.toPrecision(4)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
