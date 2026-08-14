"use client";

/** Representation benchmarks: probes, retrieval, clustering, stability, and
 *  the sequence-identity vs embedding-cosine scatter. Every number comes from
 *  the /benchmark payload — nothing is hardcoded. */

import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, getBenchmark } from "@/lib/data";
import { OTHER_COLOR, SERIES } from "@/lib/palette";
import type { BenchmarkPayload, BenchmarkRow } from "@/lib/types";

const GRID = "#2c2c2a";
const ACCENT = "var(--accent)";

function num(r: BenchmarkRow, key: string): number | null {
  const v = r[key];
  return typeof v === "number" ? v : null;
}

function fmt(v: number | null, digits = 3): string {
  return v === null ? "—" : v.toFixed(digits);
}

function isEsm(representation: string): boolean {
  return representation.startsWith("esm");
}

/* ---------------------------------------------------------------- tooltip */

interface TipState {
  x: number;
  y: number;
  lines: string[];
}

function Tip({ tip, boxWidth }: { tip: TipState; boxWidth: number }) {
  return (
    <div
      className="pointer-events-none absolute z-10 rounded border border-bds bg-surface2 px-2.5 py-1.5 shadow-xl"
      style={{
        left: Math.min(tip.x + 12, Math.max(0, boxWidth - 190)),
        top: tip.y + 12,
      }}
    >
      {tip.lines.map((line, i) => (
        <div
          key={i}
          className={
            i === 0
              ? "font-mono text-[11px] text-ink"
              : "tabular font-mono text-[11px] text-ink2"
          }
        >
          {line}
        </div>
      ))}
    </div>
  );
}

function useRelativeMouse() {
  const boxRef = useRef<HTMLDivElement>(null);
  const at = (e: React.MouseEvent): { x: number; y: number } => {
    const r = boxRef.current?.getBoundingClientRect();
    return r ? { x: e.clientX - r.left, y: e.clientY - r.top } : { x: 0, y: 0 };
  };
  const width = () => boxRef.current?.getBoundingClientRect().width ?? 640;
  return { boxRef, at, width };
}

/* --------------------------------------------------------- probe bar chart */

const BAR_W = 640;
const LABEL_W = 128;
const VALUE_W = 54;
const ROW_H = 22;
const BAR_H = 8;
const AXIS_H = 16;

function ProbeChart({ rows }: { rows: BenchmarkRow[] }) {
  const { boxRef, at, width } = useRelativeMouse();
  const [tip, setTip] = useState<TipState | null>(null);

  const sorted = useMemo(
    () =>
      [...rows].sort(
        (a, b) => (num(b, "macro_f1") ?? -1) - (num(a, "macro_f1") ?? -1),
      ),
    [rows],
  );
  const plotW = BAR_W - LABEL_W - VALUE_W;
  const h = sorted.length * ROW_H + AXIS_H;
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div ref={boxRef} className="relative">
      <svg viewBox={`0 0 ${BAR_W} ${h}`} className="w-full" role="img">
        {ticks.map((t) => (
          <g key={t}>
            <line
              x1={LABEL_W + t * plotW}
              x2={LABEL_W + t * plotW}
              y1={0}
              y2={h - AXIS_H}
              stroke={GRID}
              strokeWidth={1}
            />
            <text
              x={LABEL_W + t * plotW}
              y={h - 4}
              textAnchor="middle"
              style={{ fill: "var(--ink-3)", fontFamily: "var(--mono)", fontSize: 9 }}
            >
              {t}
            </text>
          </g>
        ))}
        {sorted.map((r, i) => {
          const f1 = num(r, "macro_f1");
          const w = f1 === null ? 0 : f1 * plotW;
          const y = i * ROW_H + (ROW_H - BAR_H) / 2;
          return (
            <g key={r.representation}>
              <text
                x={LABEL_W - 8}
                y={i * ROW_H + ROW_H / 2 + 3.5}
                textAnchor="end"
                style={{ fill: "var(--ink-2)", fontFamily: "var(--mono)", fontSize: 11 }}
              >
                {r.representation}
              </text>
              <rect
                x={LABEL_W}
                y={y}
                width={Math.max(w, 1)}
                height={BAR_H}
                rx={1.5}
                fill={isEsm(r.representation) ? ACCENT : OTHER_COLOR}
              />
              <text
                x={LABEL_W + w + 6}
                y={i * ROW_H + ROW_H / 2 + 3.5}
                style={{ fill: "var(--ink)", fontFamily: "var(--mono)", fontSize: 11 }}
                className="tabular"
              >
                {fmt(f1)}
              </text>
              <rect
                x={0}
                y={i * ROW_H}
                width={BAR_W}
                height={ROW_H}
                fill="transparent"
                onMouseMove={(e) => {
                  const auroc = num(r, "auroc") ?? num(r, "auroc_macro_ovr");
                  const lines = [
                    r.representation,
                    `macro F1  ${fmt(f1)}`,
                    `accuracy  ${fmt(num(r, "accuracy"))}`,
                  ];
                  if (auroc !== null) lines.push(`AUROC     ${fmt(auroc)}`);
                  setTip({ ...at(e), lines });
                }}
                onMouseLeave={() => setTip(null)}
              />
            </g>
          );
        })}
      </svg>
      {tip && <Tip tip={tip} boxWidth={width()} />}
    </div>
  );
}

/* --------------------------------------------------------------- tables */

function metricCell(value: number | null, best: number, digits = 3) {
  const highlighted = value !== null && value === best;
  return (
    <td
      className={`tabular py-1.5 pl-4 text-right font-mono ${
        highlighted ? "font-semibold text-ink" : "text-ink2"
      }`}
    >
      {fmt(value, digits)}
    </td>
  );
}

function RetrievalTable({ rows }: { rows: BenchmarkRow[] }) {
  const metrics = ["precision@1", "precision@5", "precision@10"];
  const tasks = [...new Set(rows.map((r) => r.task ?? ""))].sort();
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[520px] text-[12.5px]">
        <thead>
          <tr className="text-left">
            <th className="label-mono pb-2 font-normal">representation</th>
            <th className="label-mono pb-2 font-normal">task</th>
            <th className="label-mono pb-2 pl-4 text-right font-normal">P@1</th>
            <th className="label-mono pb-2 pl-4 text-right font-normal">P@5</th>
            <th className="label-mono pb-2 pl-4 text-right font-normal">P@10</th>
          </tr>
        </thead>
        <tbody>
          {tasks.flatMap((task) => {
            const group = rows
              .filter((r) => (r.task ?? "") === task)
              .sort(
                (a, b) =>
                  (num(b, "precision@1") ?? -1) - (num(a, "precision@1") ?? -1),
              );
            const best = metrics.map((m) =>
              Math.max(...group.map((r) => num(r, m) ?? -Infinity)),
            );
            return group.map((r) => (
              <tr key={`${task}-${r.representation}`} className="border-t border-bd">
                <td className="py-1.5 font-mono text-ink2">{r.representation}</td>
                <td className="py-1.5 font-mono text-[11px] text-ink3">{task}</td>
                {metrics.map((m, i) => (
                  <MetricTd key={m} value={num(r, m)} best={best[i]} />
                ))}
              </tr>
            ));
          })}
        </tbody>
      </table>
    </div>
  );
}

function MetricTd({
  value,
  best,
  digits = 3,
}: {
  value: number | null;
  best: number;
  digits?: number;
}) {
  return metricCell(value, best, digits);
}

function ClusteringTable({ rows }: { rows: BenchmarkRow[] }) {
  const sorted = [...rows].sort(
    (a, b) => (num(b, "purity") ?? -1) - (num(a, "purity") ?? -1),
  );
  const bestPurity = Math.max(...sorted.map((r) => num(r, "purity") ?? -Infinity));
  const bestNmi = Math.max(...sorted.map((r) => num(r, "nmi") ?? -Infinity));
  return (
    <table className="w-full max-w-md text-[12.5px]">
      <thead>
        <tr className="text-left">
          <th className="label-mono pb-2 font-normal">representation</th>
          <th className="label-mono pb-2 pl-4 text-right font-normal">purity</th>
          <th className="label-mono pb-2 pl-4 text-right font-normal">NMI</th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((r) => (
          <tr key={r.representation} className="border-t border-bd">
            <td className="py-1.5 font-mono text-ink2">{r.representation}</td>
            <MetricTd value={num(r, "purity")} best={bestPurity} />
            <MetricTd value={num(r, "nmi")} best={bestNmi} />
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function StabilityTable({ rows }: { rows: BenchmarkRow[] }) {
  const sorted = [...rows].sort(
    (a, b) => (num(b, "cosine_mean") ?? -1) - (num(a, "cosine_mean") ?? -1),
  );
  return (
    <table className="w-full max-w-lg text-[12.5px]">
      <thead>
        <tr className="text-left">
          <th className="label-mono pb-2 font-normal">representation</th>
          <th className="label-mono pb-2 pl-4 text-right font-normal">cosine mean ± std</th>
          <th className="label-mono pb-2 pl-4 text-right font-normal">p05</th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((r) => (
          <tr key={r.representation} className="border-t border-bd">
            <td className="py-1.5 font-mono text-ink2">{r.representation}</td>
            <td className="tabular py-1.5 pl-4 text-right font-mono text-ink2">
              {fmt(num(r, "cosine_mean"), 4)} ± {fmt(num(r, "cosine_std"), 4)}
            </td>
            <td className="tabular py-1.5 pl-4 text-right font-mono text-ink2">
              {fmt(num(r, "cosine_p05"), 4)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* ------------------------------------------------- identity vs cosine plot */

const SC_W = 640;
const SC_H = 400;
const SC_M = { t: 12, r: 12, b: 38, l: 50 };

interface Pair {
  a: string;
  b: string;
  identity: number;
  cosine: number;
  same_family: boolean;
}

function niceTicks(min: number, max: number, n = 5): number[] {
  const span = max - min || 1;
  const raw = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const err = raw / mag;
  const step = mag * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) {
    out.push(Math.round(v * 1000) / 1000);
  }
  return out;
}

function IdentityScatter({ pairs }: { pairs: Pair[] }) {
  const { boxRef, at, width } = useRelativeMouse();
  const svgRef = useRef<SVGSVGElement>(null);
  const [tip, setTip] = useState<TipState | null>(null);

  const shown = useMemo(() => pairs.slice(0, 4000), [pairs]);

  const plotW = SC_W - SC_M.l - SC_M.r;
  const plotH = SC_H - SC_M.t - SC_M.b;

  const domain = useMemo(() => {
    let yMin = Infinity;
    let yMax = -Infinity;
    for (const p of shown) {
      if (p.cosine < yMin) yMin = p.cosine;
      if (p.cosine > yMax) yMax = p.cosine;
    }
    if (!Number.isFinite(yMin)) return { yMin: 0, yMax: 1 };
    yMin = Math.floor(yMin * 10) / 10;
    yMax = Math.min(1, Math.ceil(yMax * 10) / 10);
    if (yMax - yMin < 0.1) yMin = yMax - 0.1;
    return { yMin, yMax };
  }, [shown]);

  const px = (identity: number) => SC_M.l + identity * plotW;
  const py = (cosine: number) =>
    SC_M.t + (1 - (cosine - domain.yMin) / (domain.yMax - domain.yMin)) * plotH;

  const coords = useMemo(
    () => shown.map((p) => ({ p, x: px(p.identity), y: py(p.cosine) })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [shown, domain],
  );

  const xTicks = [0, 0.2, 0.4, 0.6, 0.8, 1];
  const yTicks = niceTicks(domain.yMin, domain.yMax);

  const onMove = (e: React.MouseEvent) => {
    const svg = svgRef.current;
    if (!svg) return setTip(null);
    const rect = svg.getBoundingClientRect();
    const vx = ((e.clientX - rect.left) * SC_W) / rect.width;
    const vy = ((e.clientY - rect.top) * SC_H) / rect.height;
    let bestD = 64; // 8px in viewBox units, squared
    let hit: { p: Pair } | null = null;
    for (const c of coords) {
      const d = (c.x - vx) * (c.x - vx) + (c.y - vy) * (c.y - vy);
      if (d < bestD) {
        bestD = d;
        hit = c;
      }
    }
    if (!hit) return setTip(null);
    setTip({
      ...at(e),
      lines: [
        `${hit.p.a} × ${hit.p.b}`,
        `identity  ${hit.p.identity.toFixed(3)}`,
        `cosine    ${hit.p.cosine.toFixed(3)}`,
        hit.p.same_family ? "same family" : "different family",
      ],
    });
  };

  return (
    <div ref={boxRef} className="relative">
      <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1">
        <span className="flex items-center gap-1.5 font-mono text-[11px] text-ink2">
          <span className="inline-block h-2 w-2 rounded-full" style={{ background: SERIES[0] }} />
          different family
        </span>
        <span className="flex items-center gap-1.5 font-mono text-[11px] text-ink2">
          <span className="inline-block h-2 w-2 rounded-full" style={{ background: SERIES[1] }} />
          same family
        </span>
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${SC_W} ${SC_H}`}
        className="w-full"
        role="img"
        onMouseMove={onMove}
        onMouseLeave={() => setTip(null)}
      >
        {xTicks.map((t) => (
          <g key={`x${t}`}>
            <line x1={px(t)} x2={px(t)} y1={SC_M.t} y2={SC_M.t + plotH} stroke={GRID} strokeWidth={1} />
            <text
              x={px(t)}
              y={SC_M.t + plotH + 14}
              textAnchor="middle"
              style={{ fill: "var(--ink-3)", fontFamily: "var(--mono)", fontSize: 10 }}
            >
              {t}
            </text>
          </g>
        ))}
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={SC_M.l} x2={SC_M.l + plotW} y1={py(t)} y2={py(t)} stroke={GRID} strokeWidth={1} />
            <text
              x={SC_M.l - 6}
              y={py(t) + 3}
              textAnchor="end"
              style={{ fill: "var(--ink-3)", fontFamily: "var(--mono)", fontSize: 10 }}
            >
              {t}
            </text>
          </g>
        ))}
        <text
          x={SC_M.l + plotW / 2}
          y={SC_H - 6}
          textAnchor="middle"
          style={{ fill: "var(--ink-3)", fontFamily: "var(--mono)", fontSize: 10 }}
        >
          pairwise sequence identity
        </text>
        <text
          x={14}
          y={SC_M.t + plotH / 2}
          textAnchor="middle"
          transform={`rotate(-90 14 ${SC_M.t + plotH / 2})`}
          style={{ fill: "var(--ink-3)", fontFamily: "var(--mono)", fontSize: 10 }}
        >
          embedding cosine (mean pooling)
        </text>
        <g fill={SERIES[0]} fillOpacity={0.35}>
          {coords
            .filter((c) => !c.p.same_family)
            .map((c, i) => (
              <circle key={i} cx={c.x} cy={c.y} r={1.5} />
            ))}
        </g>
        <g fill={SERIES[1]} fillOpacity={0.6}>
          {coords
            .filter((c) => c.p.same_family)
            .map((c, i) => (
              <circle key={i} cx={c.x} cy={c.y} r={1.5} />
            ))}
        </g>
      </svg>
      {tip && <Tip tip={tip} boxWidth={width()} />}
    </div>
  );
}

/* ------------------------------------------------------------------ page */

const PROBE_TASK_ORDER = ["enzyme_vs_nonenzyme", "ec_class", "subcellular_localization"];

export default function BenchmarksPage() {
  const [payload, setPayload] = useState<BenchmarkPayload | null>(null);
  const [error, setError] = useState<ApiError | Error | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    getBenchmark()
      .then((p) => {
        if (!cancelled) setPayload(p);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e : new Error(String(e)));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const rows = payload?.rows ?? [];
  const probeRows = rows.filter((r) => r.axis === "probe");
  const probeTasks = [...new Set(probeRows.map((r) => r.task ?? ""))].sort((a, b) => {
    const ia = PROBE_TASK_ORDER.indexOf(a);
    const ib = PROBE_TASK_ORDER.indexOf(b);
    return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
  });
  const retrievalRows = rows.filter((r) => r.axis === "retrieval");
  const clusteringRows = rows.filter((r) => r.axis === "clustering");
  const stabilityRows = rows.filter((r) => r.axis === "stability");
  const pairs = payload?.seq_vs_emb ?? [];
  const anomalous = pairs.filter((p) => p.identity < 0.2 && p.cosine > 0.9).length;

  const notGenerated =
    (error instanceof ApiError && error.status === 404) ||
    (payload !== null && rows.length === 0 && pairs.length === 0);

  return (
    <div className="mx-auto max-w-[1000px] px-4 py-8">
      <h1 className="text-xl font-semibold text-ink">Benchmarks</h1>
      <p className="mt-1 text-[13px] text-ink3">
        Frozen-representation evaluation: linear probes, neighbor retrieval, clustering
        agreement, and single-substitution perturbation stability, against sequence
        baselines.
      </p>

      {loading && (
        <div className="mt-6 space-y-4">
          <div className="panel loading-pulse h-48 bg-surface2" />
          <div className="panel loading-pulse h-32 bg-surface2" />
        </div>
      )}

      {!loading && notGenerated && (
        <div className="panel mt-6 p-4 text-[13px] text-ink2">
          Benchmarks not generated yet — run{" "}
          <code className="font-mono text-ink">scripts/run_benchmarks.py</code>.
        </div>
      )}

      {!loading && error && !notGenerated && (
        <div className="panel mt-6 p-4 text-[13px] text-ink2">
          {error instanceof ApiError && error.status === 501
            ? "Requires the live API — run the backend locally."
            : error.message}
        </div>
      )}

      {!loading && payload && !notGenerated && (
        <div className="mt-6 space-y-4">
          {/* a) Probes ----------------------------------------------------- */}
          {probeRows.length > 0 && (
            <section className="panel p-4">
              <h2 className="label-mono">Probes — macro F1</h2>
              <p className="mt-1 text-[12px] text-ink3">
                Logistic-regression probes on frozen representations, family-grouped
                splits.
              </p>
              <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1">
                <span className="flex items-center gap-1.5 font-mono text-[11px] text-ink2">
                  <span className="inline-block h-2 w-2 rounded-full" style={{ background: "var(--accent)" }} />
                  ESM-2 pooling
                </span>
                <span className="flex items-center gap-1.5 font-mono text-[11px] text-ink2">
                  <span className="inline-block h-2 w-2 rounded-full" style={{ background: OTHER_COLOR }} />
                  sequence baseline
                </span>
              </div>
              <div className="mt-4 space-y-5">
                {probeTasks.map((task) => (
                  <div key={task}>
                    <p className="mb-1.5 font-mono text-[11px] text-ink2">{task}</p>
                    <ProbeChart rows={probeRows.filter((r) => (r.task ?? "") === task)} />
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* b) Retrieval --------------------------------------------------- */}
          {retrievalRows.length > 0 && (
            <section className="panel p-4">
              <h2 className="label-mono">Retrieval — precision@k</h2>
              <p className="mt-1 text-[12px] text-ink3">
                Fraction of a protein&apos;s k nearest neighbors sharing its label.
                Best value per task and column in bold.
              </p>
              <div className="mt-3">
                <RetrievalTable rows={retrievalRows} />
              </div>
            </section>
          )}

          {/* c) Clustering --------------------------------------------------- */}
          {clusteringRows.length > 0 && (
            <section className="panel p-4">
              <h2 className="label-mono">Clustering agreement</h2>
              <p className="mt-1 text-[12px] text-ink3">
                k-means clusters vs family annotations: purity and normalized mutual
                information.
              </p>
              <div className="mt-3">
                <ClusteringTable rows={clusteringRows} />
              </div>
            </section>
          )}

          {/* d) Stability ---------------------------------------------------- */}
          {stabilityRows.length > 0 && (
            <section className="panel p-4">
              <h2 className="label-mono">Perturbation stability</h2>
              <p className="mt-1 text-[12px] text-ink3">
                Cosine similarity between wild-type and single-substitution embeddings.
              </p>
              <div className="mt-3">
                <StabilityTable rows={stabilityRows} />
              </div>
            </section>
          )}

          {/* e) Identity vs cosine ------------------------------------------- */}
          {pairs.length > 0 && (
            <section className="panel p-4">
              <h2 className="label-mono">Sequence identity vs embedding cosine</h2>
              <p className="mt-1 text-[12px] text-ink3">
                {pairs.length.toLocaleString()} sampled pairs
                {pairs.length > 4000 ? " (4,000 shown)" : ""}.
              </p>
              <div className="mt-3">
                <IdentityScatter pairs={pairs} />
              </div>
              <p className="mt-3 max-w-2xl text-[12px] leading-relaxed text-ink3">
                High-cosine, low-identity pairs are representation-space neighbors — the
                model places them close, which is not by itself evidence of homology or
                shared function. {anomalous.toLocaleString()} of{" "}
                {pairs.length.toLocaleString()} sampled pairs have identity &lt; 0.2 with
                cosine &gt; 0.9; the sample deliberately includes an embedding
                nearest-neighbor stratum to populate the high-similarity region, so these
                are counts within the sample, not corpus-level rates.
              </p>
            </section>
          )}
        </div>
      )}
    </div>
  );
}
