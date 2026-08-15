"use client";

/** Representation trajectory builder: a chain of mutations applied
 *  cumulatively, visualized as sequential movement in embedding space.
 *
 *  The schematic places each step left-to-right with horizontal spacing
 *  proportional to its step displacement — an honest 1-D unrolling of the
 *  path, explicitly NOT a projection into the UMAP map. Labeled as
 *  representation-space movement, never an evolutionary trajectory.
 */

import { useEffect, useRef, useState } from "react";
import { errorMessage, isLive, trajectoryOf } from "@/lib/data";
import type { Pooling, TrajectoryPayload } from "@/lib/types";

const MUTATION_RE = /^[A-Z]\d+[A-Z]$/;
const MAX_STEPS = 10;

export interface TrajectoryPanelProps {
  accession: string;
  pooling: Pooling;
  /** Substitution picked in the landscape view, offered as a quick add. */
  seedMutation?: string | null;
}

export function TrajectoryPanel({ accession, pooling, seedMutation = null }: TrajectoryPanelProps) {
  const [chain, setChain] = useState<string[]>([]);
  const [draft, setDraft] = useState("");
  const [draftError, setDraftError] = useState<string | null>(null);
  const [result, setResult] = useState<TrajectoryPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const req = useRef(0);

  const invalidate = () => {
    req.current += 1;
    setResult(null);
    setError(null);
  };

  // Results are pooling-specific; a pooling switch must not leave numbers
  // computed under the previous strategy on screen.
  useEffect(() => {
    invalidate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pooling]);

  const add = (mutation: string) => {
    const clean = mutation.trim().toUpperCase();
    if (!MUTATION_RE.test(clean)) {
      setDraftError("Format: wild-type, position, mutant — e.g. E7V");
      return;
    }
    if (chain.length >= MAX_STEPS) {
      setDraftError(`Maximum ${MAX_STEPS} steps.`);
      return;
    }
    setDraftError(null);
    setChain((c) => [...c, clean]);
    setDraft("");
    invalidate();
  };

  const run = async () => {
    if (!chain.length) return;
    invalidate();
    const mine = req.current;
    setLoading(true);
    try {
      const payload = await trajectoryOf({ accession }, chain, pooling);
      if (req.current === mine) setResult(payload);
    } catch (e) {
      if (req.current === mine) {
        setError(errorMessage(e));
      }
    } finally {
      if (req.current === mine) setLoading(false);
    }
  };

  if (!isLive) {
    return (
      <p className="text-[12px] text-ink3">
        Trajectories run live ESM inference — run the backend locally to use them.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        {chain.map((m, i) => (
          <span
            key={`${m}-${i}`}
            className="flex items-center gap-1 rounded border border-bd bg-surface2 px-1.5 py-0.5 font-mono text-[11px] text-ink2"
          >
            <span className="text-ink3">{i + 1}.</span> {m}
            <button
              type="button"
              aria-label={`truncate from ${m}`}
              title="Remove this and later steps (later labels depend on this one)"
              className="ml-0.5 text-ink3 hover:text-ink"
              onClick={() => {
                setChain((c) => c.slice(0, i));
                invalidate();
              }}
            >
              ×
            </button>
          </span>
        ))}
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && add(draft)}
          placeholder={chain.length ? "next mutation…" : "e.g. E7V"}
          className="w-28 rounded border border-bd bg-surface2 px-2 py-1 font-mono text-[12px] text-ink placeholder:text-ink3 focus:border-bds focus:outline-none"
        />
        <button
          type="button"
          onClick={() => add(draft)}
          className="rounded border border-bd px-2 py-1 font-mono text-[11px] text-ink2 hover:bg-surface2"
        >
          add
        </button>
        {seedMutation && !chain.includes(seedMutation) && (
          <button
            type="button"
            onClick={() => add(seedMutation)}
            className="rounded border border-bd px-2 py-1 font-mono text-[11px] text-accent hover:bg-surface2"
          >
            add {seedMutation}
          </button>
        )}
        <button
          type="button"
          disabled={!chain.length || loading}
          onClick={() => void run()}
          className="rounded bg-accent px-2.5 py-1 font-mono text-[11px] text-white disabled:opacity-40"
        >
          {loading ? "running…" : "run trajectory"}
        </button>
        {chain.length > 0 && (
          <button
            type="button"
            onClick={() => {
              setChain([]);
              invalidate();
            }}
            className="font-mono text-[11px] text-ink3 hover:text-ink"
          >
            clear
          </button>
        )}
      </div>
      {draftError && <p className="font-mono text-[11px] text-ink3">{draftError}</p>}
      {error && <p className="panel px-3 py-2 text-[12px] text-ink2">{error}</p>}
      {loading && (
        <p className="loading-pulse font-mono text-[11px] text-ink3">
          encoding {chain.length + 1} sequences…
        </p>
      )}

      {result && <TrajectoryResult result={result} />}
    </div>
  );
}

function TrajectoryResult({ result }: { result: TrajectoryPayload }) {
  // Layout: each gap = a fixed minimum plus a share proportional to that
  // step's displacement, then the whole path is normalized onto the viewBox.
  const width = 680;
  const PAD = 24;
  const MIN_GAP = 1; // in units of "one max-sized displacement"
  const maxD = Math.max(...result.steps.map((s) => s.step_displacement), 1e-9);
  const gaps = result.steps.map((s) => MIN_GAP + s.step_displacement / maxD);
  const total = gaps.reduce((a, b) => a + b, 0);
  const px: number[] = [PAD];
  for (const gap of gaps) {
    px.push(px[px.length - 1] + (gap / total) * (width - 2 * PAD));
  }
  const y = 46;

  return (
    <div className="space-y-3">
      <svg viewBox={`0 0 ${width} 92`} className="w-full" role="img"
           aria-label="representation trajectory schematic">
        {px.slice(0, -1).map((x, i) => (
          <g key={i}>
            <line x1={x} y1={y} x2={px[i + 1]} y2={y} stroke="#3987e5" strokeWidth={1.5} />
            <text x={(x + px[i + 1]) / 2} y={y - 26}
                  textAnchor="middle"
                  style={{ fill: "var(--ink-3)", fontFamily: "var(--mono)", fontSize: 9 }}>
              {result.steps[i].step_displacement.toFixed(3)}
            </text>
            <text x={(x + px[i + 1]) / 2} y={y - 14}
                  textAnchor="middle"
                  style={{ fill: "var(--ink-2)", fontFamily: "var(--mono)", fontSize: 10 }}>
              {result.steps[i].mutation}
            </text>
          </g>
        ))}
        {px.map((x, i) => (
          <g key={i}>
            <circle cx={x} cy={y} r={i === 0 ? 5 : 4}
                    fill={i === 0 ? "#f2f2ef" : "#3987e5"} />
            <text x={x} y={y + 22} textAnchor="middle"
                  style={{ fill: "var(--ink-3)", fontFamily: "var(--mono)", fontSize: 9 }}>
              {i === 0 ? "WT" : `+${i}`}
            </text>
          </g>
        ))}
      </svg>
      <div className="flex flex-wrap gap-x-6 gap-y-1 font-mono text-[11px] text-ink2 tabular">
        <span><span className="text-ink3">path length</span> {result.path_length.toFixed(3)}</span>
        <span><span className="text-ink3">net ‖Δz‖ from WT</span> {result.net_displacement.toFixed(3)}</span>
        <span><span className="text-ink3">directness</span> {result.directness.toFixed(2)}</span>
        <span><span className="text-ink3">cos(WT, final)</span> {result.steps[result.n_steps - 1].cosine_to_wt.toFixed(4)}</span>
      </div>
      <p className="max-w-2xl text-[11px] leading-relaxed text-ink3">
        Schematic: steps unrolled left-to-right with spacing proportional to each
        step&apos;s displacement — movement in embedding space, not the UMAP layout.{" "}
        {result.note}
      </p>
    </div>
  );
}
