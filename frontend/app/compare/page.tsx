"use client";

/** Pairwise comparison: how far apart two proteins sit in sequence space vs
 *  representation space, per pooling strategy. The interesting quadrants are
 *  the discordant ones — low identity with high cosine (representation-space
 *  neighbors) and the reverse. */

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";
import { ApiError, compareProteins, findProteins, isLive } from "@/lib/data";
import { blueRamp } from "@/lib/palette";
import type { ComparePayload, ProteinSummary } from "@/lib/types";

const LIVE_NOTICE = "Requires the live API — run the backend locally.";

function Finder({
  label,
  value,
  onPick,
}: {
  label: string;
  value: ProteinSummary | null;
  onPick: (p: ProteinSummary | null) => void;
}) {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<ProteinSummary[]>([]);
  const [open, setOpen] = useState(false);
  const seq = useRef(0);

  useEffect(() => {
    const q = query.trim();
    if (q.length < 2) {
      setHits([]);
      return;
    }
    const mine = ++seq.current;
    const timer = setTimeout(() => {
      findProteins(q, 8)
        .then((results) => {
          if (seq.current === mine) {
            setHits(results);
            setOpen(true);
          }
        })
        .catch(() => undefined);
    }, 150);
    return () => clearTimeout(timer);
  }, [query]);

  return (
    <div className="relative">
      <span className="label-mono block pb-1">{label}</span>
      {value ? (
        <div className="flex items-center gap-2">
          <span className="font-mono text-[12px] text-accent">{value.accession}</span>
          <span className="min-w-0 flex-1 truncate text-[13px] text-ink">{value.name}</span>
          <button
            type="button"
            className="font-mono text-[11px] text-ink3 hover:text-ink"
            onClick={() => {
              setQuery("");
              onPick(null);
            }}
          >
            change
          </button>
        </div>
      ) : (
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => hits.length && setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          placeholder="name, gene, or accession…"
          className="w-full rounded border border-bd bg-surface2 px-2 py-1.5 text-[13px] text-ink placeholder:text-ink3 focus:border-bds focus:outline-none"
        />
      )}
      {open && hits.length > 0 && !value && (
        <div className="absolute z-20 mt-1 w-full overflow-hidden rounded border border-bds bg-surface2 shadow-xl">
          {hits.map((h) => (
            <button
              key={h.accession}
              type="button"
              className="flex w-full items-baseline gap-2 px-2.5 py-1.5 text-left hover:bg-surface"
              onMouseDown={() => {
                onPick(h);
                setOpen(false);
                setQuery("");
              }}
            >
              <span className="font-mono text-[11px] text-accent">{h.accession}</span>
              <span className="min-w-0 flex-1 truncate text-[12.5px] text-ink">{h.name}</span>
              <span className="shrink-0 text-[11px] text-ink3">{h.organism}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function SummaryCard({ p }: { p: ProteinSummary }) {
  return (
    <div className="panel p-3">
      <Link
        href={`/protein/${encodeURIComponent(p.accession)}`}
        className="font-mono text-[12px] text-accent hover:underline"
      >
        {p.accession}
      </Link>
      <div className="truncate pt-0.5 text-[13.5px] text-ink" title={p.name}>
        {p.name}
      </div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-1 pt-2 font-mono text-[11px] text-ink2">
        <span><span className="text-ink3">org </span>{p.organism}</span>
        <span><span className="text-ink3">len </span>{p.length} aa</span>
        <span className="col-span-2 truncate" title={p.family ?? undefined}>
          <span className="text-ink3">family </span>{p.family ?? "—"}
        </span>
        <span><span className="text-ink3">pfam </span>{p.pfam ?? "—"}</span>
        <span><span className="text-ink3">EC </span>{p.ec_class ?? "—"}</span>
      </div>
    </div>
  );
}

function MetricBar({ label, value }: { label: string; value: number }) {
  const clamped = Math.max(0, Math.min(1, value));
  return (
    <div className="flex items-center gap-3">
      <span className="w-24 shrink-0 font-mono text-[11px] text-ink3">{label}</span>
      <span className="block h-[5px] flex-1 overflow-hidden rounded-full bg-surface2">
        <span
          className="block h-full rounded-full"
          style={{ width: `${clamped * 100}%`, background: blueRamp(clamped) }}
        />
      </span>
      <span className="w-14 shrink-0 text-right font-mono text-[12px] text-ink tabular">
        {value.toFixed(4)}
      </span>
    </div>
  );
}

function CompareWorkbench() {
  const params = useSearchParams();
  const [a, setA] = useState<ProteinSummary | null>(null);
  const [b, setB] = useState<ProteinSummary | null>(null);
  const [result, setResult] = useState<ComparePayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const req = useRef(0);
  const booted = useRef(false);

  useEffect(() => {
    if (booted.current) return;
    booted.current = true;
    for (const [key, set] of [["a", setA], ["b", setB]] as const) {
      const accession = params.get(key);
      if (accession) {
        findProteins(accession, 1)
          .then((hits) => hits[0] && hits[0].accession === accession.toUpperCase() && set(hits[0]))
          .catch(() => undefined);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    setResult(null);
    setError(null);
    if (!a || !b) return;
    if (a.accession === b.accession) {
      setError("Choose two different proteins.");
      return;
    }
    const mine = ++req.current;
    setLoading(true);
    compareProteins(a.accession, b.accession)
      .then((payload) => req.current === mine && setResult(payload))
      .catch((e: unknown) => {
        if (req.current !== mine) return;
        setError(
          e instanceof ApiError && e.status === 501
            ? LIVE_NOTICE
            : e instanceof Error
              ? e.message
              : "Comparison failed.",
        );
      })
      .finally(() => req.current === mine && setLoading(false));
  }, [a, b]);

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <h1 className="pb-1 text-lg font-semibold text-ink">Compare two proteins</h1>
      <p className="pb-5 text-[12.5px] text-ink3">
        Sequence identity vs representation similarity, across all four pooling
        strategies.{!isLive && ` ${LIVE_NOTICE}`}
      </p>

      <div className="grid gap-4 sm:grid-cols-2">
        <Finder label="protein A" value={a} onPick={setA} />
        <Finder label="protein B" value={b} onPick={setB} />
      </div>

      {loading && (
        <p className="loading-pulse pt-5 font-mono text-[11px] text-ink3">comparing…</p>
      )}
      {error && <div className="panel mt-5 px-3 py-2 text-[12.5px] text-ink2">{error}</div>}

      {result && (
        <div className="space-y-5 pt-6">
          <div className="grid gap-3 sm:grid-cols-2">
            <SummaryCard p={result.a} />
            <SummaryCard p={result.b} />
          </div>

          <div className="panel space-y-2.5 p-4">
            <div className="flex items-baseline justify-between">
              <span className="label-mono">similarity</span>
              <span className="font-mono text-[10px] text-ink3">
                {result.same_family ? "same UniProt family" : "different families"}
                {result.shared_pfam.length > 0 &&
                  ` · shared Pfam: ${result.shared_pfam.join(", ")}`}
              </span>
            </div>
            <MetricBar label="identity" value={result.sequence_identity} />
            {Object.entries(result.cosine_by_pooling).map(([pooling, cosine]) => (
              <MetricBar key={pooling} label={`cos ${pooling}`} value={cosine} />
            ))}
            <p className="pt-1 text-[11px] leading-relaxed text-ink3">{result.note}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default function ComparePage() {
  return (
    <Suspense
      fallback={
        <div className="mx-auto max-w-4xl px-4 py-8">
          <p className="loading-pulse font-mono text-xs text-ink3">loading…</p>
        </div>
      }
    >
      <CompareWorkbench />
    </Suspense>
  );
}
