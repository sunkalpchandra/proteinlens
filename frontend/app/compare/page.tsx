"use client";

/** Pairwise comparison: how far apart two proteins sit in sequence space vs
 *  representation space, per pooling strategy. The interesting quadrants are
 *  the discordant ones — low identity with high cosine (representation-space
 *  neighbors) and the reverse. */

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";
import { ProteinFinder } from "@/components/protein-finder";
import { compareProteins, errorMessage, findProteins, isLive, LIVE_NOTICE } from "@/lib/data";
import { blueRamp } from "@/lib/palette";
import type { ComparePayload, ProteinSummary } from "@/lib/types";


function PickedRow({ p, onClear }: { p: ProteinSummary; onClear: () => void }) {
  return (
    <div className="flex items-center gap-2">
      <span className="font-mono text-[12px] text-accent">{p.accession}</span>
      <span className="min-w-0 flex-1 truncate text-[13px] text-ink">{p.name}</span>
      <button
        type="button"
        className="font-mono text-[11px] text-ink3 hover:text-ink"
        onClick={onClear}
      >
        change
      </button>
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

  // Hydrate ?a=&b= once; the read-then-set is idempotent, so no boot guard.
  useEffect(() => {
    const hydrate = (accession: string | null, set: (p: ProteinSummary) => void) => {
      if (!accession) return;
      findProteins(accession, 1)
        .then((hits) => {
          if (hits[0] && hits[0].accession === accession.toUpperCase()) set(hits[0]);
        })
        .catch(() => undefined);
    };
    hydrate(params.get("a"), setA);
    hydrate(params.get("b"), setB);
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
        setError(errorMessage(e));
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
        <div>
          <span className="label-mono block pb-1">protein A</span>
          {a ? (
            <PickedRow p={a} onClear={() => setA(null)} />
          ) : (
            <ProteinFinder onPick={setA} reflectPick={false} className="" />
          )}
        </div>
        <div>
          <span className="label-mono block pb-1">protein B</span>
          {b ? (
            <PickedRow p={b} onClear={() => setB(null)} />
          ) : (
            <ProteinFinder onPick={setB} reflectPick={false} className="" />
          )}
        </div>
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
