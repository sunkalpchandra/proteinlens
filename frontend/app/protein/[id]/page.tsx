"use client";

/** Protein profile: identity header, representation stats, attention-pooling
 *  weights (when available), nearest neighbors in embedding space, and the
 *  full sequence with residue selection wired to the mutation page.
 */

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { ResidueTrack } from "@/components/residue-track";
import { SequenceViewer } from "@/components/sequence-viewer";
import { StatGrid, type StatItem } from "@/components/stat-grid";
import { ApiError, getAttention, getProfile, isLive } from "@/lib/data";
import { AA_CATEGORY, aaColor } from "@/lib/palette";
import type { AttentionPayload, Pooling, ProteinProfile, SearchHit } from "@/lib/types";

const POOLINGS: Pooling[] = ["mean", "max", "bos", "attention"];
const MAX_KEYWORDS = 12;

function errorText(e: unknown): string {
  if (e instanceof ApiError && e.status === 501) {
    return "Requires the live API — run the backend locally.";
  }
  if (e instanceof Error) return e.message;
  return "Something went wrong.";
}

function ordinal(n: number): string {
  const r10 = n % 10;
  const r100 = n % 100;
  if (r10 === 1 && r100 !== 11) return `${n}st`;
  if (r10 === 2 && r100 !== 12) return `${n}nd`;
  if (r10 === 3 && r100 !== 13) return `${n}rd`;
  return `${n}th`;
}

function fmt(v: number | null | undefined, digits = 3): string {
  return v == null ? "—" : v.toFixed(digits);
}

// ---------------------------------------------------------------------------

function Section({
  title,
  action,
  children,
}: {
  title: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="mt-7 border-t border-bd pt-4">
      <div className="flex items-baseline justify-between gap-3 pb-3">
        <h2 className="label-mono">{title}</h2>
        {action}
      </div>
      {children}
    </section>
  );
}

function MetaItem({ label, value }: { label: string; value: string | null }) {
  return (
    <span className="flex items-baseline gap-1.5 font-mono text-[11px]">
      <span className="text-ink3">{label}</span>
      <span className="text-ink2">{value ?? "—"}</span>
    </span>
  );
}

/** Compact neighbor list: rank, accession link, name, organism, similarity bar. */
function NeighborList({ hits }: { hits: SearchHit[] }) {
  if (!hits.length) {
    return <p className="text-[13px] text-ink3">No neighbors available for this protein.</p>;
  }
  return (
    <div className="panel divide-y divide-bd">
      {hits.map((h) => (
        <div key={h.protein.accession} className="flex items-center gap-3 px-3 py-1.5">
          <span className="w-5 shrink-0 text-right font-mono text-[11px] text-ink3 tabular">
            {h.rank}
          </span>
          <Link
            href={`/protein/${encodeURIComponent(h.protein.accession)}`}
            className="w-[72px] shrink-0 font-mono text-[12px] text-accent hover:underline"
          >
            {h.protein.accession}
          </Link>
          <span className="min-w-0 flex-1 truncate text-[13px] text-ink" title={h.protein.name}>
            {h.protein.name}
          </span>
          <span
            className="hidden w-44 shrink-0 truncate text-right text-[12px] text-ink3 md:block"
            title={h.protein.organism}
          >
            {h.protein.organism}
          </span>
          <span
            className="hidden h-[3px] w-24 shrink-0 overflow-hidden rounded-full bg-surface2 sm:block"
            aria-hidden
          >
            <span
              className="block h-full rounded-full bg-accent"
              style={{ width: `${Math.max(0, Math.min(1, h.similarity)) * 100}%` }}
            />
          </span>
          <span className="w-12 shrink-0 text-right font-mono text-[11px] text-ink2 tabular">
            {h.similarity.toFixed(3)}
          </span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------

type AttentionState = "loading" | "ok" | "missing" | "error";

export default function ProteinPage() {
  const params = useParams<{ id: string | string[] }>();
  const raw = params?.id;
  const id = decodeURIComponent(typeof raw === "string" ? raw : (raw?.[0] ?? ""));

  const [pooling, setPooling] = useState<Pooling>("mean");
  const [profile, setProfile] = useState<ProteinProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [attention, setAttention] = useState<AttentionPayload | null>(null);
  const [attState, setAttState] = useState<AttentionState>("loading");
  const [attError, setAttError] = useState<string | null>(null);

  const [selected, setSelected] = useState<number | null>(null); // 1-based

  // Profile (re-fetched on pooling change; selection resets on id change).
  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    getProfile(id, pooling)
      .then((p) => {
        if (!cancelled) setProfile(p);
      })
      .catch((e) => {
        if (!cancelled) {
          setProfile(null);
          setError(errorText(e));
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [id, pooling]);

  useEffect(() => {
    setSelected(null);
  }, [id]);

  // Attention loads lazily once the profile is in; keyed on accession so a
  // pooling re-fetch does not re-request it.
  const accession = profile?.protein.accession ?? null;
  useEffect(() => {
    if (!accession) return;
    let cancelled = false;
    setAttention(null);
    setAttState("loading");
    setAttError(null);
    getAttention(accession)
      .then((a) => {
        if (cancelled) return;
        setAttention(a);
        setAttState("ok");
      })
      .catch((e) => {
        if (cancelled) return;
        if (e instanceof ApiError && e.status === 404) {
          setAttState("missing");
        } else {
          setAttState("error");
          setAttError(errorText(e));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [accession]);

  const toggleSelect = (pos: number) => setSelected((s) => (s === pos ? null : pos));

  // --- loading / error shells ------------------------------------------------

  if (!profile && loading) {
    return (
      <div className="mx-auto max-w-[1100px] px-4 py-6">
        <div className="loading-pulse space-y-6">
          <div className="space-y-2">
            <div className="h-6 w-64 rounded bg-surface" />
            <div className="h-4 w-full max-w-[36rem] rounded bg-surface" />
            <div className="h-3 w-80 max-w-full rounded bg-surface" />
          </div>
          <div className="grid grid-cols-2 gap-px md:grid-cols-4">
            {Array.from({ length: 8 }, (_, i) => (
              <div key={i} className="h-16 bg-surface" />
            ))}
          </div>
          <div className="h-32 rounded bg-surface" />
          <div className="h-64 rounded bg-surface" />
        </div>
        <p className="pt-4 font-mono text-[11px] text-ink3">loading profile…</p>
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="mx-auto max-w-[1100px] px-4 py-6">
        <div className="panel p-4">
          <div className="label-mono pb-1">error</div>
          <p className="text-[13px] text-ink2">{error ?? "Profile unavailable."}</p>
          <p className="mt-1 font-mono text-[11px] text-ink3">{id}</p>
        </div>
      </div>
    );
  }

  // --- loaded ------------------------------------------------------------------

  const p = profile.protein;
  const sequence = profile.sequence;
  const stats = profile.stats;
  const selectedAA =
    selected !== null && selected >= 1 && selected <= sequence.length
      ? sequence[selected - 1]
      : null;

  const statItems: StatItem[] = [
    { label: "embedding norm", value: fmt(stats.embedding_norm, 2) },
    { label: "dim", value: stats.dim },
    { label: "nn distance", value: fmt(stats.nn_distance), hint: "cosine, nearest neighbor" },
    { label: "knn mean distance", value: fmt(stats.knn_mean_distance), hint: "mean over k nearest" },
    { label: "cluster", value: stats.cluster == null ? "—" : `#${stats.cluster}` },
    {
      label: "outlier",
      value:
        stats.outlier_score == null
          ? "—"
          : `${stats.outlier_score.toFixed(2)} (${ordinal(Math.round(stats.outlier_score * 100))} pct)`,
    },
    { label: "model", value: profile.model, mono: true },
    { label: "pooling", value: pooling, mono: true },
  ];

  const keywords = profile.keywords.slice(0, MAX_KEYWORDS);
  const extraKeywords = profile.keywords.length - keywords.length;

  return (
    <div className="mx-auto max-w-[1100px] px-4 py-6">
      {/* Header ------------------------------------------------------------- */}
      <header>
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <h1 className="text-lg font-semibold text-ink">{p.name}</h1>
          <span className="font-mono text-[12px] text-ink3">{p.accession}</span>
          <span className="font-mono text-[12px] text-ink3 tabular">{p.length} aa</span>
          {isLive && (
            <label className="ml-auto flex items-center gap-2 font-mono text-[11px] text-ink3">
              pooling
              <select
                value={pooling}
                onChange={(e) => setPooling(e.target.value as Pooling)}
                className="rounded border border-bd bg-surface px-1.5 py-0.5 font-mono text-[11px] text-ink2 outline-none focus:border-bds"
              >
                {POOLINGS.map((pl) => (
                  <option key={pl} value={pl}>
                    {pl}
                  </option>
                ))}
              </select>
              {loading && <span className="loading-pulse">loading…</span>}
            </label>
          )}
        </div>
        <p className="mt-1 line-clamp-2 max-w-[72ch] text-[13px] leading-snug text-ink2">
          {profile.protein_name_full}
        </p>
        <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1.5">
          <MetaItem label="gene" value={p.gene} />
          <MetaItem label="organism" value={p.organism} />
          <MetaItem label="family" value={p.family} />
          <MetaItem label="localization" value={p.localization} />
          <MetaItem label="ec" value={p.ec_class} />
          <MetaItem label="pfam" value={p.pfam} />
        </div>
        {keywords.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {keywords.map((kw) => (
              <span
                key={kw}
                className="rounded border border-bd bg-surface px-1.5 py-0.5 text-[11px] text-ink2"
              >
                {kw}
              </span>
            ))}
            {extraKeywords > 0 && (
              <span className="px-1 py-0.5 font-mono text-[11px] text-ink3">
                +{extraKeywords} more
              </span>
            )}
          </div>
        )}
      </header>

      {/* Representation ------------------------------------------------------ */}
      <Section
        title="representation"
        action={
          <Link
            href={`/explorer?p=${encodeURIComponent(p.accession)}`}
            className="text-[12px] text-accent hover:underline"
          >
            View in explorer →
          </Link>
        }
      >
        <StatGrid items={statItems} />
      </Section>

      {/* Attention ------------------------------------------------------------ */}
      <Section title="attention">
        {attState === "loading" && (
          <p className="loading-pulse font-mono text-[11px] text-ink3">loading attention weights…</p>
        )}
        {attState === "missing" && (
          <p className="text-[13px] text-ink3">No precomputed attention weights for this protein.</p>
        )}
        {attState === "error" && (
          <p className="text-[13px] text-ink3">{attError ?? "Attention weights unavailable."}</p>
        )}
        {attState === "ok" && attention && (
          <div>
            <ResidueTrack
              values={attention.weights}
              marked={selected}
              label="attention pooling weights α"
              onClickPos={toggleSelect}
            />
            <p className="mt-1 text-xs text-ink3">{attention.note}</p>
            {attention.top_positions.length > 0 && (
              <div className="mt-2.5 flex flex-wrap items-baseline gap-1.5">
                <span className="label-mono mr-1">top positions</span>
                {attention.top_positions.map((pos) => {
                  const isSel = selected === pos;
                  return (
                    <button
                      key={pos}
                      type="button"
                      onClick={() => toggleSelect(pos)}
                      className={`rounded border px-1.5 py-0.5 font-mono text-[11px] tabular transition-colors ${
                        isSel
                          ? "border-bds bg-surface2 text-ink"
                          : "border-bd text-ink2 hover:border-bds hover:text-ink"
                      }`}
                    >
                      {sequence[pos - 1] ?? "?"}
                      {pos}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </Section>

      {/* Similar proteins ------------------------------------------------------ */}
      <Section title="similar proteins">
        <NeighborList hits={profile.neighbors} />
      </Section>

      {/* Sequence --------------------------------------------------------------- */}
      <Section title="sequence">
        <SequenceViewer
          sequence={sequence}
          selected={selected}
          onSelect={toggleSelect}
          intensity={attState === "ok" ? (attention?.weights ?? null) : null}
          intensityLabel="attention α"
        />
        {selected !== null && selectedAA !== null && (
          <div className="panel mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 px-3 py-2">
            <span className="font-mono text-[12px] text-ink tabular">
              {selectedAA}
              {selected}
            </span>
            <span className="flex items-center gap-1.5 text-[12px] text-ink2">
              <span
                className="inline-block h-2 w-2 rounded-sm"
                style={{ background: aaColor(selectedAA) }}
              />
              {AA_CATEGORY[selectedAA] ?? "unknown"}
            </span>
            <span className="font-mono text-[11px] text-ink3 tabular">
              position {selected} / {sequence.length}
            </span>
            <Link
              href={`/mutation?p=${encodeURIComponent(p.accession)}&pos=${selected}`}
              className="ml-auto text-[12px] text-accent hover:underline"
            >
              Analyze mutations at this position →
            </Link>
          </div>
        )}
      </Section>
    </div>
  );
}
