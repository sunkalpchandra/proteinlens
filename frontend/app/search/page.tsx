"use client";

/** Search — metadata lookup over the corpus, or nearest-neighbor retrieval
 *  for a pasted amino-acid sequence (live API only; demo mode disables it). */

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { DownloadJson } from "@/components/download-json";
import { HitTable } from "@/components/hit-table";
import { ApiError, errorMessage, fetchProtein, findProteins, isLive, searchBySequence } from "@/lib/data";
import type { FetchProteinPayload, Pooling, ProteinSummary, SearchHit } from "@/lib/types";

const VALID_AA = new Set("ACDEFGHIKLMNPQRSTVWY".split(""));
const MIN_LEN = 10;
const MAX_LEN = 2048;

const EXAMPLE_SEQUENCE =
  "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR";

const POOLINGS: Pooling[] = ["mean", "max", "bos", "attention"];
const K_OPTIONS = [10, 25, 50] as const;

const LIVE_ONLY_NOTICE = "Requires the live API — run the backend locally.";

type Tab = "metadata" | "sequence";

function summariesToHits(results: ProteinSummary[]): SearchHit[] {
  return results.map((p, i) => ({ rank: i + 1, similarity: 0, protein: p }));
}

function TableSkeleton({ rows = 8 }: { rows?: number }) {
  return (
    <div className="loading-pulse space-y-2 py-2" aria-hidden>
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="h-6 rounded bg-surface2" />
      ))}
    </div>
  );
}

function ErrorPanel({ message }: { message: string }) {
  return (
    <div className="panel border-bds px-3 py-2.5 text-[13px] text-ink2">
      <span className="label-mono mr-2 text-ink3">error</span>
      {message}
    </div>
  );
}

function QuietNotice({ children }: { children: React.ReactNode }) {
  return <p className="py-2 font-mono text-[12px] text-ink3">{children}</p>;
}

function SearchWorkbench() {
  const router = useRouter();
  const params = useSearchParams();
  const booted = useRef(false);
  const [tab, setTab] = useState<Tab>("metadata");

  // --- metadata tab -------------------------------------------------------
  const [metaQuery, setMetaQuery] = useState("");
  const [metaResults, setMetaResults] = useState<ProteinSummary[] | null>(null);
  const [metaLoading, setMetaLoading] = useState(false);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [fetched, setFetched] = useState<FetchProteinPayload | null>(null);
  const [fetchLoading, setFetchLoading] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  const runFetch = useCallback(async (accession: string) => {
    setFetchLoading(true);
    setFetchError(null);
    setFetched(null);
    try {
      setFetched(await fetchProtein(accession));
    } catch (e) {
      setFetchError(errorMessage(e));
    } finally {
      setFetchLoading(false);
    }
  }, []);

  const runMetaSearch = useCallback(
    async (q: string) => {
      const query = q.trim();
      if (!query) return;
      setFetched(null);
      setFetchError(null);
      // Shareable searches: the query lives in the URL.
      router.replace(`/search?q=${encodeURIComponent(query)}`, { scroll: false });
      setMetaLoading(true);
      setMetaError(null);
      try {
        setMetaResults(await findProteins(query));
      } catch (e) {
        setMetaResults(null);
        setMetaError(e instanceof Error ? e.message : String(e));
      } finally {
        setMetaLoading(false);
      }
    },
    [router],
  );

  // Arrival via /search?q=… — run once.
  useEffect(() => {
    if (booted.current) return;
    booted.current = true;
    const q = params.get("q");
    if (q) {
      setMetaQuery(q);
      void runMetaSearch(q);
    }
  }, [params, runMetaSearch]);

  // --- sequence tab -------------------------------------------------------
  const [seqRaw, setSeqRaw] = useState("");
  const [k, setK] = useState<number>(10);
  const [pooling, setPooling] = useState<Pooling>("mean");
  const [seqHits, setSeqHits] = useState<SearchHit[] | null>(null);
  const [seqLoading, setSeqLoading] = useState(false);
  const [seqError, setSeqError] = useState<string | null>(null);
  const [seqNeedsLive, setSeqNeedsLive] = useState(false);

  const cleaned = useMemo(() => seqRaw.toUpperCase().replace(/\s+/g, ""), [seqRaw]);

  const seqValidation = useMemo<string | null>(() => {
    if (!cleaned) return null;
    const bad = [...new Set(cleaned.split("").filter((c) => !VALID_AA.has(c)))];
    if (bad.length > 0) {
      return `Invalid characters: ${bad.slice(0, 8).join(" ")}${bad.length > 8 ? " …" : ""} — only the 20 standard amino acids are allowed.`;
    }
    if (cleaned.length < MIN_LEN) {
      return `Sequence too short: ${cleaned.length} aa (minimum ${MIN_LEN}).`;
    }
    if (cleaned.length > MAX_LEN) {
      return `Sequence too long: ${cleaned.length} aa (maximum ${MAX_LEN}).`;
    }
    return null;
  }, [cleaned]);

  const seqSubmittable = cleaned.length > 0 && seqValidation === null;

  const runSeqSearch = useCallback(async () => {
    if (!seqSubmittable) return;
    setSeqLoading(true);
    setSeqError(null);
    setSeqNeedsLive(false);
    try {
      setSeqHits(await searchBySequence(cleaned, pooling, k));
    } catch (e) {
      setSeqHits(null);
      if (e instanceof ApiError && e.status === 501) {
        setSeqNeedsLive(true);
      } else {
        setSeqError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setSeqLoading(false);
    }
  }, [cleaned, pooling, k, seqSubmittable]);

  const selectClass =
    "rounded border border-bd bg-surface px-2 py-1 font-mono text-[12px] text-ink outline-none focus:border-bds";

  return (
    <div className="mx-auto max-w-[1100px] px-4 py-6">
      <div className="flex items-baseline justify-between">
        <h1 className="text-[15px] font-semibold text-ink">Search</h1>
        <span className="label-mono">
          {isLive ? "live api" : "demo corpus"}
        </span>
      </div>
      <p className="pt-1 text-[13px] text-ink3">
        Find proteins by metadata, or retrieve nearest neighbors in embedding space
        for a pasted sequence.
      </p>

      {/* tabs */}
      <div className="mt-4 flex gap-4 border-b border-bd" role="tablist">
        {(
          [
            ["metadata", "Metadata"],
            ["sequence", "Sequence"],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={tab === id}
            onClick={() => setTab(id)}
            className={`-mb-px border-b px-1 pb-2 font-mono text-[12px] tracking-[0.06em] uppercase transition-colors ${
              tab === id
                ? "border-accent text-ink"
                : "border-transparent text-ink3 hover:text-ink2"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* ------------------------------------------------ metadata tab --- */}
      {tab === "metadata" && (
        <section className="pt-4">
          <form
            className="flex flex-wrap items-center gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              void runMetaSearch(metaQuery);
            }}
          >
            <input
              type="text"
              value={metaQuery}
              onChange={(e) => setMetaQuery(e.target.value)}
              placeholder="name, gene, family, or accession…"
              spellCheck={false}
              className="w-full max-w-[420px] rounded border border-bd bg-surface px-2.5 py-1.5 text-[13px] text-ink placeholder:text-ink3 outline-none focus:border-bds"
            />
            <button
              type="submit"
              disabled={!metaQuery.trim() || metaLoading}
              className="rounded border border-bds bg-surface2 px-3 py-1.5 font-mono text-[12px] text-ink transition-colors hover:border-accent disabled:cursor-default disabled:opacity-40"
            >
              search
            </button>
            <button
              type="button"
              onClick={() => {
                setMetaQuery("hemoglobin");
                void runMetaSearch("hemoglobin");
              }}
              className="font-mono text-[12px] text-ink3 underline decoration-bd underline-offset-4 hover:text-accent"
            >
              try hemoglobin
            </button>
          </form>

          <div className="pt-4">
            {metaLoading ? (
              <TableSkeleton />
            ) : metaError ? (
              <ErrorPanel message={metaError} />
            ) : metaResults === null ? (
              <QuietNotice>
                Matches names, genes, families, and accessions in the corpus.
              </QuietNotice>
            ) : metaResults.length === 0 ? (
              <div className="space-y-3">
                <QuietNotice>No matches in the corpus</QuietNotice>
                {isLive &&
                  /^[A-Za-z][A-Za-z0-9]{5,9}$/.test(metaQuery.trim()) &&
                  !fetched && (
                    <button
                      type="button"
                      disabled={fetchLoading}
                      onClick={() => void runFetch(metaQuery.trim().toUpperCase())}
                      className="rounded border border-bds bg-surface2 px-3 py-1.5 font-mono text-[12px] text-ink hover:border-accent disabled:opacity-40"
                    >
                      {fetchLoading
                        ? "fetching + embedding…"
                        : `fetch ${metaQuery.trim().toUpperCase()} from UniProt`}
                    </button>
                  )}
                {fetchError && <ErrorPanel message={fetchError} />}
              </div>
            ) : (
              <div className="panel px-1 py-1">
                <HitTable hits={summariesToHits(metaResults)} showSimilarity={false} />
              </div>
            )}

            {fetched && (
              <div className="mt-4 space-y-3">
                <div className="panel flex flex-wrap items-baseline gap-x-3 gap-y-1 px-3 py-2">
                  <span className="rounded border border-bds px-1.5 py-0.5 font-mono text-[10px] uppercase text-ink3">
                    {fetched.source === "uniprot" ? "external · UniProt" : "in corpus"}
                  </span>
                  <span className="font-mono text-[12px] text-accent">
                    {fetched.protein.accession}
                  </span>
                  <span className="min-w-0 flex-1 truncate text-[13px] text-ink">
                    {fetched.protein.name}
                  </span>
                  <span className="font-mono text-[11px] text-ink3">
                    {fetched.protein.organism} · {fetched.protein.length} aa
                  </span>
                </div>
                <p className="label-mono">nearest corpus proteins</p>
                <div className="panel px-1 py-1">
                  <HitTable hits={fetched.hits} showSimilarity />
                </div>
                <p className="max-w-2xl text-[11px] text-ink3">{fetched.note}</p>
              </div>
            )}
          </div>
        </section>
      )}

      {/* ------------------------------------------------ sequence tab --- */}
      {tab === "sequence" && (
        <section className="pt-4">
          {!isLive ? (
            <QuietNotice>{LIVE_ONLY_NOTICE}</QuietNotice>
          ) : (
            <>
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  void runSeqSearch();
                }}
              >
                <div className="flex items-baseline justify-between">
                  <label htmlFor="seq-input" className="label-mono">
                    amino-acid sequence
                  </label>
                  <span className="font-mono text-[11px] text-ink3 tabular">
                    {cleaned.length} aa
                  </span>
                </div>
                <textarea
                  id="seq-input"
                  value={seqRaw}
                  onChange={(e) => setSeqRaw(e.target.value)}
                  rows={5}
                  spellCheck={false}
                  placeholder="Paste a sequence (10–2048 aa, standard residues)…"
                  className="mt-1 w-full resize-y rounded border border-bd bg-surface px-2.5 py-2 font-mono text-[12px] leading-5 text-ink placeholder:text-ink3 outline-none focus:border-bds scroll-thin"
                />
                {seqValidation && (
                  <p className="pt-1 font-mono text-[11px] text-[#e66767]">
                    {seqValidation}
                  </p>
                )}

                <div className="flex flex-wrap items-center gap-3 pt-3">
                  <label className="flex items-center gap-1.5">
                    <span className="label-mono">k</span>
                    <select
                      value={k}
                      onChange={(e) => setK(Number(e.target.value))}
                      className={selectClass}
                    >
                      {K_OPTIONS.map((n) => (
                        <option key={n} value={n}>
                          {n}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="flex items-center gap-1.5">
                    <span className="label-mono">pooling</span>
                    <select
                      value={pooling}
                      onChange={(e) => setPooling(e.target.value as Pooling)}
                      className={selectClass}
                    >
                      {POOLINGS.map((p) => (
                        <option key={p} value={p}>
                          {p}
                        </option>
                      ))}
                    </select>
                  </label>
                  <button
                    type="submit"
                    disabled={!seqSubmittable || seqLoading}
                    className="rounded border border-bds bg-surface2 px-3 py-1.5 font-mono text-[12px] text-ink transition-colors hover:border-accent disabled:cursor-default disabled:opacity-40"
                  >
                    {seqLoading ? "searching…" : "search"}
                  </button>
                  <button
                    type="button"
                    onClick={() => setSeqRaw(EXAMPLE_SEQUENCE)}
                    className="font-mono text-[12px] text-ink3 underline decoration-bd underline-offset-4 hover:text-accent"
                  >
                    example sequence
                  </button>
                </div>
              </form>

              <div className="pt-4">
                {seqLoading ? (
                  <TableSkeleton />
                ) : seqNeedsLive ? (
                  <QuietNotice>{LIVE_ONLY_NOTICE}</QuietNotice>
                ) : seqError ? (
                  <ErrorPanel message={seqError} />
                ) : seqHits === null ? null : seqHits.length === 0 ? (
                  <QuietNotice>No matches</QuietNotice>
                ) : (
                  <div className="space-y-2">
                    <div className="flex justify-end">
                      <DownloadJson payload={seqHits} filename="proteinlens-search.json" />
                    </div>
                    <div className="panel px-1 py-1">
                      <HitTable hits={seqHits} showSimilarity />
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
        </section>
      )}
    </div>
  );
}

export default function SearchPage() {
  return (
    <Suspense
      fallback={
        <div className="mx-auto max-w-[1100px] px-4 py-6">
          <p className="loading-pulse font-mono text-xs text-ink3">loading…</p>
        </div>
      }
    >
      <SearchWorkbench />
    </Suspense>
  );
}
