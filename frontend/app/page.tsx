"use client";

/** Landing: hero, live miniature embedding map, corpus stats, workflow. */

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { EmbeddingMap } from "@/components/embedding-map";
import { getHealth, getMap } from "@/lib/data";
import { categoricalScale } from "@/lib/palette";
import type { Health, MapPayload, MapPoint } from "@/lib/types";

export default function LandingPage() {
  const router = useRouter();
  const [map, setMap] = useState<MapPayload | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [mapError, setMapError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const [m, h] = await Promise.allSettled([getMap("mean"), getHealth()]);
      if (cancelled) return;
      if (m.status === "fulfilled") setMap(m.value);
      else setMapError(m.reason instanceof Error ? m.reason.message : String(m.reason));
      if (h.status === "fulfilled") setHealth(h.value);
      setLoading(false);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const scale = useMemo(
    () => (map ? categoricalScale(map.points.map((p) => p.family)) : null),
    [map],
  );
  const colorOf = useMemo(
    () => (scale ? (p: MapPoint) => scale.colorOf(p.family) : null),
    [scale],
  );

  const corpusSize = health?.corpus_size ?? map?.points.length ?? null;
  const poolingCount = health?.poolings.length ?? null;

  return (
    <div className="mx-auto max-w-[1100px] px-4">
      {/* Hero ------------------------------------------------------------- */}
      <section className="max-w-3xl pb-12 pt-16">
        <p className="label-mono mb-4">ProteinLens</p>
        <h1 className="text-3xl font-semibold leading-tight tracking-tight text-ink md:text-4xl">
          Explore the geometry of protein representations.
        </h1>
        <p className="mt-4 max-w-xl leading-relaxed text-ink2">
          A frozen ESM-2 encoder over a Swiss-Prot corpus: per-protein embeddings, exact
          nearest-neighbor retrieval, single-substitution mutation perturbation, and
          benchmarks against sequence baselines.
        </p>
        <div className="mt-7 flex items-center gap-3">
          <Link
            href="/explorer"
            className="rounded bg-accent px-3.5 py-1.5 text-[13px] font-medium text-white transition-opacity hover:opacity-90"
          >
            Explore proteins
          </Link>
          <Link
            href="/search"
            className="rounded border border-bds px-3.5 py-1.5 text-[13px] text-ink transition-colors hover:bg-surface"
          >
            Search sequence
          </Link>
        </div>
      </section>

      {/* Miniature embedding map ------------------------------------------ */}
      <section className="pb-8">
        {loading && (
          <div className="panel">
            <div className="loading-pulse h-[420px] rounded-md bg-surface2" />
          </div>
        )}
        {!loading && mapError && (
          <div className="panel p-4 text-[13px] text-ink2">
            <span className="label-mono mr-3">map unavailable</span>
            {mapError}
          </div>
        )}
        {!loading && map && colorOf && scale && (
          <div className="panel overflow-hidden">
            <div className="flex items-center justify-between border-b border-bd px-3 py-2">
              <span className="label-mono">Embedding map — mean pooling</span>
              <span className="font-mono text-[11px] text-ink3">colored by family</span>
            </div>
            {/* Capture-phase stops disable zoom/pan; hover + click stay live. */}
            <div
              className="h-[420px]"
              onWheelCapture={(e) => e.stopPropagation()}
              onMouseDownCapture={(e) => e.stopPropagation()}
              onDoubleClickCapture={(e) => e.stopPropagation()}
              onTouchStartCapture={(e) => e.stopPropagation()}
            >
              <EmbeddingMap
                points={map.points}
                colorOf={colorOf}
                onSelect={(p) => router.push(`/protein/${p.id}`)}
                className="h-full w-full"
              />
            </div>
            <div className="space-y-1.5 border-t border-bd px-3 py-2">
              <p className="font-mono text-[11px] text-ink3">
                {map.points.length.toLocaleString()} proteins · {map.model} · PCA → UMAP of
                mean-pooled embeddings · click a point to open its profile
              </p>
              <div className="flex flex-wrap gap-x-3 gap-y-1">
                {scale.legend.map((item) => (
                  <span
                    key={item.label}
                    className="flex items-center gap-1.5 font-mono text-[11px] text-ink2"
                  >
                    <span
                      className="inline-block h-2 w-2 rounded-full"
                      style={{ background: item.color }}
                    />
                    <span className="max-w-44 truncate">{item.label}</span>
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </section>

      {/* Stats row --------------------------------------------------------- */}
      <section className="pb-12">
        <div className="panel grid grid-cols-2 sm:grid-cols-4">
          {[
            {
              label: "corpus",
              value: corpusSize !== null ? corpusSize.toLocaleString() : "—",
              unit: "proteins",
            },
            { label: "model params", value: "35M", unit: "esm2_t12" },
            { label: "embedding dim", value: "480", unit: "per pooling" },
            {
              label: "poolings",
              value: poolingCount !== null ? String(poolingCount) : "—",
              unit: "mean·max·bos·attn",
            },
          ].map((s, i) => (
            <div
              key={s.label}
              className={`px-4 py-3 ${i > 0 ? "border-l border-bd" : ""}`}
            >
              <p className="label-mono">{s.label}</p>
              <p className="tabular mt-1 font-mono text-xl text-ink">{s.value}</p>
              <p className="font-mono text-[10px] text-ink3">{s.unit}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Signature workflow ------------------------------------------------ */}
      <section className="border-t border-bd py-10">
        <h2 className="label-mono mb-6">Signature workflow</h2>
        <ol className="grid gap-8 md:grid-cols-3">
          {[
            {
              n: "01",
              href: "/search",
              title: "Search",
              body: "Find a protein by accession or gene, or paste a raw sequence to embed it.",
            },
            {
              n: "02",
              href: "/explorer",
              title: "Inspect neighbors",
              body: "Open its profile, then examine its nearest neighbors on the embedding map.",
            },
            {
              n: "03",
              href: "/mutation",
              title: "Mutate a residue",
              body: "Substitute one position and measure how far the representation moves.",
            },
          ].map((step) => (
            <li key={step.n}>
              <span className="font-mono text-[11px] text-ink3">{step.n}</span>
              <Link
                href={step.href}
                className="mt-1 block text-[13px] font-medium text-ink hover:text-accent"
              >
                {step.title} →
              </Link>
              <p className="mt-1 text-[13px] leading-relaxed text-ink2">{step.body}</p>
            </li>
          ))}
        </ol>
        <p className="mt-8 max-w-3xl text-[12.5px] leading-relaxed text-ink3">
          Deeper cuts: <Link href="/clusters" className="text-accent hover:underline">representation
          clusters</Link> with per-cluster composition,{" "}
          <Link href="/compare" className="text-accent hover:underline">pairwise comparison</Link>{" "}
          of identity vs cosine across poolings, curated domain tracks with region-level
          search on every profile, and{" "}
          <Link href="/benchmarks" className="text-accent hover:underline">extended benchmarks</Link>{" "}
          covering ESM-2 checkpoint scaling, contrastive pooler training, and a
          structure-aware ProstT5 reference.
        </p>
      </section>

      {/* Footer ------------------------------------------------------------ */}
      <footer className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-bd py-6 text-[12px] text-ink3">
        <span>Data: UniProtKB/Swiss-Prot (CC BY 4.0)</span>
        <span className="font-mono text-[11px]">facebook/esm2_t12_35M_UR50D</span>
        <span className="ml-auto flex gap-4">
          <Link href="/about" className="hover:text-ink">
            About
          </Link>
          <a
            href="https://github.com/sunkalpchandra/proteinlens"
            target="_blank"
            rel="noreferrer"
            className="hover:text-ink"
          >
            GitHub
          </a>
        </span>
      </footer>
    </div>
  );
}
