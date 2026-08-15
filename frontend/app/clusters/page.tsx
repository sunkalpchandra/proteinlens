"use client";

/** Cluster browser: k-means structure of the representation space with
 *  per-cluster composition. Cluster ids match the explorer's cluster color
 *  mode; membership is geometric (nearest centroid), and dominant families or
 *  organisms describe composition — not a functional claim about clusters. */

import Link from "next/link";
import { useEffect, useState } from "react";
import { getClusters } from "@/lib/data";
import type { ClustersPayload, ClusterSummary } from "@/lib/types";

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : "Failed to load clusters.";
}

function TopChips({ counts }: { counts: Record<string, number> }) {
  const entries = Object.entries(counts).slice(0, 3);
  if (!entries.length) return <span className="text-[11px] text-ink3">—</span>;
  return (
    <span className="flex flex-wrap gap-1">
      {entries.map(([label, count]) => (
        <span
          key={label}
          title={label}
          className="max-w-[220px] truncate rounded border border-bd bg-surface2 px-1.5 py-0.5 font-mono text-[10px] text-ink2"
        >
          {label} <span className="text-ink3">({count})</span>
        </span>
      ))}
    </span>
  );
}

type Algorithm = "kmeans" | "hdbscan";

export default function ClustersPage() {
  const [algorithm, setAlgorithm] = useState<Algorithm>("kmeans");
  const [hdbscanAvailable, setHdbscanAvailable] = useState(true);
  const [payload, setPayload] = useState<ClustersPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stale = false;
    setPayload(null);
    setError(null);
    getClusters("mean", algorithm)
      .then((p) => !stale && setPayload(p))
      .catch((e: unknown) => {
        if (stale) return;
        if (algorithm === "hdbscan") {
          // View not built for this deployment — fall back quietly.
          setHdbscanAvailable(false);
          setAlgorithm("kmeans");
        } else {
          setError(errorText(e));
        }
      });
    return () => {
      stale = true;
    };
  }, [algorithm]);

  if (error) {
    return (
      <div className="mx-auto max-w-5xl px-4 py-8">
        <div className="panel px-4 py-3 text-[13px] text-ink2">{error}</div>
      </div>
    );
  }
  if (!payload) {
    return (
      <div className="mx-auto max-w-5xl px-4 py-8">
        <p className="loading-pulse font-mono text-xs text-ink3">loading clusters…</p>
      </div>
    );
  }

  const maxSize = Math.max(...payload.clusters.map((c) => c.size), 1);
  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <div className="flex flex-wrap items-baseline justify-between gap-2 pb-4">
        <h1 className="text-lg font-semibold text-ink">Representation clusters</h1>
        <span className="flex items-center gap-3 font-mono text-[11px] text-ink3">
          <span className="flex overflow-hidden rounded border border-bd">
            {(["kmeans", "hdbscan"] as const).map((a) => (
              <button
                key={a}
                type="button"
                disabled={a === "hdbscan" && !hdbscanAvailable}
                onClick={() => setAlgorithm(a)}
                className={`px-2 py-0.5 disabled:opacity-40 ${
                  algorithm === a ? "bg-surface2 text-ink" : "text-ink3 hover:text-ink2"
                }`}
              >
                {a}
              </button>
            ))}
          </span>
          <span>
            {algorithm === "kmeans"
              ? `k=${payload.clustering.n_clusters ?? payload.clusters.length} · silhouette ${payload.clustering.silhouette_cosine?.toFixed(3) ?? "—"}`
              : `${payload.clustering.n_clusters ?? "—"} density islands · unassigned bulk labeled noise`}
          </span>
          <Link href="/explorer?color=cluster" className="text-accent hover:underline">
            map →
          </Link>
        </span>
      </div>

      <div className="panel divide-y divide-bd">
        {payload.clusters.map((c: ClusterSummary) => (
          <div key={c.cluster} className="grid grid-cols-[44px_1fr] gap-3 px-3 py-2.5 sm:grid-cols-[44px_130px_1fr_1fr]">
            <span className="font-mono text-[13px] text-ink tabular">
              {c.cluster === -1 ? "noise" : c.cluster}
            </span>
            <span className="flex flex-col gap-1">
              <span className="font-mono text-[11px] text-ink2 tabular">
                {c.size.toLocaleString()} proteins
              </span>
              <span className="block h-[3px] w-full max-w-[110px] overflow-hidden rounded-full bg-surface2">
                <span
                  className="block h-full rounded-full bg-accent"
                  style={{ width: `${(c.size / maxSize) * 100}%` }}
                />
              </span>
              <span className="font-mono text-[10px] text-ink3 tabular">
                ~{Math.round(c.mean_length)} aa · {(c.enzyme_fraction * 100).toFixed(0)}% enzyme
              </span>
            </span>
            <span className="col-span-2 sm:col-span-1">
              <span className="label-mono block pb-1 text-[9px]">families</span>
              <TopChips counts={c.top_families} />
            </span>
            <span className="col-span-2 sm:col-span-1">
              <span className="label-mono block pb-1 text-[9px]">organisms</span>
              <TopChips counts={c.top_organisms} />
            </span>
          </div>
        ))}
      </div>
      <p className="max-w-3xl pt-3 text-[11px] leading-relaxed text-ink3">
        Clusters are unsupervised structure in the embedding geometry. Dominant families
        and organisms describe what each cluster happens to contain in this corpus — they
        are composition statistics, not functional assignments.
      </p>
    </div>
  );
}
