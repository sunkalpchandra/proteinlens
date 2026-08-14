"use client";

/** Explorer: full-viewport embedding map with color modes, a protein finder,
 *  and a neighbor side panel.
 *
 *  All data flows through lib/data.ts. Selection syncs to ?p=ACCESSION
 *  (read once on mount, written via router.replace). The attention pooling
 *  option quietly falls back to mean when no attention map exists.
 */

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { EmbeddingMap } from "@/components/embedding-map";
import { MapLegend } from "@/components/map-legend";
import { buildColorScale, COLOR_MODES, type ColorMode } from "@/lib/color-modes";
import {
  ApiError,
  availableMapPoolings,
  findProteins,
  getMap,
  searchByAccession,
} from "@/lib/data";
import type {
  MapPayload,
  MapPreset,
  MapPoint,
  Pooling,
  ProteinSummary,
  SearchHit,
} from "@/lib/types";

const LIVE_NOTICE = "Requires the live API — run the backend locally.";
const NEIGHBOR_K = 15;
const EMPTY_POINTS: MapPoint[] = [];

const SELECT_CLS =
  "rounded border border-bd bg-surface2 px-2 py-1 font-mono text-[12px] text-ink2 focus:border-bds focus:outline-none";

function errorMessage(e: unknown): string {
  if (e instanceof ApiError && e.status === 501) return LIVE_NOTICE;
  return e instanceof Error ? e.message : String(e);
}

export default function ExplorerPage() {
  const router = useRouter();

  const [pooling, setPooling] = useState<Pooling>("mean");
  const [attentionAvailable, setAttentionAvailable] = useState(true);
  const [preset, setPreset] = useState<MapPreset>("default");
  const [presetAvailable, setPresetAvailable] = useState(true);
  const [map, setMap] = useState<MapPayload | null>(null);
  const [mapLoading, setMapLoading] = useState(true);
  const [mapError, setMapError] = useState<string | null>(null);

  const [colorMode, setColorMode] = useState<ColorMode>("family");

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [focusId, setFocusId] = useState<string | null>(null);
  const [neighbors, setNeighbors] = useState<SearchHit[] | null>(null);
  const [neighborsLoading, setNeighborsLoading] = useState(false);
  const [neighborsError, setNeighborsError] = useState<string | null>(null);

  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<ProteinSummary[]>([]);
  const [finderOpen, setFinderOpen] = useState(false);

  // --- URL → state (once, on mount) ----------------------------------------
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const p = params.get("p");
    if (p) {
      setSelectedId(p);
      setFocusId(p);
    }
    const color = params.get("color");
    if (color && COLOR_MODES.some((mode) => mode.id === color)) {
      setColorMode(color as ColorMode);
    }
  }, []);

  // --- pooling availability --------------------------------------------------
  useEffect(() => {
    let stale = false;
    availableMapPoolings()
      .then((ps) => {
        if (!stale && !ps.includes("attention")) setAttentionAvailable(false);
      })
      .catch(() => {
        /* keep enabled; the load-time fallback below still applies */
      });
    return () => {
      stale = true;
    };
  }, []);

  // --- map load ----------------------------------------------------------------
  useEffect(() => {
    let stale = false;
    setMapLoading(true);
    setMapError(null);
    getMap(pooling, preset)
      .then((payload) => {
        if (stale) return;
        setMap(payload);
        setMapLoading(false);
      })
      .catch((e: unknown) => {
        if (stale) return;
        if (pooling === "attention") {
          // No attention projection: disable the option and fall back.
          setAttentionAvailable(false);
          setPooling("mean");
        } else if (preset !== "default") {
          // Preset not built for this deployment — fall back quietly.
          setPresetAvailable(false);
          setPreset("default");
        } else {
          setMapError(errorMessage(e));
          setMapLoading(false);
        }
      });
    return () => {
      stale = true;
    };
  }, [pooling, preset]);

  // --- neighbors for the selection ---------------------------------------------
  useEffect(() => {
    if (!selectedId) {
      setNeighbors(null);
      setNeighborsError(null);
      setNeighborsLoading(false);
      return;
    }
    let stale = false;
    setNeighbors(null);
    setNeighborsError(null);
    setNeighborsLoading(true);
    searchByAccession(selectedId, pooling, NEIGHBOR_K)
      .then((h) => {
        if (stale) return;
        setNeighbors(h);
        setNeighborsLoading(false);
      })
      .catch((e: unknown) => {
        if (stale) return;
        setNeighborsError(errorMessage(e));
        setNeighborsLoading(false);
      });
    return () => {
      stale = true;
    };
  }, [selectedId, pooling]);

  // --- finder (debounced) ---------------------------------------------------------
  useEffect(() => {
    const q = query.trim();
    if (!q) {
      setHits([]);
      return;
    }
    let stale = false;
    const timer = setTimeout(() => {
      findProteins(q, 8)
        .then((r) => {
          if (!stale) setHits(r);
        })
        .catch(() => {
          if (!stale) setHits([]);
        });
    }, 150);
    return () => {
      stale = true;
      clearTimeout(timer);
    };
  }, [query]);

  // --- selection helpers ------------------------------------------------------------
  const selectProtein = useCallback(
    (id: string, focus = false) => {
      setSelectedId(id);
      setFocusId(focus ? id : null);
      router.replace(`/explorer?p=${encodeURIComponent(id)}`, { scroll: false });
    },
    [router],
  );

  const clearSelection = useCallback(() => {
    setSelectedId(null);
    setFocusId(null);
    router.replace("/explorer", { scroll: false });
  }, [router]);

  // --- derived ------------------------------------------------------------------------
  /** accession → cosine sim; includes the selection itself at 1.0 so the
   *  similarity color mode paints it as the reference point. */
  const neighborSims = useMemo(() => {
    if (!selectedId || !neighbors) return null;
    const m = new Map<string, number>();
    m.set(selectedId, 1);
    for (const h of neighbors) {
      if (h.protein.accession !== selectedId) m.set(h.protein.accession, h.similarity);
    }
    return m;
  }, [neighbors, selectedId]);

  const neighborRows = useMemo(
    () => neighbors?.filter((h) => h.protein.accession !== selectedId) ?? null,
    [neighbors, selectedId],
  );

  const points = map?.points ?? EMPTY_POINTS;

  const scale = useMemo(
    () => buildColorScale(points, colorMode, neighborSims ?? undefined),
    [points, colorMode, neighborSims],
  );

  const selectedPoint = useMemo(() => {
    if (!selectedId || !map) return null;
    return map.points.find((p) => p.id === selectedId) ?? null;
  }, [selectedId, map]);

  const modeLabel =
    COLOR_MODES.find((m) => m.id === colorMode)?.label ?? colorMode;

  const showFinderDropdown = finderOpen && query.trim() !== "" && hits.length > 0;

  return (
    <div className="flex h-[calc(100vh-48px)]">
      {/* ------------------------------------------------------------- map */}
      <div className="relative min-w-0 flex-1">
        {/* controls overlay */}
        <div className="absolute left-3 top-3 z-20 flex flex-wrap items-center gap-2">
          <select
            value={colorMode}
            onChange={(e) => setColorMode(e.target.value as ColorMode)}
            className={SELECT_CLS}
            aria-label="Color mode"
          >
            {COLOR_MODES.map((m) => (
              <option key={m.id} value={m.id}>
                color: {m.label}
              </option>
            ))}
          </select>

          <select
            value={pooling}
            onChange={(e) => setPooling(e.target.value as Pooling)}
            className={SELECT_CLS}
            aria-label="Pooling"
          >
            <option value="mean">pooling: mean</option>
            <option value="attention" disabled={!attentionAvailable}>
              pooling: attention{attentionAvailable ? "" : " (n/a)"}
            </option>
          </select>

          <select
            value={preset}
            onChange={(e) => setPreset(e.target.value as MapPreset)}
            className={SELECT_CLS}
            aria-label="UMAP neighborhood preset"
            title="UMAP neighborhood: local sharpens fine structure, global preserves broad layout"
            disabled={pooling !== "mean"}
          >
            <option value="default">umap: balanced</option>
            <option value="local" disabled={!presetAvailable}>
              umap: local (n=5)
            </option>
            <option value="global" disabled={!presetAvailable}>
              umap: global (n=50)
            </option>
          </select>

          <div className="relative">
            <input
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setFinderOpen(true);
              }}
              onFocus={() => setFinderOpen(true)}
              onBlur={() => setFinderOpen(false)}
              placeholder="find protein…"
              spellCheck={false}
              aria-label="Find protein"
              className="w-56 rounded border border-bd bg-surface2 px-2 py-1 font-mono text-[12px] text-ink placeholder:text-ink3 focus:border-bds focus:outline-none"
            />
            {showFinderDropdown && (
              <div className="absolute left-0 top-full z-30 mt-1 w-72 overflow-hidden rounded border border-bds bg-surface2 shadow-xl">
                {hits.map((p) => (
                  <button
                    key={p.accession}
                    type="button"
                    onMouseDown={(e) => {
                      // mousedown so the row wins over the input's blur
                      e.preventDefault();
                      setQuery("");
                      setHits([]);
                      setFinderOpen(false);
                      selectProtein(p.accession, true);
                    }}
                    className="block w-full px-2.5 py-1.5 text-left hover:bg-surface"
                  >
                    <div className="flex items-baseline justify-between gap-2">
                      <span className="min-w-0 truncate text-[12px] text-ink">
                        {p.name}
                      </span>
                      <span className="shrink-0 font-mono text-[10px] text-ink3">
                        {p.accession}
                      </span>
                    </div>
                    <div className="truncate font-mono text-[10px] text-ink3">
                      {p.organism}
                      {p.gene ? ` · ${p.gene}` : ""}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* map / error */}
        {mapError ? (
          <div className="flex h-full items-center justify-center p-6">
            <div className="panel max-w-md px-4 py-3 text-[13px] text-ink2">
              {mapError}
            </div>
          </div>
        ) : map ? (
          <EmbeddingMap
            points={map.points}
            colorOf={scale.colorOf}
            selectedId={selectedId}
            neighborIds={neighborSims}
            onSelect={(p) => selectProtein(p.id)}
            focusId={focusId}
            className="h-full w-full"
          />
        ) : null}

        {mapLoading && (
          <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
            <span className="loading-pulse font-mono text-[12px] text-ink3">
              loading map…
            </span>
          </div>
        )}

        {/* legend / similarity hint */}
        {colorMode === "similarity" && !selectedId ? (
          <div className="absolute bottom-3 left-3 z-10 rounded border border-bd bg-surface/90 px-2.5 py-1.5 font-mono text-[11px] text-ink3 backdrop-blur">
            similarity coloring needs a selected protein
          </div>
        ) : (
          <MapLegend title={modeLabel} legend={scale.legend} kind={scale.kind} />
        )}
      </div>

      {/* ------------------------------------------------------- side panel */}
      <aside className="scroll-thin flex w-[320px] shrink-0 flex-col overflow-y-auto border-l border-bd bg-surface">
        {/* corpus header */}
        {map ? (
          <div className="border-b border-bd px-3 py-2 font-mono text-[10px] leading-relaxed">
            <div className="flex justify-between gap-3">
              <span className="text-ink3">corpus</span>
              <span className="text-ink2 tabular">
                {map.points.length.toLocaleString()} proteins
              </span>
            </div>
            <div className="flex justify-between gap-3">
              <span className="text-ink3">model</span>
              <span className="min-w-0 truncate text-ink2" title={map.model}>
                {map.model}
              </span>
            </div>
            {typeof map.projection.pca_explained_variance === "number" && (
              <div className="flex justify-between gap-3">
                <span className="text-ink3">pca var</span>
                <span className="text-ink2 tabular">
                  {(map.projection.pca_explained_variance * 100).toFixed(1)}%
                </span>
              </div>
            )}
            {typeof map.clustering.silhouette_cosine === "number" && (
              <div className="flex justify-between gap-3">
                <span className="text-ink3">silhouette</span>
                <span className="text-ink2 tabular">
                  {map.clustering.silhouette_cosine.toFixed(3)}
                </span>
              </div>
            )}
          </div>
        ) : mapLoading ? (
          <div className="border-b border-bd px-3 py-2">
            <span className="loading-pulse font-mono text-[10px] text-ink3">
              loading…
            </span>
          </div>
        ) : null}

        {selectedId ? (
          <>
            {/* selection metadata */}
            <div className="border-b border-bd px-3 py-3">
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="truncate text-[13px] font-medium text-ink">
                    {selectedPoint?.name ?? selectedId}
                  </div>
                  <Link
                    href={`/protein/${encodeURIComponent(selectedId)}`}
                    className="font-mono text-[11px] text-accent hover:underline"
                  >
                    {selectedId}
                  </Link>
                </div>
                <button
                  type="button"
                  onClick={clearSelection}
                  className="shrink-0 rounded border border-bd px-1.5 py-0.5 font-mono text-[10px] text-ink3 hover:border-bds hover:text-ink2"
                >
                  clear
                </button>
              </div>

              {selectedPoint && (
                <dl className="mt-2 grid grid-cols-[64px_1fr] gap-x-3 gap-y-1 text-[11px]">
                  <dt className="font-mono text-[10px] uppercase text-ink3">org</dt>
                  <dd className="truncate text-ink2">{selectedPoint.org}</dd>
                  <dt className="font-mono text-[10px] uppercase text-ink3">family</dt>
                  <dd className="truncate text-ink2">
                    {selectedPoint.family ?? "—"}
                  </dd>
                  <dt className="font-mono text-[10px] uppercase text-ink3">length</dt>
                  <dd className="text-ink2 tabular">{selectedPoint.len} aa</dd>
                  <dt className="font-mono text-[10px] uppercase text-ink3">cluster</dt>
                  <dd className="text-ink2 tabular">{selectedPoint.cluster}</dd>
                  <dt className="font-mono text-[10px] uppercase text-ink3">outlier</dt>
                  <dd className="text-ink2 tabular">
                    {selectedPoint.outlier.toFixed(3)}
                  </dd>
                </dl>
              )}

              <Link
                href={`/protein/${encodeURIComponent(selectedId)}`}
                className="mt-2 inline-block font-mono text-[11px] text-accent hover:underline"
              >
                Open profile -&gt;
              </Link>
            </div>

            {/* neighbors */}
            <div className="flex-1 pb-3">
              <div className="label-mono px-3 pb-1 pt-3">
                nearest neighbors · {pooling}
              </div>
              {neighborsLoading && (
                <div className="px-3 py-2">
                  <span className="loading-pulse font-mono text-[11px] text-ink3">
                    loading neighbors…
                  </span>
                </div>
              )}
              {neighborsError && (
                <div className="panel mx-3 my-2 px-2.5 py-2 text-[11px] text-ink2">
                  {neighborsError}
                </div>
              )}
              {neighborRows?.map((h) => (
                <button
                  key={h.protein.accession}
                  type="button"
                  onClick={() => selectProtein(h.protein.accession, true)}
                  className="block w-full px-3 py-1.5 text-left transition-colors hover:bg-surface2"
                >
                  <div className="flex items-baseline gap-2">
                    <span className="w-4 shrink-0 font-mono text-[10px] text-ink3 tabular">
                      {h.rank}
                    </span>
                    <span className="min-w-0 flex-1 truncate text-[12px] text-ink">
                      {h.protein.name}
                    </span>
                    <span className="shrink-0 font-mono text-[11px] text-accent tabular">
                      {h.similarity.toFixed(3)}
                    </span>
                  </div>
                  <div className="flex min-w-0 gap-2 pl-6 font-mono text-[10px] text-ink3">
                    <span className="shrink-0">{h.protein.accession}</span>
                    <span className="truncate">{h.protein.organism}</span>
                  </div>
                </button>
              ))}
            </div>
          </>
        ) : (
          <div className="px-3 py-4 text-[12px] leading-relaxed text-ink3">
            Click a point on the map or use the finder to inspect a protein and
            its nearest neighbors in representation space.
          </div>
        )}
      </aside>
    </div>
  );
}
