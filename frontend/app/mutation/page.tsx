"use client";

/** Mutation workbench: pick a protein and a position, run the 19-substitution
 *  landscape, then drill into a single substitution. Live mode runs ESM
 *  forward passes through the API; demo mode serves precomputed landscapes
 *  for showcase positions and quietly points everything else at the live API.
 */

import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";
import {
  METRIC_LABEL,
  MutationHeatmap,
  type LandscapeMetric,
} from "@/components/mutation-heatmap";
import { ResidueTrack } from "@/components/residue-track";
import { SequenceViewer } from "@/components/sequence-viewer";
import { ProteinFinder } from "@/components/protein-finder";
import { TrajectoryPanel } from "@/components/trajectory-panel";
import {
  analyzeMutation,
  ApiError,
  demoLandscapePositions,
  findProteins,
  getProfile,
  isLive,
  mutationLandscape,
} from "@/lib/data";
import type {
  Landscape,
  MutationResult,
  Pooling,
  ProteinProfile,
  ProteinSummary,
} from "@/lib/types";

const POOLINGS: Pooling[] = ["mean", "max", "bos", "attention"];
const LIVE_NOTICE = "Requires the live API — run the backend locally.";

const selectCls =
  "rounded border border-bd bg-surface px-2 py-1 font-mono text-[11px] text-ink2 outline-none focus:border-bds disabled:cursor-default disabled:opacity-40";
const buttonCls =
  "rounded border border-bds bg-surface2 px-3 py-1.5 font-mono text-[11px] text-ink transition-colors hover:border-accent disabled:cursor-default disabled:opacity-40 disabled:hover:border-bds";

function errMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------

function TrajectorySchematic({
  mutation,
  result,
}: {
  mutation: string;
  result: MutationResult;
}) {
  return (
    <div className="pt-2">
      <div className="label-mono pb-1">representation trajectory</div>
      <svg
        viewBox="0 0 360 100"
        role="img"
        aria-label={`Schematic: wild-type and ${mutation} embeddings separated by displacement ${result.displacement.toPrecision(3)}`}
        className="block w-full max-w-[400px]"
      >
        <defs>
          <marker
            id="traj-arrow"
            viewBox="0 0 8 8"
            refX="7"
            refY="4"
            markerWidth="7"
            markerHeight="7"
            orient="auto"
          >
            <path d="M0,0 L8,4 L0,8 Z" fill="#c3c2b7" />
          </marker>
        </defs>
        <line
          x1={53}
          y1={46}
          x2={303}
          y2={46}
          stroke="#c3c2b7"
          strokeWidth={1.25}
          markerEnd="url(#traj-arrow)"
        />
        <circle cx={44} cy={46} r={5} fill="#898781" />
        <circle cx={316} cy={46} r={5} fill="#3987e5" />
        <text
          x={44}
          y={70}
          textAnchor="middle"
          fill="#898781"
          fontSize={10}
          fontFamily="var(--mono)"
        >
          WT
        </text>
        <text
          x={316}
          y={70}
          textAnchor="middle"
          fill="#c3c2b7"
          fontSize={10}
          fontFamily="var(--mono)"
        >
          {mutation}
        </text>
        <text
          x={180}
          y={34}
          textAnchor="middle"
          fill="#f2f2ef"
          fontSize={11}
          fontFamily="var(--mono)"
        >
          ‖Δz‖ = {result.displacement.toPrecision(3)}
        </text>
        <text
          x={180}
          y={62}
          textAnchor="middle"
          fill="#898781"
          fontSize={10}
          fontFamily="var(--mono)"
        >
          cos = {result.cosine_similarity.toFixed(4)}
        </text>
      </svg>
      <p className="text-xs text-ink3">
        movement in embedding space (schematic, not the UMAP projection)
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------

function StatCell({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="label-mono">{label}</div>
      <div className="pt-0.5 font-mono text-[15px] text-ink tabular">
        {value.toPrecision(4)}
      </div>
    </div>
  );
}

function MutationWorkbench() {
  const search = useSearchParams();

  const [accession, setAccession] = useState<string | null>(null);
  const [profile, setProfile] = useState<ProteinProfile | null>(null);
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileError, setProfileError] = useState<string | null>(null);
  const [demoPositions, setDemoPositions] = useState<number[]>([]);

  const [position, setPosition] = useState<number | null>(null);
  const [pooling, setPooling] = useState<Pooling>("mean");

  const [landscape, setLandscape] = useState<Landscape | null>(null);
  const [landscapeLoading, setLandscapeLoading] = useState(false);
  const [landscapeError, setLandscapeError] = useState<string | null>(null);
  const [metric, setMetric] = useState<LandscapeMetric>("displacement");

  const [picked, setPicked] = useState<string | null>(null);
  const [detail, setDetail] = useState<MutationResult | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<{ status: number; message: string } | null>(null);

  const profileSeq = useRef(0);
  const detailSeq = useRef(0);
  const landscapeSeq = useRef(0);
  const booted = useRef(false);

  const clearResults = () => {
    // Invalidate any in-flight landscape: a response computed for the previous
    // protein/position must never render against the new selection.
    landscapeSeq.current += 1;
    setLandscape(null);
    setLandscapeError(null);
    setPicked(null);
    setDetail(null);
    setDetailError(null);
  };

  const pickProtein = async (acc: string, initialPos: number | null = null) => {
    const req = ++profileSeq.current;
    setAccession(acc);
    setProfile(null);
    setProfileError(null);
    setProfileLoading(true);
    setDemoPositions([]);
    setPosition(null);
    clearResults();
    try {
      const prof = await getProfile(acc);
      if (profileSeq.current !== req) return;
      setProfile(prof);
      if (
        initialPos !== null &&
        Number.isInteger(initialPos) &&
        initialPos >= 1 &&
        initialPos <= prof.sequence.length
      ) {
        setPosition(initialPos);
      }
    } catch (err) {
      if (profileSeq.current !== req) return;
      setProfileError(errMessage(err));
    } finally {
      if (profileSeq.current === req) setProfileLoading(false);
    }
    if (!isLive) {
      const positions = await demoLandscapePositions(acc);
      if (profileSeq.current === req) setDemoPositions(positions);
    }
  };

  // Arrival via ?p=ACCESSION&pos=N — run once.
  useEffect(() => {
    if (booted.current) return;
    booted.current = true;
    const p = search.get("p");
    if (!p) return;
    const posRaw = search.get("pos");
    const pos = posRaw !== null ? Number.parseInt(posRaw, 10) : Number.NaN;
    void pickProtein(p.toUpperCase(), Number.isFinite(pos) ? pos : null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectPosition = (pos: number) => {
    setPosition(pos);
    clearResults();
  };

  const computeLandscape = async () => {
    if (!accession || position === null) return;
    clearResults(); // bumps landscapeSeq, invalidating older in-flight requests
    const req = landscapeSeq.current;
    setLandscapeLoading(true);
    try {
      const result = await mutationLandscape({ accession }, position, pooling);
      if (landscapeSeq.current === req) setLandscape(result);
    } catch (err) {
      if (landscapeSeq.current === req) setLandscapeError(errMessage(err));
    } finally {
      if (landscapeSeq.current === req) setLandscapeLoading(false);
    }
  };

  const pickMutation = async (mutation: string) => {
    setPicked(mutation);
    setDetail(null);
    setDetailError(null);
    if (!isLive || !accession) return;
    const req = ++detailSeq.current;
    setDetailLoading(true);
    try {
      const result = await analyzeMutation({ accession }, mutation, pooling);
      if (detailSeq.current === req) setDetail(result);
    } catch (err) {
      if (detailSeq.current !== req) return;
      if (err instanceof ApiError) {
        setDetailError({ status: err.status, message: err.message });
      } else {
        setDetailError({ status: 0, message: errMessage(err) });
      }
    } finally {
      if (detailSeq.current === req) setDetailLoading(false);
    }
  };

  const wt = profile && position !== null ? profile.sequence[position - 1] : null;
  const demoPositionOk = position !== null && demoPositions.includes(position);
  const computeDisabled =
    !profile || position === null || landscapeLoading || (!isLive && !demoPositionOk);
  const demoEffect =
    !isLive && picked && landscape
      ? (landscape.effects.find((e) => e.mutation === picked) ?? null)
      : null;

  return (
    <div className="mx-auto max-w-[1400px] space-y-4 px-4 py-6">
      <div>
        <h1 className="text-lg font-medium text-ink">Mutation</h1>
        <p className="text-[13px] text-ink2">
          Perturb one residue and measure how the frozen ESM-2 representation moves —
          19 substitutions per site, then per-residue detail for any one of them.
        </p>
      </div>

      <div className="panel space-y-3 p-4">
        <div className="label-mono">protein</div>
        <ProteinFinder onPick={(p) => void pickProtein(p.accession)} />
        {profileLoading && (
          <div className="loading-pulse font-mono text-xs text-ink3">loading profile…</div>
        )}
        {profileError && (
          <div className="panel border-bds p-3 text-[13px] text-ink2">
            <span className="label-mono mr-2">error</span>
            {profileError}
          </div>
        )}
        {profile && (
          <div className="font-mono text-xs text-ink2">
            <span className="text-accent">{profile.protein.accession}</span>
            {" · "}
            {profile.protein_name_full || profile.protein.name}
            {profile.protein.gene ? ` · ${profile.protein.gene}` : ""}
            {" · "}
            {profile.protein.organism}
            {" · "}
            {profile.protein.length} aa
            {profile.protein.family ? ` · ${profile.protein.family}` : ""}
          </div>
        )}
      </div>

      {profile && (
        <div className="panel space-y-3 p-4">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <div className="label-mono">sequence — click a residue to set the position</div>
            {wt && position !== null && (
              <div className="font-mono text-[11px] text-accent tabular">
                position {wt}
                {position}
              </div>
            )}
          </div>
          <SequenceViewer
            sequence={profile.sequence}
            selected={position}
            onSelect={selectPosition}
          />

          {!isLive && demoPositions.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5 border-t border-bd pt-3">
              <span className="label-mono mr-1">precomputed positions</span>
              {demoPositions.map((pos) => {
                const label = `${profile.sequence[pos - 1] ?? "?"}${pos}`;
                const active = position === pos;
                return (
                  <button
                    key={pos}
                    type="button"
                    onClick={() => selectPosition(pos)}
                    className={`rounded border px-2 py-0.5 font-mono text-[11px] transition-colors ${
                      active
                        ? "border-accent text-ink"
                        : "border-bd text-ink2 hover:border-bds"
                    }`}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
          )}

          <div className="flex flex-wrap items-center gap-3 border-t border-bd pt-3">
            <label className="flex items-center gap-1.5">
              <span className="label-mono">pooling</span>
              <select
                value={pooling}
                onChange={(e) => setPooling(e.target.value as Pooling)}
                disabled={!isLive}
                title={isLive ? undefined : "Pooling is fixed in demo mode"}
                className={selectCls}
              >
                {POOLINGS.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              onClick={() => void computeLandscape()}
              disabled={computeDisabled}
              className={buttonCls}
            >
              Compute landscape
            </button>
            {landscapeLoading && (
              <span className="loading-pulse font-mono text-[11px] text-ink3">
                {isLive ? "running 19 forward passes…" : "loading precomputed landscape…"}
              </span>
            )}
            {position === null && !landscapeLoading && (
              <span className="font-mono text-[11px] text-ink3">no position selected</span>
            )}
          </div>

          {!isLive && position !== null && !demoPositionOk && (
            <p className="font-mono text-xs text-ink3">
              {wt}
              {position} is not in the precomputed demo set. {LIVE_NOTICE}
            </p>
          )}
          {!isLive && demoPositions.length === 0 && (
            <p className="font-mono text-xs text-ink3">
              No precomputed landscapes for this protein in demo mode. {LIVE_NOTICE}
            </p>
          )}
        </div>
      )}

      {landscapeError && (
        <div className="panel border-bds p-3 text-[13px] text-ink2">
          <span className="label-mono mr-2">error</span>
          {landscapeError}
        </div>
      )}

      {landscape && (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
          <div className="panel space-y-3 p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="label-mono">
                {landscape.wildtype}
                {landscape.position} · 19 substitutions · pooling {landscape.pooling}
              </div>
              <label className="flex items-center gap-1.5">
                <span className="label-mono">metric</span>
                <select
                  value={metric}
                  onChange={(e) => setMetric(e.target.value as LandscapeMetric)}
                  className={selectCls}
                >
                  <option value="displacement">displacement</option>
                  <option value="cosine_similarity">cosine similarity</option>
                  <option value="local_delta">local Δ</option>
                  <option value="llr" disabled={landscape.effects[0]?.llr === undefined}>
                    LM log-likelihood
                  </option>
                </select>
              </label>
            </div>
            <MutationHeatmap
              landscape={landscape}
              metric={metric}
              onPick={(m) => void pickMutation(m)}
              picked={picked}
            />
            <p className="text-xs text-ink3">{landscape.note}</p>
          </div>

          <div className="panel space-y-3 p-4">
            <div className="label-mono">substitution detail</div>
            {!picked && (
              <p className="font-mono text-xs text-ink3">
                pick a substitution in the landscape to inspect it
              </p>
            )}

            {picked && detailLoading && (
              <div className="loading-pulse font-mono text-xs text-ink3">
                analyzing {picked}…
              </div>
            )}

            {picked && detailError && detailError.status === 501 && (
              <p className="font-mono text-xs text-ink3">{LIVE_NOTICE}</p>
            )}
            {picked && detailError && detailError.status !== 501 && (
              <div className="panel border-bds p-3 text-[13px] text-ink2">
                <span className="label-mono mr-2">error</span>
                {detailError.message}
              </div>
            )}

            {picked && detail && (
              <div className="space-y-4">
                <div className="font-mono text-[13px] text-ink">{detail.mutation}</div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3">
                  <StatCell label="LM log-likelihood ratio" value={detail.llr} />
                  <StatCell label="‖Δz‖ displacement" value={detail.displacement} />
                  <StatCell label="relative ‖Δz‖" value={detail.relative_displacement} />
                  <StatCell label="cos(z_wt, z_mut)" value={detail.cosine_similarity} />
                  <StatCell label="local Δ (±8)" value={detail.local_delta} />
                  <StatCell label="global residue Δ" value={detail.global_residue_delta} />
                </div>
                <ResidueTrack
                  values={detail.per_residue_delta}
                  marked={position}
                  label="per-residue ‖Δh‖"
                  onClickPos={selectPosition}
                />
                <div>
                  <div className="label-mono pb-1">top dimensions</div>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 sm:grid-cols-3">
                    {detail.top_dimensions.map((d) => (
                      <span
                        key={d.dim}
                        className="font-mono text-[11px] text-ink2 tabular"
                      >
                        dim {d.dim}
                        {"  "}Δ {d.delta >= 0 ? "+" : "−"}
                        {Math.abs(d.delta).toFixed(3)}
                      </span>
                    ))}
                  </div>
                </div>
                <TrajectorySchematic mutation={detail.mutation} result={detail} />
                <p className="text-xs text-ink3">{detail.note}</p>
              </div>
            )}

            {picked && !isLive && demoEffect && (
              <div className="space-y-4">
                <div className="font-mono text-[13px] text-ink">{demoEffect.mutation}</div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3">
                  <StatCell
                    label={METRIC_LABEL.displacement}
                    value={demoEffect.displacement}
                  />
                  <StatCell
                    label={METRIC_LABEL.cosine_similarity}
                    value={demoEffect.cosine_similarity}
                  />
                  <StatCell label={METRIC_LABEL.local_delta} value={demoEffect.local_delta} />
                </div>
                <p className="font-mono text-xs text-ink3">
                  Per-residue detail needs the live API. {LIVE_NOTICE}
                </p>
                <p className="text-xs text-ink3">{landscape.note}</p>
              </div>
            )}
          </div>
        </div>
      )}

      {accession && profile && (
        <section className="mt-7 border-t border-bd pt-4">
          <div className="flex items-baseline justify-between pb-3">
            <h2 className="label-mono">Representation trajectory</h2>
            <span className="font-mono text-[10px] text-ink3">
              cumulative mutations, applied in order
            </span>
          </div>
          <TrajectoryPanel accession={accession} pooling={pooling} seedMutation={picked} />
        </section>
      )}
    </div>
  );
}

export default function MutationPage() {
  return (
    <Suspense
      fallback={
        <div className="mx-auto max-w-[1400px] px-4 py-6">
          <div className="loading-pulse font-mono text-xs text-ink3">loading…</div>
        </div>
      }
    >
      <MutationWorkbench />
    </Suspense>
  );
}
