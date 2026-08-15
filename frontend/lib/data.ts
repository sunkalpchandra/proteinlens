/** Data access layer with two backends:
 *
 *  - live:  NEXT_PUBLIC_API_URL points at the FastAPI server (full features,
 *           including ESM inference for pasted sequences and mutations).
 *  - demo:  no API configured; precomputed JSON under /public/demo serves
 *           browsing, the map, retrieval, and showcase mutation landscapes.
 *           Live-inference features are cleanly disabled.
 *
 *  Components import from here only — they never fetch directly.
 */

import type {
  AttentionPayload,
  BenchmarkPayload,
  ClustersPayload,
  ComparePayload,
  DomainsPayload,
  Health,
  Landscape,
  MapPayload,
  MapPreset,
  MutationResult,
  Pooling,
  ProteinProfile,
  ProteinSummary,
  RegionSearchPayload,
  SearchHit,
  TrajectoryPayload,
} from "./types";

const API = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? null;

// Static assets live under the base path when deployed to a subpath host
// (GitHub Pages project sites). next/link handles this automatically; raw
// fetch() calls must prefix it themselves.
const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export const isLive = API !== null;

export class ApiError extends Error {
  status: number;
  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const base = API ?? "";
  const res = await fetch(`${base}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

async function demoFile<T>(name: string): Promise<T> {
  const res = await fetch(`${BASE_PATH}/demo/${name}`);
  if (!res.ok) {
    throw new ApiError(
      res.status,
      res.status === 404
        ? "Demo dataset not published yet for this deployment — the corpus pipeline publishes it automatically; check back shortly. (Running locally? Point NEXT_PUBLIC_API_URL at the backend or run scripts/build_demo_bundle.py.)"
        : `Demo asset failed to load: ${name} (HTTP ${res.status})`,
    );
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------

export async function getHealth(): Promise<Health> {
  if (!isLive) return demoFile<Health>("health.json");
  return request<Health>("/health");
}

export async function getMap(
  pooling: Pooling = "mean",
  preset: MapPreset = "default",
): Promise<MapPayload> {
  const suffix = preset === "default" ? "" : `_${preset}`;
  if (!isLive) return demoFile<MapPayload>(`map_${pooling}${suffix}.json`);
  return request<MapPayload>(`/map?pooling=${pooling}&preset=${preset}`);
}

/** Poolings for which a map projection exists. */
export async function availableMapPoolings(): Promise<Pooling[]> {
  if (!isLive) return ["mean"];
  const health = await getHealth();
  const withMaps: Pooling[] = [];
  for (const pooling of ["mean", "attention"] as const) {
    if (health.poolings.includes(pooling)) withMaps.push(pooling);
  }
  return withMaps;
}

let demoSummaries: ProteinSummary[] | null = null;

export async function findProteins(q: string, limit = 20): Promise<ProteinSummary[]> {
  if (!isLive) {
    demoSummaries ??= await demoFile<ProteinSummary[]>("proteins.json");
    const needle = q.trim().toLowerCase();
    if (!needle) return [];
    const scored = demoSummaries
      .map((p) => {
        const name = p.name.toLowerCase();
        const gene = (p.gene ?? "").toLowerCase();
        let score = 0;
        if (p.accession.toLowerCase() === needle) score += 120;
        if (gene === needle) score += 100;
        if (name.startsWith(needle)) score += 80;
        if (name.includes(needle)) score += 40;
        if (gene.includes(needle)) score += 25;
        if ((p.family ?? "").toLowerCase().includes(needle)) score += 10;
        return { p, score };
      })
      .filter((s) => s.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
    return scored.map((s) => s.p);
  }
  return request<ProteinSummary[]>(`/proteins?q=${encodeURIComponent(q)}&limit=${limit}`);
}

export async function getProfile(
  accession: string,
  pooling: Pooling = "mean",
): Promise<ProteinProfile> {
  if (!isLive) return demoFile<ProteinProfile>(`profiles/${accession}.json`);
  return request<ProteinProfile>(`/protein/${encodeURIComponent(accession)}?pooling=${pooling}`);
}

export async function searchByAccession(
  accession: string,
  pooling: Pooling = "mean",
  k = 10,
): Promise<SearchHit[]> {
  if (!isLive) {
    const profile = await getProfile(accession, pooling);
    return profile.neighbors.slice(0, k);
  }
  const res = await request<{ hits: SearchHit[] }>("/search", {
    method: "POST",
    body: JSON.stringify({ accession, pooling, k }),
  });
  return res.hits;
}

export async function searchBySequence(
  sequence: string,
  pooling: Pooling = "mean",
  k = 10,
): Promise<SearchHit[]> {
  if (!isLive) {
    throw new ApiError(501, "Sequence search needs the live API (ESM inference).");
  }
  const res = await request<{ hits: SearchHit[] }>("/search", {
    method: "POST",
    body: JSON.stringify({ sequence, pooling, k }),
  });
  return res.hits;
}

export async function analyzeMutation(
  target: { accession?: string; sequence?: string },
  mutation: string,
  pooling: Pooling = "mean",
): Promise<MutationResult> {
  if (!isLive) {
    throw new ApiError(501, "Mutation analysis needs the live API (ESM inference).");
  }
  return request<MutationResult>("/mutation", {
    method: "POST",
    body: JSON.stringify({ ...target, mutation, pooling }),
  });
}

export async function mutationLandscape(
  target: { accession?: string; sequence?: string },
  position: number,
  pooling: Pooling = "mean",
): Promise<Landscape> {
  if (!isLive) {
    if (!target.accession) {
      throw new ApiError(501, "Landscape for pasted sequences needs the live API.");
    }
    const bundle = await demoFile<Record<string, Landscape>>(
      `landscapes/${target.accession}.json`,
    );
    const hit = bundle[String(position)];
    if (!hit) {
      throw new ApiError(
        404,
        "This position is not in the precomputed demo set. Run the API locally for arbitrary positions.",
      );
    }
    return hit;
  }
  return request<Landscape>("/mutation-landscape", {
    method: "POST",
    body: JSON.stringify({ ...target, position, pooling }),
  });
}

/** Positions with precomputed landscapes for a demo protein (demo mode only). */
export async function demoLandscapePositions(accession: string): Promise<number[]> {
  if (isLive) return [];
  try {
    const bundle = await demoFile<Record<string, Landscape>>(`landscapes/${accession}.json`);
    return Object.keys(bundle).map(Number).sort((a, b) => a - b);
  } catch {
    return [];
  }
}

export async function getAttention(accession: string): Promise<AttentionPayload> {
  if (!isLive) {
    const profile = await getProfile(accession);
    if (!profile.attention_weights) {
      throw new ApiError(404, "No precomputed attention for this demo protein.");
    }
    const weights = profile.attention_weights;
    const top = weights
      .map((w, i) => [w, i + 1] as const)
      .sort((a, b) => b[0] - a[0])
      .slice(0, 10)
      .map(([, pos]) => pos);
    return {
      accession,
      length: weights.length,
      weights,
      top_positions: top,
      note: "Learned attention-pooling weights: a model-dependent interpretability signal, not a functional-residue annotation.",
    };
  }
  return request<AttentionPayload>(`/protein/${encodeURIComponent(accession)}/attention`);
}

export async function getBenchmark(): Promise<BenchmarkPayload> {
  if (!isLive) return demoFile<BenchmarkPayload>("benchmark.json");
  return request<BenchmarkPayload>("/benchmark");
}

export async function getDomains(accession: string): Promise<DomainsPayload> {
  if (!isLive) {
    // Demo bundles inline curated domains on the profile payload.
    const profile = (await getProfile(accession)) as ProteinProfile & {
      domains?: DomainsPayload["domains"];
    };
    return {
      accession,
      length: profile.sequence.length,
      domains: profile.domains ?? [],
      note: "UniProt-curated DOMAIN features; coverage is partial.",
    };
  }
  return request<DomainsPayload>(`/protein/${encodeURIComponent(accession)}/domains`);
}

export async function regionSearch(
  accession: string,
  start: number,
  end: number,
  k = 10,
): Promise<RegionSearchPayload> {
  if (!isLive) {
    throw new ApiError(501, "Region search needs the live API (ESM inference).");
  }
  return request<RegionSearchPayload>("/region-search", {
    method: "POST",
    body: JSON.stringify({ accession, start, end, k }),
  });
}

export async function trajectoryOf(
  target: { accession?: string; sequence?: string },
  mutations: string[],
  pooling: Pooling = "mean",
): Promise<TrajectoryPayload> {
  if (!isLive) {
    throw new ApiError(501, "Trajectories need the live API (ESM inference).");
  }
  return request<TrajectoryPayload>("/trajectory", {
    method: "POST",
    body: JSON.stringify({ ...target, mutations, pooling }),
  });
}

export async function getClusters(
  pooling: Pooling = "mean",
  algorithm: "kmeans" | "hdbscan" = "kmeans",
): Promise<ClustersPayload> {
  const suffix = algorithm === "kmeans" ? "" : "_hdbscan";
  if (!isLive) return demoFile<ClustersPayload>(`clusters_${pooling}${suffix}.json`);
  return request<ClustersPayload>(`/clusters?pooling=${pooling}&algorithm=${algorithm}`);
}

export async function compareProteins(a: string, b: string): Promise<ComparePayload> {
  if (!isLive) {
    throw new ApiError(501, "Pairwise comparison needs the live API.");
  }
  return request<ComparePayload>("/compare", {
    method: "POST",
    body: JSON.stringify({ a, b }),
  });
}

export const LIVE_NOTICE = "Requires the live API — run the backend locally.";

/** Uniform user-facing message for data-layer failures: 501s become the
 *  live-API notice; everything else surfaces its own message. */
export function errorMessage(e: unknown): string {
  if (e instanceof ApiError && e.status === 501) return LIVE_NOTICE;
  return e instanceof Error ? e.message : String(e);
}
