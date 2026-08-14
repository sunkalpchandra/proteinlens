/** Mirrors of the FastAPI response schemas (api/schemas.py). */

export type Pooling = "mean" | "max" | "bos" | "attention";

export interface MapPoint {
  id: string;
  name: string;
  gene: string | null;
  org: string;
  len: number;
  family: string | null;
  pfam: string | null;
  ec: string | null;
  enzyme: boolean;
  loc: string | null;
  x: number;
  y: number;
  cluster: number;
  knn_dist: number;
  outlier: number;
}

export interface MapPayload {
  pooling: Pooling;
  model: string;
  projection: { pca_dim?: number; pca_explained_variance?: number; n?: number };
  clustering: { n_clusters?: number; silhouette_cosine?: number };
  points: MapPoint[];
}

export interface ProteinSummary {
  accession: string;
  name: string;
  gene: string | null;
  organism: string;
  length: number;
  family: string | null;
  pfam: string | null;
  ec_class: string | null;
  localization: string | null;
}

export interface SearchHit {
  rank: number;
  similarity: number;
  protein: ProteinSummary;
}

export interface RepresentationStats {
  embedding_norm: number;
  dim: number;
  nn_distance: number | null;
  knn_mean_distance: number | null;
  cluster: number | null;
  outlier_score: number | null;
  x: number | null;
  y: number | null;
}

export interface ProteinProfile {
  protein: ProteinSummary;
  protein_name_full: string;
  keywords: string[];
  sequence: string;
  model: string;
  stats: RepresentationStats;
  neighbors: SearchHit[];
  /** Present in demo bundles for showcase proteins. */
  attention_weights?: number[];
}

export interface MutationResult {
  mutation: string;
  pooling: Pooling;
  displacement: number;
  relative_displacement: number;
  cosine_similarity: number;
  local_delta: number;
  global_residue_delta: number;
  per_residue_delta: number[];
  top_dimensions: { dim: number; delta: number }[];
  note: string;
}

export interface LandscapeEffect {
  mutant: string;
  mutation: string;
  displacement: number;
  cosine_similarity: number;
  local_delta: number;
}

export interface Landscape {
  position: number;
  wildtype: string;
  pooling: Pooling;
  effects: LandscapeEffect[];
  max_displacement: string;
  min_displacement: string;
  note: string;
}

export interface AttentionPayload {
  accession: string;
  length: number;
  weights: number[];
  top_positions: number[];
  note: string;
}

export interface BenchmarkRow {
  representation: string;
  axis: "probe" | "retrieval" | "clustering" | "stability";
  task?: string;
  [metric: string]: string | number | null | undefined;
}

export interface BenchmarkPayload {
  rows: BenchmarkRow[];
  seq_vs_emb?: { a: string; b: string; identity: number; cosine: number; same_family: boolean }[];
  markdown?: string;
  /** Extended study rows (checkpoint scaling, pooling objectives, ProstT5). */
  extended?: ExtendedBenchmarkRow[];
}

export interface Health {
  status: string;
  model: string;
  corpus_size: number;
  poolings: Pooling[];
  device: string;
  encoder_loaded: boolean;
}

export interface DomainFeature {
  name: string;
  start: number; // 1-based, inclusive
  end: number;
}

export interface DomainsPayload {
  accession: string;
  length: number;
  domains: DomainFeature[];
  note: string;
}

export interface RegionSearchPayload {
  accession: string;
  start: number;
  end: number;
  span_length: number;
  hits: SearchHit[];
  note: string;
}

export interface ExtendedBenchmarkRow {
  representation: string;
  group: "esm2-scaling" | "pooling-objective" | "structure-aware" | "baseline";
  params_m: number;
  pooling: string;
  dim: number;
  probe_f1_mean: number;
  p_at_1: number;
  p_at_10: number;
  nmi: number;
  [metric: string]: string | number;
}

export interface TrajectoryStep {
  step: number;
  mutation: string;
  cumulative: string[];
  step_displacement: number;
  displacement_from_wt: number;
  cosine_to_wt: number;
}

export interface TrajectoryPayload {
  pooling: Pooling;
  n_steps: number;
  steps: TrajectoryStep[];
  path_length: number;
  net_displacement: number;
  directness: number;
  note: string;
}

export interface ClusterSummary {
  cluster: number;
  size: number;
  mean_length: number;
  top_families: Record<string, number>;
  top_organisms: Record<string, number>;
  enzyme_fraction: number;
}

export interface ClustersPayload {
  pooling: string;
  clustering: { n_clusters?: number; silhouette_cosine?: number; inertia?: number };
  clusters: ClusterSummary[];
}

export interface ComparePayload {
  a: ProteinSummary;
  b: ProteinSummary;
  cosine_by_pooling: Record<string, number>;
  sequence_identity: number;
  same_family: boolean;
  shared_pfam: string[];
  a_domains: DomainFeature[];
  b_domains: DomainFeature[];
  note: string;
}

export type MapPreset = "default" | "local" | "global";
