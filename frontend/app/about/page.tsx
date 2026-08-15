import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "About — ProteinLens",
  description: "Methodology: model, pooling, corpus, retrieval, splits, and limitations.",
};

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border-t border-bd pt-6">
      <h2 className="label-mono">{title}</h2>
      <div className="mt-3 space-y-3 text-[13.5px] leading-relaxed text-ink2">
        {children}
      </div>
    </section>
  );
}

function CodeLine({ children }: { children: React.ReactNode }) {
  return (
    <code className="block rounded border border-bd bg-surface2 px-3 py-1.5 font-mono text-[12.5px] text-ink">
      {children}
    </code>
  );
}

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-10">
      <p className="label-mono mb-3">Methodology</p>
      <h1 className="text-2xl font-semibold tracking-tight text-ink">
        How ProteinLens works
      </h1>
      <p className="mt-3 max-w-xl text-[13.5px] leading-relaxed text-ink2">
        Every view in this app is a readout of one frozen protein language model over one
        fixed corpus. This page states exactly what is computed, and what the numbers do
        and do not mean.
      </p>

      <div className="mt-10 space-y-8">
        <Section title="Model">
          <p>
            All representations come from{" "}
            <code className="font-mono text-[12.5px] text-ink">
              facebook/esm2_t12_35M_UR50D
            </code>{" "}
            — a 12-layer, 35M-parameter ESM-2 encoder with 480-dimensional hidden states.
            The model is frozen: no weights are fine-tuned at any point. Each sequence is
            encoded once; the special tokens (BOS/EOS, padding) are stripped before
            pooling, so residue representations align 1:1 with sequence positions.
          </p>
        </Section>

        <Section title="Pooling">
          <p>
            A protein-level vector is produced from the residue states{" "}
            <code className="font-mono text-[12.5px] text-ink">h_1 … h_L</code> in four
            ways: <span className="text-ink">mean</span> (average over residues),{" "}
            <span className="text-ink">max</span> (elementwise maximum),{" "}
            <span className="text-ink">bos</span> (the BOS token state, kept for this
            purpose before stripping), and a learned{" "}
            <span className="text-ink">additive attention</span> pooler trained on top of
            the frozen encoder:
          </p>
          <CodeLine>alpha_i = softmax(w2^T tanh(W1 h_i))</CodeLine>
          <CodeLine>z = sum_i alpha_i h_i</CodeLine>
          <p>
            The attention weights <code className="font-mono text-[12.5px] text-ink">alpha_i</code>{" "}
            are the per-residue values shown in the attention track.
          </p>
        </Section>

        <Section title="Corpus">
          <p>
            Reviewed UniProtKB/Swiss-Prot entries from 9 organisms, filtered to sequences
            of at most 512 residues and deduplicated — roughly 12k proteins after
            filtering. Annotations used downstream (family, Pfam, EC class, subcellular
            localization) come from the same Swiss-Prot records. Data is used under CC BY
            4.0.
          </p>
        </Section>

        <Section title="Retrieval">
          <p>
            Embeddings are L2-normalized and indexed with FAISS{" "}
            <code className="font-mono text-[12.5px] text-ink">IndexFlatIP</code>, so
            inner product equals cosine similarity and search is exact — no approximate
            nearest-neighbor structure, no recall loss. Neighbor lists everywhere in the
            app are exact cosine rankings within the corpus.
          </p>
        </Section>

        <Section title="Map">
          <p>
            The 2-D map is PCA to 50 dimensions followed by UMAP to 2, computed once and
            cached. UMAP preserves local neighborhoods at the expense of global geometry:
            distances between distant clusters and cluster sizes are not meaningful.
            Cluster labels shown on points come from k-means in the original embedding
            space, not from the 2-D layout.
          </p>
        </Section>

        <Section title="Splits">
          <p>
            Probe evaluations use family-grouped splits: all members of a family land in
            the same fold, so a probe cannot score by memorizing near-duplicates.
            Proteins without a family annotation are grouped by k-mer Jaccard similarity
            as a fallback, and a leakage audit checks for high-identity pairs crossing
            the train/test boundary.
          </p>
        </Section>

        <Section title="Mutation analysis">
          <p>
            For a substitution, the mutant sequence is re-encoded and compared to
            wild-type in representation space:
          </p>
          <CodeLine>delta_z = z_mut - z_wt</CodeLine>
          <p>
            Reported quantities are the displacement ‖delta_z‖ (absolute and relative to
            ‖z_wt‖), the cosine similarity between wild-type and mutant embeddings, and
            per-residue deltas in a local window around the mutated position. This is a{" "}
            <span className="text-ink">representation-space perturbation measure</span>:
            it quantifies how much the model&apos;s encoding moves. It is not a fitness,
            stability, or pathogenicity prediction, and it is not validated against
            experimental mutational data.
          </p>
        </Section>

        <Section title="Benchmarks">
          <p>
            Four axes, all on frozen representations, all against k-mer and one-hot
            composition baselines: linear probes (enzyme/non-enzyme, EC class,
            subcellular localization; accuracy and macro F1 on grouped splits), retrieval
            precision@k against family labels, k-means clustering purity and NMI against
            family annotations, and perturbation stability (cosine between wild-type and
            single-substitution embeddings). The identity-vs-cosine scatter samples
            protein pairs to show where embedding similarity departs from sequence
            similarity.
          </p>
        </Section>

        <Section title="Extended studies">
          <p>
            A fixed, stratified ~3,000-protein evaluation subset supports comparisons the
            full corpus cannot afford: ESM-2 checkpoint scaling (8M / 35M / 150M behind
            one registry-driven interface), attention poolers trained with cross-entropy
            versus a supervised-contrastive objective (SupCon directly optimizes the
            cosine geometry retrieval uses; early stopping on holdout 1-NN accuracy), and
            the ProstT5 encoder as a structure-aware reference (3Di-translation training,
            ~1.2B parameters, fp16 — a reference point rather than a like-for-like
            pooling comparison). Every candidate embeds the identical subset and runs the
            identical probe/retrieval/clustering suite. Density-based clustering adds
            an honest footnote: HDBSCAN on this corpus (PCA-50 space, leaf selection)
            finds 47 tight islands while ~90% of proteins sit on one connected
            low-density manifold — embedding space is mostly a continuum, not a set of
            well-separated blobs, which is why the browsable partition uses k-means.
          </p>
        </Section>

        <Section title="Split validation">
          <p>
            Two independent grouping methods guard against homology leakage: the default
            union-find over family and Pfam annotation tokens, and MMseqs2 clustering at
            30% identity / 80% coverage. On 200,000 sampled protein pairs, no pair joined
            by identity clustering is separated by the annotation grouping — the
            annotation method is strictly more conservative. Probe metrics measured under
            the looser identity splits come out higher, which is exactly the signature of
            readmitted homology; reported numbers use the stricter grouping.
          </p>
        </Section>

        <Section title="Index backends">
          <p>
            Retrieval serves exact cosine search (FAISS IndexFlatIP) at the current 12k
            scale and switches to HNSW past 50k vectors, where measured recall@10 stays
            above 0.999 at several times the exact-search throughput. IVF is available as
            the memory-lean alternative. All three sit behind one interface with
            persisted backend metadata.
          </p>
        </Section>

        <Section title="Region embeddings">
          <p>
            A region embedding mean-pools the residue vectors of a contiguous span —
            the same construction as protein-level mean pooling, restricted to the span —
            and queries the protein-level index with it. UniProt-curated DOMAIN
            coordinates (about a third of the corpus carries them) seed the spans;
            arbitrary spans work too. Results are a cross-granularity cosine comparison
            and are labeled as such.
          </p>
        </Section>

        <Section title="Limitations">
          <p>
            The serving encoder is the 35M-parameter ESM-2 — the smallest checkpoints of
            the family organize sequence space measurably worse than large ones, and the
            scaling study quantifies exactly that gap on this corpus. The UMAP layout
            distorts global structure and should be read only through local
            neighborhoods. Annotation sparsity in Swiss-Prot (missing families,
            incomplete EC and localization labels) caps probe and retrieval scores
            independently of representation quality. Attention weights are
            model-dependent interpretability signals, not functional-residue annotations.
            Structural information enters only indirectly, through ProstT5&apos;s
            supervision, and only as a benchmark reference — the served representations
            derive from sequence alone.
          </p>
        </Section>
      </div>
    </div>
  );
}
