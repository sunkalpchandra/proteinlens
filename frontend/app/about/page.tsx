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

        <Section title="Limitations">
          <p>
            The encoder is the 35M-parameter ESM-2 — the smallest of the family; larger
            models organize sequence space measurably better, so absolute numbers here
            understate what the architecture can do. The UMAP layout distorts global
            structure and should be read only through local neighborhoods. Annotation
            sparsity in Swiss-Prot (missing families, incomplete EC and localization
            labels) caps probe and retrieval scores independently of representation
            quality. Attention weights are model-dependent interpretability signals, not
            functional-residue annotations. And nothing here uses structural information
            — all signals derive from sequence alone.
          </p>
        </Section>
      </div>
    </div>
  );
}
