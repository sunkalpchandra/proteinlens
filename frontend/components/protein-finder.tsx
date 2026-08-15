"use client";

/** Shared protein typeahead: debounced metadata search with a dropdown.
 *
 *  One implementation for every page that needs a picker, so debounce,
 *  cancellation, Enter/Escape handling, and dropdown mechanics stay uniform.
 *  ``onPick`` receives the full summary; callers keep whatever state shape
 *  they need. */

import { useEffect, useState } from "react";
import { findProteins } from "@/lib/data";
import type { ProteinSummary } from "@/lib/types";

export interface ProteinFinderProps {
  onPick: (protein: ProteinSummary) => void;
  placeholder?: string;
  /** Shown inside the input after picking; cleared on typing. */
  reflectPick?: boolean;
  className?: string;
}

export function ProteinFinder({
  onPick,
  placeholder = "find a protein — name, gene, or accession",
  reflectPick = true,
  className = "max-w-xl",
}: ProteinFinderProps) {
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<ProteinSummary[]>([]);
  const [open, setOpen] = useState(false);
  const [searching, setSearching] = useState(false);

  useEffect(() => {
    const needle = q.trim();
    if (!needle) {
      setHits([]);
      setOpen(false);
      setSearching(false);
      return;
    }
    let cancelled = false;
    setSearching(true);
    const timer = setTimeout(async () => {
      try {
        const res = await findProteins(needle, 12);
        if (!cancelled) {
          setHits(res);
          setOpen(true);
        }
      } catch {
        if (!cancelled) setHits([]);
      } finally {
        if (!cancelled) setSearching(false);
      }
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [q]);

  const choose = (p: ProteinSummary) => {
    setQ(reflectPick ? `${p.accession} · ${p.name}` : "");
    setOpen(false);
    onPick(p);
  };

  return (
    <div className={`relative ${className}`}>
      <input
        type="text"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        onFocus={() => hits.length > 0 && setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && hits.length > 0) choose(hits[0]);
          if (e.key === "Escape") setOpen(false);
        }}
        placeholder={placeholder}
        spellCheck={false}
        className="w-full rounded border border-bd bg-surface2 px-3 py-1.5 font-mono text-[12px] text-ink outline-none placeholder:text-ink3 focus:border-bds"
      />
      {searching && (
        <span className="loading-pulse absolute right-3 top-1/2 -translate-y-1/2 font-mono text-[10px] text-ink3">
          searching…
        </span>
      )}
      {open && hits.length > 0 && (
        <div className="scroll-thin absolute z-20 mt-1 max-h-72 w-full overflow-y-auto rounded border border-bds bg-surface2 shadow-xl">
          {hits.map((p) => (
            <button
              key={p.accession}
              type="button"
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => choose(p)}
              className="flex w-full items-baseline gap-2 px-3 py-1.5 text-left hover:bg-surface"
            >
              <span className="font-mono text-[11px] text-accent">{p.accession}</span>
              <span className="truncate text-[12px] text-ink">{p.name}</span>
              <span className="ml-auto shrink-0 font-mono text-[10px] text-ink3">
                {p.gene ? `${p.gene} · ` : ""}
                {p.organism} · {p.length} aa
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
