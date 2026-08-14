"use client";

/** Sortable results table for SearchHit[]. Similarity renders as tabular text
 *  plus a thin single-hue bar (magnitude → accent, per the palette rules). */

import Link from "next/link";
import { useMemo, useState } from "react";
import type { SearchHit } from "@/lib/types";

type SortKey =
  | "rank"
  | "accession"
  | "name"
  | "organism"
  | "length"
  | "family"
  | "similarity";

interface Column {
  key: SortKey;
  label: string;
  numeric: boolean;
}

const BASE_COLUMNS: Column[] = [
  { key: "rank", label: "Rank", numeric: true },
  { key: "accession", label: "Accession", numeric: false },
  { key: "name", label: "Name", numeric: false },
  { key: "organism", label: "Organism", numeric: false },
  { key: "length", label: "Length", numeric: true },
  { key: "family", label: "Family", numeric: false },
];

const SIMILARITY_COLUMN: Column = { key: "similarity", label: "Similarity", numeric: true };

function sortValue(hit: SearchHit, key: SortKey): string | number {
  switch (key) {
    case "rank":
      return hit.rank;
    case "similarity":
      return hit.similarity;
    case "accession":
      return hit.protein.accession;
    case "name":
      return hit.protein.name;
    case "organism":
      return hit.protein.organism;
    case "length":
      return hit.protein.length;
    case "family":
      return hit.protein.family ?? "";
  }
}

export interface HitTableProps {
  hits: SearchHit[];
  showSimilarity?: boolean;
  className?: string;
}

export function HitTable({ hits, showSimilarity = true, className = "" }: HitTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("rank");
  const [sortAsc, setSortAsc] = useState(true);

  const columns = showSimilarity ? [...BASE_COLUMNS, SIMILARITY_COLUMN] : BASE_COLUMNS;

  const sorted = useMemo(() => {
    const copy = [...hits];
    copy.sort((a, b) => {
      const va = sortValue(a, sortKey);
      const vb = sortValue(b, sortKey);
      let cmp: number;
      if (typeof va === "number" && typeof vb === "number") {
        cmp = va - vb;
      } else {
        cmp = String(va).localeCompare(String(vb));
      }
      if (cmp === 0) cmp = a.rank - b.rank;
      return sortAsc ? cmp : -cmp;
    });
    return copy;
  }, [hits, sortKey, sortAsc]);

  const toggleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortAsc((asc) => !asc);
    } else {
      setSortKey(key);
      setSortAsc(true);
    }
  };

  return (
    <div className={`overflow-x-auto scroll-thin ${className}`}>
      <table className="w-full border-collapse text-[13px]">
        <thead>
          <tr className="border-b border-bds">
            {columns.map((col) => (
              <th
                key={col.key}
                className={`px-2 py-1.5 ${col.numeric ? "text-right" : "text-left"}`}
              >
                <button
                  type="button"
                  onClick={() => toggleSort(col.key)}
                  className="label-mono inline-flex items-center gap-1 hover:text-ink2"
                  aria-sort={
                    sortKey === col.key ? (sortAsc ? "ascending" : "descending") : undefined
                  }
                >
                  {col.label}
                  <span className="w-2 text-[9px]">
                    {sortKey === col.key ? (sortAsc ? "▴" : "▾") : ""}
                  </span>
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((hit) => {
            const p = hit.protein;
            return (
              <tr
                key={p.accession}
                className="border-b border-bd transition-colors hover:bg-surface2"
              >
                <td className="px-2 py-1.5 text-right font-mono text-ink3 tabular">
                  {hit.rank}
                </td>
                <td className="px-2 py-1.5">
                  <Link
                    href={`/protein/${encodeURIComponent(p.accession)}`}
                    className="font-mono text-accent hover:underline"
                  >
                    {p.accession}
                  </Link>
                </td>
                <td className="max-w-[240px] truncate px-2 py-1.5 text-ink" title={p.name}>
                  {p.name}
                </td>
                <td className="px-2 py-1.5 text-ink2">{p.organism}</td>
                <td className="px-2 py-1.5 text-right font-mono text-ink2 tabular">
                  {p.length}
                </td>
                <td
                  className="max-w-[180px] truncate px-2 py-1.5 text-ink2"
                  title={p.family ?? undefined}
                >
                  {p.family ?? <span className="text-ink3">—</span>}
                </td>
                {showSimilarity && (
                  <td className="px-2 py-1.5">
                    <div className="flex items-center justify-end gap-2">
                      <span className="font-mono text-ink tabular">
                        {hit.similarity.toFixed(3)}
                      </span>
                      <span className="h-[3px] w-16 shrink-0 overflow-hidden rounded-full bg-surface2">
                        <span
                          className="block h-full bg-accent"
                          style={{
                            width: `${Math.max(0, Math.min(1, hit.similarity)) * 100}%`,
                          }}
                        />
                      </span>
                    </div>
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
