"use client";

/** Copy / download actions for a protein sequence. Client-side only:
 *  clipboard API for copy, a Blob object-URL anchor for the FASTA download. */

import { useEffect, useRef, useState } from "react";
import { downloadFasta } from "@/lib/download";

export function SequenceActions({
  accession,
  name,
  sequence,
}: {
  accession: string;
  name: string;
  sequence: string;
}) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => {
    if (timer.current) clearTimeout(timer.current);
  }, []);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(sequence);
      setCopied(true);
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable (permissions/http) — button just stays quiet */
    }
  };


  return (
    <span className="flex items-center gap-2">
      <button
        type="button"
        onClick={() => void copy()}
        className="rounded border border-bd px-2 py-0.5 font-mono text-[10px] text-ink2 hover:bg-surface2"
      >
        {copied ? "copied" : "copy"}
      </button>
      <button
        type="button"
        onClick={() => downloadFasta(`${accession}.fasta`, `${accession} ${name}`, sequence)}
        className="rounded border border-bd px-2 py-0.5 font-mono text-[10px] text-ink2 hover:bg-surface2"
      >
        fasta
      </button>
    </span>
  );
}
