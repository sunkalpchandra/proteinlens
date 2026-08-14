"use client";

/** Copy / download actions for a protein sequence. Client-side only:
 *  clipboard API for copy, a Blob object-URL anchor for the FASTA download. */

import { useEffect, useRef, useState } from "react";

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

  const downloadFasta = () => {
    const lines = sequence.match(/.{1,60}/g) ?? [sequence];
    const fasta = `>${accession} ${name}\n${lines.join("\n")}\n`;
    const url = URL.createObjectURL(new Blob([fasta], { type: "text/plain" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${accession}.fasta`;
    anchor.click();
    URL.revokeObjectURL(url);
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
        onClick={downloadFasta}
        className="rounded border border-bd px-2 py-0.5 font-mono text-[10px] text-ink2 hover:bg-surface2"
      >
        fasta
      </button>
    </span>
  );
}
