"use client";

/** JSON export button; download mechanics live in lib/download. */

import { downloadJson } from "@/lib/download";

export function DownloadJson({
  payload,
  filename,
}: {
  payload: unknown;
  filename: string;
}) {
  return (
    <button
      type="button"
      onClick={() => downloadJson(filename, payload)}
      className="rounded border border-bd px-2 py-0.5 font-mono text-[10px] text-ink2 hover:bg-surface2"
    >
      download json
    </button>
  );
}
