"use client";

/** Tiny client-side JSON export: serializes the given payload to a Blob and
 *  triggers a download. No server round-trip. */

export function DownloadJson({
  payload,
  filename,
}: {
  payload: unknown;
  filename: string;
}) {
  const download = () => {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <button
      type="button"
      onClick={download}
      className="rounded border border-bd px-2 py-0.5 font-mono text-[10px] text-ink2 hover:bg-surface2"
    >
      download json
    </button>
  );
}
