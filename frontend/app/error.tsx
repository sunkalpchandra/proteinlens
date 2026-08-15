"use client";

/** Route-level error boundary: a quiet panel with a retry, in place of a
 *  white screen, when a page throws during render. */

export default function RouteError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="mx-auto max-w-2xl px-4 py-16">
      <div className="panel px-4 py-4">
        <p className="label-mono pb-2">something broke</p>
        <p className="text-[13px] text-ink2">{error.message || "Unexpected error."}</p>
        <button
          type="button"
          onClick={reset}
          className="mt-3 rounded border border-bds px-3 py-1 font-mono text-[12px] text-ink hover:bg-surface2"
        >
          try again
        </button>
      </div>
    </div>
  );
}
