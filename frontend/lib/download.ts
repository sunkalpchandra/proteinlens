/** Client-side file download via a Blob object URL.
 *
 *  The anchor is appended to the document before clicking (Firefox ignores
 *  clicks on detached anchors) and the object URL is revoked on a timeout —
 *  revoking synchronously can invalidate the URL before the browser's
 *  download machinery dereferences it.
 */

export function triggerDownload(filename: string, content: string, mimeType: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: mimeType }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export function downloadJson(filename: string, payload: unknown): void {
  triggerDownload(filename, JSON.stringify(payload, null, 2), "application/json");
}

export function downloadFasta(filename: string, header: string, sequence: string): void {
  const lines = sequence.match(/.{1,60}/g) ?? [sequence];
  triggerDownload(filename, `>${header}\n${lines.join("\n")}\n`, "text/plain");
}
