/** Server wrapper for the protein profile.
 *
 *  With static export (GitHub Pages), every route must be pre-rendered:
 *  generateStaticParams reads the demo bundle's protein list at build time
 *  and emits one page per demo accession. The page body itself is fully
 *  client-side (it fetches profile JSON at runtime), so the pre-rendered
 *  shell is identical for every protein. In `next dev` against the live API,
 *  any accession resolves dynamically as usual.
 */

import fs from "node:fs";
import path from "node:path";
import { ProteinPageClient } from "./profile-client";

export function generateStaticParams(): { id: string }[] {
  const manifest = path.join(process.cwd(), "public", "demo", "proteins.json");
  try {
    const proteins = JSON.parse(fs.readFileSync(manifest, "utf8")) as {
      accession: string;
    }[];
    return proteins.map((p) => ({ id: p.accession }));
  } catch {
    // No demo bundle yet (e.g. CI before artifacts are built). Next rejects
    // an empty list under `output: export`, so emit one known showcase
    // accession as a placeholder shell (the page fetches data client-side).
    return [{ id: "P69905" }];
  }
}

export default function ProteinPage() {
  return <ProteinPageClient />;
}
