import fs from "node:fs";
import path from "node:path";
import type { MetadataRoute } from "next";

export const dynamic = "force-static";

const BASE = "https://sunkalpchandra.github.io/proteinlens";

export default function sitemap(): MetadataRoute.Sitemap {
  const pages: MetadataRoute.Sitemap = ["", "/explorer", "/search", "/mutation",
                 "/clusters", "/compare", "/benchmarks", "/about"].map((route) => ({
    url: `${BASE}${route}`,
    changeFrequency: "weekly",
  }));

  // Demo protein profiles, when the bundle exists at build time.
  try {
    const manifest = path.join(process.cwd(), "public", "demo", "proteins.json");
    const proteins = JSON.parse(fs.readFileSync(manifest, "utf8")) as {
      accession: string;
    }[];
    for (const p of proteins) {
      pages.push({
        url: `${BASE}/protein/${p.accession}`,
        changeFrequency: "monthly",
      });
    }
  } catch {
    /* no bundle at build time — page list alone */
  }
  return pages;
}
