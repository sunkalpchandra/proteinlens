import type { NextConfig } from "next";

// Static export: the demo runs entirely from precomputed JSON under
// public/demo, so the built site is plain files — deployable to GitHub Pages
// (with NEXT_PUBLIC_BASE_PATH=/proteinlens) or any static host. Local
// development (`next dev`) is unaffected and supports the live API.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  output: "export",
  basePath,
  trailingSlash: true, // directory/index.html URLs — robust on GitHub Pages
  images: { unoptimized: true },
};

export default nextConfig;
