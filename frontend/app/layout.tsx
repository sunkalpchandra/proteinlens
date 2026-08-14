import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { NavLinks } from "@/components/nav-links";

export const metadata: Metadata = {
  title: "ProteinLens",
  description:
    "Explore how a frozen ESM-2 protein language model organizes sequence space: embeddings, retrieval, mutation perturbation analysis, and representation benchmarks.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <header className="sticky top-0 z-40 border-b border-bd bg-page/90 backdrop-blur">
          <div className="mx-auto flex h-12 max-w-[1400px] items-center gap-6 px-4">
            <Link href="/" className="flex items-baseline gap-2">
              <span className="font-mono text-[13px] font-semibold tracking-[0.18em] text-ink">
                PROTEINLENS
              </span>
              <span className="hidden font-mono text-[10px] text-ink3 sm:inline">
                ESM-2 representation explorer
              </span>
            </Link>
            <NavLinks />
          </div>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
