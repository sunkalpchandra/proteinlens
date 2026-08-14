"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/explorer", label: "Explorer" },
  { href: "/search", label: "Search" },
  { href: "/mutation", label: "Mutation" },
  { href: "/clusters", label: "Clusters" },
  { href: "/compare", label: "Compare" },
  { href: "/benchmarks", label: "Benchmarks" },
  { href: "/about", label: "About" },
];

export function NavLinks() {
  const pathname = usePathname();
  return (
    <nav className="ml-auto flex items-center gap-1">
      {LINKS.map(({ href, label }) => {
        const active = pathname === href || pathname.startsWith(`${href}/`);
        return (
          <Link
            key={href}
            href={href}
            className={`rounded px-2.5 py-1 text-[13px] transition-colors ${
              active ? "bg-surface2 text-ink" : "text-ink2 hover:bg-surface hover:text-ink"
            }`}
          >
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
