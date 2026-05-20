"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

const navigation = [
  { href: "/", label: "Overview" },
  { href: "/ingest", label: "Ingest" },
  { href: "/repos", label: "Repositories" },
];

function normalize(pathname: string): string {
  return pathname.endsWith("/") && pathname !== "/" ? pathname.slice(0, -1) : pathname;
}

export function Header() {
  const pathname = normalize(usePathname());
  const [mobileOpen, setMobileOpen] = useState(false);

  const repoSubNav = useMemo(() => {
    const parts = pathname.split("/").filter(Boolean);
    if (parts[0] !== "repos" || !parts[1]) {
      return null;
    }
    const repoId = parts[1];
    return [
      { href: `/repos/${repoId}`, label: "Files" },
      { href: `/repos/${repoId}/chat`, label: "Chat" },
      { href: `/repos/${repoId}/graph`, label: "Graph" },
    ];
  }, [pathname]);

  return (
    <header className="sticky top-0 z-40 border-b border-surface-border/80 bg-surface/85 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-4 sm:px-6 lg:px-8">
        <div>
          <Link className="text-lg font-semibold tracking-tight text-white" href="/">
            AI Codebase Intelligence
          </Link>
        </div>

        <nav className="hidden items-center gap-2 rounded-full border border-surface-border bg-white/5 p-1 md:flex">
          {navigation.map((item) => {
            const active = pathname === item.href || (item.href !== "/" && pathname.startsWith(`${item.href}/`));
            return (
              <Link
                key={item.href}
                className={cn(
                  "rounded-full px-4 py-2 text-sm font-medium transition",
                  active
                    ? "bg-brand-600 text-white"
                    : "text-slate-300 hover:bg-white/8 hover:text-white",
                )}
                href={item.href}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>

        <button
          className="inline-flex items-center rounded-lg border border-surface-border bg-white/5 px-3 py-2 text-sm text-slate-200 md:hidden"
          onClick={() => setMobileOpen((current) => !current)}
          type="button"
        >
          Menu
        </button>
      </div>

      {repoSubNav ? (
        <div className="border-t border-surface-border/70 px-4 py-2 sm:px-6 lg:px-8">
          <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-2">
            {repoSubNav.map((item) => {
              const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
              return (
                <Link
                  className={cn(
                    "rounded-full border px-3 py-1.5 text-xs font-medium transition",
                    active
                      ? "border-brand-500/40 bg-brand-500/10 text-brand-100"
                      : "border-surface-border bg-surface-card text-slate-300 hover:border-brand-500/30 hover:text-white",
                  )}
                  href={item.href}
                  key={item.href}
                >
                  {item.label}
                </Link>
              );
            })}
          </div>
        </div>
      ) : null}

      {mobileOpen ? (
        <div className="fixed inset-0 z-50 bg-black/50 md:hidden" onClick={() => setMobileOpen(false)}>
          <div className="ml-auto h-full w-72 border-l border-surface-border bg-surface p-4" onClick={(event) => event.stopPropagation()}>
            <div className="mb-4 flex items-center justify-between">
              <p className="text-sm font-semibold text-white">Navigation</p>
              <button
                className="rounded-lg border border-surface-border px-2 py-1 text-xs text-slate-200"
                onClick={() => setMobileOpen(false)}
                type="button"
              >
                Close
              </button>
            </div>

            <div className="space-y-2">
              {navigation.map((item) => {
                const active = pathname === item.href || (item.href !== "/" && pathname.startsWith(`${item.href}/`));
                return (
                  <Link
                    className={cn(
                      "block rounded-lg px-3 py-2 text-sm transition",
                      active
                        ? "bg-brand-600 text-white"
                        : "text-slate-200 hover:bg-surface-hover",
                    )}
                    href={item.href}
                    key={item.href}
                    onClick={() => setMobileOpen(false)}
                  >
                    {item.label}
                  </Link>
                );
              })}

              {repoSubNav ? (
                <>
                  <p className="pt-3 text-[11px] uppercase tracking-[0.08em] text-surface-muted">Current repo</p>
                  {repoSubNav.map((item) => {
                    const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
                    return (
                      <Link
                        className={cn(
                          "block rounded-lg px-3 py-2 text-sm transition",
                          active
                            ? "bg-brand-500/20 text-brand-100"
                            : "text-slate-200 hover:bg-surface-hover",
                        )}
                        href={item.href}
                        key={item.href}
                        onClick={() => setMobileOpen(false)}
                      >
                        {item.label}
                      </Link>
                    );
                  })}
                </>
              ) : null}
            </div>
          </div>
        </div>
      ) : null}
    </header>
  );
}
