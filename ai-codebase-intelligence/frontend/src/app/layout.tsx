import type { Metadata } from "next";

import { Header } from "@/components/Layout/Header";
import { ToastProvider } from "@/components/ui/Toast";

import "./globals.css";

export const metadata: Metadata = {
  title: "AI Codebase Intelligence",
  description: "Frontend scaffold for repository ingestion, search, and code understanding.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html className="dark" lang="en">
      <body>
        <ToastProvider>
          <Header />
          <main className="mx-auto min-h-[calc(100vh-73px)] max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
            {children}
          </main>
        </ToastProvider>
      </body>
    </html>
  );
}
