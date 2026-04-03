import { Link, useLocation } from "react-router-dom";
import LatencyBadge from "./LatencyBadge";

export default function Navbar() {
  const { pathname } = useLocation();

  const navItems = [
    { path: "/", label: "Recommendations" },
    { path: "/ab-dashboard", label: "A/B Results" },
  ];

  return (
    <nav className="border-b border-cinema-border bg-cinema-card/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">

          {/* Logo */}
          <div className="flex items-center gap-3">
            <span className="text-cinema-accent text-2xl">🎬</span>
            <span className="font-bold text-lg tracking-tight">
              DS19 <span className="text-cinema-muted font-normal text-sm">Recommender</span>
            </span>
          </div>

          {/* Navigation Links */}
          <div className="flex items-center gap-1">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors
                  ${pathname === item.path
                    ? "bg-cinema-accent/20 text-cinema-accent"
                    : "text-cinema-muted hover:text-cinema-text hover:bg-cinema-border/50"
                  }`}
              >
                {item.label}
              </Link>
            ))}
          </div>

          {/* Live Latency Badge */}
          <LatencyBadge />
        </div>
      </div>
    </nav>
  );
}