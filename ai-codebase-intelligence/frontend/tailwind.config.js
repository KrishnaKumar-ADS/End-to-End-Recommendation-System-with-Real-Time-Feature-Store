/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/hooks/**/*.{js,ts,jsx,tsx}",
    "./src/lib/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["IBM Plex Sans", "Inter", "Segoe UI", "ui-sans-serif", "sans-serif"],
        mono: ["JetBrains Mono", "Cascadia Code", "Fira Code", "ui-monospace", "monospace"],
      },
      colors: {
        brand: {
          50: "#eff8ff",
          100: "#dbeffe",
          200: "#bfe3fd",
          300: "#94d1fc",
          400: "#60b6f9",
          500: "#3b96f5",
          600: "#1e77eb",
          700: "#1661d8",
          800: "#174ead",
          900: "#184389",
          950: "#142b54",
        },
        surface: {
          DEFAULT: "#0f1117",
          card: "#171b26",
          border: "#2a3242",
          hover: "#202839",
          input: "#121722",
          muted: "#8993a4",
        },
        provider: {
          qwen: "#7c3aed",
          gemini: "#1a73e8",
          deepseek: "#00b4d8",
        },
        status: {
          queued: "#6b7280",
          cloning: "#f59e0b",
          scanning: "#f59e0b",
          parsing: "#3b82f6",
          embedding: "#8b5cf6",
          completed: "#10b981",
          failed: "#ef4444",
        },
      },
      boxShadow: {
        card: "0 24px 60px -36px rgba(7, 14, 28, 0.65)",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        pulseTrack: {
          "0%, 100%": { opacity: "0.58" },
          "50%": { opacity: "1" },
        },
        progressBar: {
          "0%": { backgroundPosition: "200% 0" },
          "100%": { backgroundPosition: "-200% 0" },
        },
      },
      animation: {
        pulseTrack: "pulseTrack 1.8s ease-in-out infinite",
        "fade-in": "fadeIn 150ms ease-out",
        "slide-up": "slideUp 220ms ease-out",
        "progress-bar": "progressBar 1.5s ease-in-out infinite",
      },
      borderRadius: {
        "4xl": "2rem",
      },
    },
  },
  plugins: [],
};
