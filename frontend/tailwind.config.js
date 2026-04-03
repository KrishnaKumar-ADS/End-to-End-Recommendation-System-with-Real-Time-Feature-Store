/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // DS19 brand colors (dark cinema theme)
        cinema: {
          bg: "#0f0f1a",          // deep dark background
          card: "#1a1a2e",        // card background
          accent: "#e94560",      // red accent (cinema red)
          accent2: "#0f3460",     // blue secondary
          text: "#e0e0e0",        // primary text
          muted: "#8892a4",       // secondary text
          border: "#2a2a40",      // subtle border
          success: "#22c55e",     // positive feedback green
          warning: "#f59e0b",     // warning amber
        }
      },
      fontFamily: {
        display: ["'Inter'", "system-ui", "sans-serif"],
      },
      animation: {
        "fade-in": "fadeIn 0.3s ease-in-out",
        "slide-up": "slideUp 0.4s ease-out",
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
      keyframes: {
        fadeIn: { "0%": { opacity: "0" }, "100%": { opacity: "1" } },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(20px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};