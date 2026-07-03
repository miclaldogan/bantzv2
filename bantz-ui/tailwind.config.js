/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // Bantz palette — Obsidian Manor
        obsidian: {
          900: "#0D0A07",
          850: "#0D0D0D",
          800: "#141414",
          700: "#1A1A1A",
          600: "#212121",
          500: "#2A2A2A",
          400: "#383838",
          300: "#555555",
          200: "#888888",
          100: "#BBBBBB",
        },
        // Radiant Ember — driven by CSS vars so the accent picker recolors
        // every ember-* utility (#493). Channel vars + <alpha-value> keep
        // opacity utilities (bg-ember-500/60) working.
        ember: {
          100: "rgb(var(--ember-100-rgb) / <alpha-value>)",
          200: "rgb(var(--ember-200-rgb) / <alpha-value>)",
          300: "rgb(var(--ember-300-rgb) / <alpha-value>)",
          400: "rgb(var(--ember-400-rgb) / <alpha-value>)",
          500: "rgb(var(--ember-500-rgb) / <alpha-value>)",
        },
        // Imperial Gold
        gold: {
          200: "#FFE040",
          300: "#F5CC00",
          400: "#E2BB0B",
          500: "#C9A80C",
        },
        // Blood Rose
        rose: {
          200: "#E03030",
          300: "#CC1111",
          400: "#A80000",
          500: "#8B0000",
        },
        // Midnight Velvet
        velvet: {
          200: "#00BFFF",
          400: "#0033AA",
          500: "#000080",
          700: "#0A192F",
          900: "#00050A",
        },
        fg: {
          primary: "#F0E8D8",
          secondary: "#BBBBBB",
          muted: "#555555",
        },
      },
      fontFamily: {
        display: ["'Syncopate'", "'JetBrains Mono'", "monospace"],
        ui: ["'Josefin Sans'", "'Trebuchet MS'", "sans-serif"],
        terminal: ["'ShareTechMono'", "'JetBrains Mono'", "monospace"],
        code: ["'JetBrains Mono'", "monospace"],
      },
      letterSpacing: {
        widest: "0.25em",
        wider: "0.15em",
        wide: "0.08em",
      },
      boxShadow: {
        ember: "0 0 12px rgb(var(--ember-500-rgb) / 0.6)",
        "ember-soft": "0 0 24px rgb(var(--ember-500-rgb) / 0.25)",
        gold: "0 0 8px rgba(212, 175, 55, 0.4)",
        "inner-well": "inset 0 0 20px rgba(0, 0, 0, 0.8)",
      },
      transitionTimingFunction: {
        bantz: "cubic-bezier(0.9, 0, 0.1, 1)",
      },
      keyframes: {
        "bantz-glitch": {
          "0%": { opacity: "1", transform: "skewX(0deg)" },
          "10%": { opacity: "0", transform: "skewX(1.5deg)" },
          "20%": { opacity: "1", transform: "skewX(-0.5deg)" },
          "30%": { opacity: "0.7" },
          "100%": { opacity: "1", transform: "skewX(0deg)" },
        },
        "bantz-pulse": {
          "0%, 100%": { filter: "drop-shadow(0 0 6px var(--ember-500))" },
          "50%": {
            filter:
              "drop-shadow(0 0 18px var(--ember-400)) drop-shadow(0 0 4px #C9A80C)",
          },
        },
        "bantz-waveform": {
          "0%, 100%": { transform: "scaleY(0.3)" },
          "50%": { transform: "scaleY(1)" },
        },
        blink: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0" },
        },
      },
      animation: {
        glitch: "bantz-glitch 150ms cubic-bezier(0.9, 0, 0.1, 1) forwards",
        pulse: "bantz-pulse 3s ease-in-out infinite",
        waveform: "bantz-waveform 1.1s ease-in-out infinite",
        blink: "blink 1s steps(2, end) infinite",
      },
    },
  },
  plugins: [],
};
