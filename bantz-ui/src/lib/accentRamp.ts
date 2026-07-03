// Derive a coherent ember-* ramp from a single accent hex and apply it as CSS
// variables, so the Tailwind `ember-*` utility classes follow the picked
// accent instead of a hardcoded hex palette (#493).
//
// Tailwind's `ember` colours are defined as `rgb(var(--ember-N-rgb) /
// <alpha-value>)`, so opacity utilities (`bg-ember-500/60`) keep working — that
// requires the channel vars (`--ember-N-rgb`) to be space-separated "r g b".
// The raw CSS in index.css still reads the colour vars (`--ember-N`), so both
// forms are set.

export type Shade = 100 | 200 | 300 | 400 | 500;

const SHADES: Shade[] = [100, 200, 300, 400, 500];

// How far each shade is mixed toward white; 500 is the accent itself. Chosen so
// the default accent (#ff4500) yields a ramp close to the original palette.
const LIGHTEN: Record<Shade, number> = {
  500: 0,
  400: 0.18,
  300: 0.36,
  200: 0.55,
  100: 0.75,
};

function clampByte(n: number): number {
  return Math.max(0, Math.min(255, Math.round(n)));
}

export function hexToRgb(hex: string): [number, number, number] {
  let h = hex.trim().replace(/^#/, "");
  if (h.length === 3) h = h.split("").map((c) => c + c).join("");
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function rgbToHex(r: number, g: number, b: number): string {
  const v = (clampByte(r) << 16) | (clampByte(g) << 8) | clampByte(b);
  return "#" + v.toString(16).padStart(6, "0");
}

export interface RampEntry {
  hex: string;   // "#rrggbb"      — for raw CSS var() colours
  rgb: string;   // "r g b" channels — for Tailwind's <alpha-value> utilities
}

export function emberRamp(accentHex: string): Record<Shade, RampEntry> {
  const [br, bg, bb] = hexToRgb(accentHex);
  const out = {} as Record<Shade, RampEntry>;
  for (const shade of SHADES) {
    const t = LIGHTEN[shade];
    const r = clampByte(br + (255 - br) * t);
    const g = clampByte(bg + (255 - bg) * t);
    const b = clampByte(bb + (255 - bb) * t);
    out[shade] = { hex: rgbToHex(r, g, b), rgb: `${r} ${g} ${b}` };
  }
  return out;
}

// Apply an accent and its derived ramp to a root element (defaults to
// documentElement). Sets both the colour vars and the -rgb channel vars.
export function applyAccent(
  accentHex: string,
  root: HTMLElement = document.documentElement,
): void {
  const ramp = emberRamp(accentHex);
  const style = root.style;
  style.setProperty("--accent", accentHex);
  style.setProperty("--accent-rgb", ramp[500].rgb);
  for (const shade of SHADES) {
    style.setProperty(`--ember-${shade}`, ramp[shade].hex);
    style.setProperty(`--ember-${shade}-rgb`, ramp[shade].rgb);
  }
}
