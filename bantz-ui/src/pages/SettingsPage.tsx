import { useEffect, useState, type ReactNode } from "react";
import { Eye, EyeOff, Check, Save } from "lucide-react";
import { PageTitle, PanelHeader } from "../components/primitives";
import { useAppStore } from "../store/useAppStore";
import { applyAccent } from "../lib/accentRamp";

// Default Ollama host, used when no base URL has been configured.
const DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434";

// Conversation backend. Must match BANTZ_LLM_PROVIDER values the router accepts.
const PROVIDERS = [
  { key: "ollama", label: "Ollama (local)" },
  { key: "claude", label: "Claude (Anthropic)" },
  { key: "gemini", label: "Gemini (Google)" },
  { key: "openai", label: "OpenAI" },
];

// Anthropic model IDs (latest families).
const CLAUDE_MODELS = [
  "claude-opus-4-8",
  "claude-sonnet-4-6",
  "claude-haiku-4-5-20251001",
  "claude-fable-5",
];

const ACCENTS = [
  { key: "ember",  hex: "#FF4500", label: "Ember" },
  { key: "gold",   hex: "#E2BB0B", label: "Gold" },
  { key: "velvet", hex: "#00BFFF", label: "Velvet" },
  { key: "rose",   hex: "#CC1111", label: "Rose" },
];

// UI uses short autonomy keys; backend (BANTZ_AUTONOMY) uses full words.
const AUTONOMY_TO_BACKEND: Record<string, string> =
  { low: "low", med: "medium", high: "high", abs: "absolute" };
const AUTONOMY_FROM_BACKEND: Record<string, SettingsState["autonomy"]> =
  { low: "low", medium: "med", high: "high", absolute: "abs" };

// Appearance prefs persist client-side only (no backend config needed).
const APPEARANCE_LS_KEY = "bantz.appearance";

function loadAppearance(): Partial<SettingsState> {
  try {
    const raw = localStorage.getItem(APPEARANCE_LS_KEY);
    if (!raw) return {};
    const a = JSON.parse(raw) as Partial<Pick<SettingsState, "accent" | "nightShift" | "crt">>;
    const out: Partial<SettingsState> = {};
    if (typeof a.accent === "string") out.accent = a.accent;
    if (typeof a.nightShift === "boolean") out.nightShift = a.nightShift;
    if (typeof a.crt === "boolean") out.crt = a.crt;
    return out;
  } catch {
    return {};
  }
}

interface SettingsState {
  provider: string;
  ollamaModel: string;
  claudeKey: string;
  claudeModel: string;
  geminiKey: string;
  geminiEnabled: boolean;
  ctx: string;
  wakeWord: boolean;
  stt: boolean;
  tts: boolean;
  lang: "TR" | "EN";
  distillation: boolean;
  shellConfirm: boolean;
  observerEnabled: boolean;
  accent: string;
  nightShift: boolean;
  crt: boolean;
  verbosity: "silent" | "standard" | "insufferable";
  autonomy: "low" | "med" | "high" | "abs";
  mood: "tolerant" | "impatient" | "resigned";
  bonding: number;
}

const DEFAULTS: SettingsState = {
  provider: "ollama",
  ollamaModel: "llama3.1:70b",
  claudeKey: "",
  claudeModel: "claude-sonnet-4-6",
  geminiKey: "",
  geminiEnabled: false,
  ctx: "32k",
  wakeWord: false,
  stt: false,
  tts: false,
  lang: "EN",
  distillation: true,
  shellConfirm: true,
  observerEnabled: false,
  accent: "ember",
  nightShift: false,
  crt: true,
  verbosity: "standard",
  autonomy: "high",
  mood: "tolerant",
  bonding: 42,
};

export function SettingsPage() {
  const configValues = useAppStore((s) => s.configValues);
  const wsSend       = useAppStore((s) => s.wsSend);

  const [s, setS]         = useState<SettingsState>(() => ({ ...DEFAULTS, ...loadAppearance() }));
  const [showKey, setShowKey] = useState(false);
  const [showClaudeKey, setShowClaudeKey] = useState(false);
  const [saved, setSaved] = useState(false);
  const [restartNeeded, setRestartNeeded] = useState(false);

  // Ollama models installed on this machine (fetched from the local API).
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [ollamaStatus, setOllamaStatus] = useState<"loading" | "ok" | "error">("loading");

  const set = <K extends keyof SettingsState>(k: K, v: SettingsState[K]) =>
    setS((prev) => ({ ...prev, [k]: v }));

  // Appearance prefs: apply to the DOM immediately and persist to localStorage.
  useEffect(() => {
    const hex = ACCENTS.find((a) => a.key === s.accent)?.hex ?? ACCENTS[0].hex;
    applyAccent(hex);   // recolor --accent + the whole --ember-* ramp (#493)
    document.body.classList.toggle("night-shift", s.nightShift);
    document.body.classList.toggle("crt", s.crt);
    localStorage.setItem(
      APPEARANCE_LS_KEY,
      JSON.stringify({ accent: s.accent, nightShift: s.nightShift, crt: s.crt }),
    );
  }, [s.accent, s.nightShift, s.crt]);

  // Ollama host from config (falls back to localhost), trailing slash stripped.
  const ollamaBaseUrl = (configValues?.ollama_base_url || DEFAULT_OLLAMA_BASE_URL)
    .replace(/\/+$/, "");

  // Fetch the installed Ollama models. Re-runs if the configured host changes
  // (e.g. once the backend config arrives over the WebSocket bridge).
  useEffect(() => {
    let cancelled = false;
    setOllamaStatus("loading");
    (async () => {
      try {
        const res = await fetch(`${ollamaBaseUrl}/api/tags`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as { models?: { name: string }[] };
        if (cancelled) return;
        const names = (data.models ?? [])
          .map((m) => m.name)
          .filter((n): n is string => typeof n === "string" && n.length > 0);
        setOllamaModels(names);
        setOllamaStatus("ok");
      } catch {
        if (!cancelled) setOllamaStatus("error");
      }
    })();
    return () => { cancelled = true; };
  }, [ollamaBaseUrl]);

  // Populate from backend config when it arrives
  useEffect(() => {
    if (!configValues) return;
    setS((prev) => ({
      ...prev,
      provider:       configValues.llm_provider     || prev.provider,
      ollamaModel:    configValues.ollama_model     || prev.ollamaModel,
      claudeKey:      configValues.anthropic_api_key || prev.claudeKey,
      claudeModel:    configValues.anthropic_model  || prev.claudeModel,
      geminiKey:      configValues.gemini_api_key   || prev.geminiKey,
      geminiEnabled:  configValues.gemini_enabled   ?? prev.geminiEnabled,
      wakeWord:       configValues.wake_word_enabled ?? prev.wakeWord,
      stt:            configValues.stt_enabled       ?? prev.stt,
      tts:            configValues.tts_enabled       ?? prev.tts,
      lang:           configValues.language?.toUpperCase() === "TR" ? "TR" : "EN",
      distillation:   configValues.distillation_enabled  ?? prev.distillation,
      shellConfirm:   configValues.shell_confirm_destructive ?? prev.shellConfirm,
      observerEnabled: configValues.observer_enabled ?? prev.observerEnabled,
      verbosity:      (configValues.verbosity as SettingsState["verbosity"]) || prev.verbosity,
      autonomy:       AUTONOMY_FROM_BACKEND[configValues.autonomy] ?? prev.autonomy,
      mood:           (configValues.mood_bias as SettingsState["mood"]) || prev.mood,
    }));
  }, [configValues]);

  function applyChanges() {
    if (!wsSend) return;
    const changes: [string, unknown][] = [
      ["llm_provider",               s.provider],
      ["ollama_model",               s.ollamaModel],
      ["anthropic_api_key",          s.claudeKey],
      ["anthropic_model",            s.claudeModel],
      ["gemini_enabled",             s.geminiEnabled],
      ["gemini_api_key",             s.geminiKey],
      ["language",                   s.lang.toLowerCase()],
      ["tts_enabled",                s.tts],
      ["stt_enabled",                s.stt],
      ["wake_word_enabled",          s.wakeWord],
      ["distillation_enabled",       s.distillation],
      ["shell_confirm_destructive",  s.shellConfirm],
      ["observer_enabled",           s.observerEnabled],
      ["verbosity",                  s.verbosity],
      ["autonomy",                   AUTONOMY_TO_BACKEND[s.autonomy]],
      ["mood_bias",                  s.mood],
    ];
    for (const [key, value] of changes) {
      wsSend({ type: "set_config", key, value });
    }
    // These subsystems are initialised once at daemon startup — changing them
    // persists but only takes effect after a restart.
    const RESTART_KEYS = ["wake_word_enabled", "tts_enabled", "stt_enabled", "observer_enabled"];
    setRestartNeeded(
      changes.some(([k, v]) =>
        RESTART_KEYS.includes(k) && configValues != null &&
        (configValues as unknown as Record<string, unknown>)[k] !== v),
    );
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

  return (
    <div className="flex h-full flex-col">
      <PageTitle
        eyebrow="Configuration"
        title="HOUSEHOLD SETTINGS"
        sub="Adjust Bantz's operational parameters. He will tolerate the changes."
        right={
          <div className="flex flex-col items-end gap-2">
            <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => { setS(DEFAULTS); }}
              className="border border-obsidian-500 bg-obsidian-800 px-3 py-2 font-ui text-[10px] font-bold uppercase tracking-widest text-obsidian-200 transition-colors hover:border-obsidian-300 hover:text-fg-primary"
            >
              Reset
            </button>
            <button
              type="button"
              onClick={applyChanges}
              disabled={!wsSend}
              className={`flex items-center gap-2 border px-4 py-2 font-ui text-[10px] font-bold uppercase tracking-widest transition-all ${
                saved
                  ? "border-[#2E7D32] bg-[#2E7D32]/10 text-[#2E7D32]"
                  : wsSend
                    ? "border-ember-500 bg-ember-500/10 text-ember-500 hover:bg-ember-500 hover:text-obsidian-900"
                    : "cursor-not-allowed border-obsidian-500 text-obsidian-400"
              }`}
            >
              <Save size={11} strokeWidth={1.5} />
              {saved ? "Saved" : wsSend ? "Apply Changes" : "Backend Offline"}
            </button>
            </div>
            {restartNeeded && (
              <div className="border border-gold-500/40 bg-gold-500/5 px-3 py-1 font-terminal text-[10px] text-gold-400">
                ⚡ Restart required for voice/observer changes to take effect.
              </div>
            )}
          </div>
        }
      />

      <div className="grid min-h-0 flex-1 grid-cols-2 gap-5 overflow-y-auto pr-2">
        {/* MODELS */}
        <Section title="Models" hint="LLM backends">
          <Row label="Conversation Provider" hint="Who answers — applies to UI + CLI">
            <select
              value={s.provider}
              onChange={(e) => set("provider", e.target.value)}
              className="w-56 border border-obsidian-500 bg-obsidian-850 px-3 py-2 font-terminal text-[12px] text-ember-300 focus:border-ember-500 focus:outline-none"
            >
              {PROVIDERS.map((p) => <option key={p.key} value={p.key}>{p.label}</option>)}
            </select>
          </Row>
          <Row label="Ollama Model" hint="Local inference">
            {ollamaStatus === "loading" ? (
              <span className="font-terminal text-[12px] text-obsidian-300">Loading models…</span>
            ) : ollamaStatus === "error" ? (
              <span className="font-terminal text-[12px] text-rose-300">Ollama unreachable</span>
            ) : (
              <select
                value={s.ollamaModel}
                onChange={(e) => set("ollamaModel", e.target.value)}
                disabled={ollamaModels.length === 0}
                className="w-56 border border-obsidian-500 bg-obsidian-850 px-3 py-2 font-terminal text-[12px] text-ember-300 focus:border-ember-500 focus:outline-none"
              >
                {ollamaModels.length === 0 ? (
                  <option disabled>No models installed</option>
                ) : (
                  <>
                    {/* keep the saved model selectable even if not in the local list */}
                    {s.ollamaModel && !ollamaModels.includes(s.ollamaModel) && (
                      <option key={s.ollamaModel}>{s.ollamaModel}</option>
                    )}
                    {ollamaModels.map((m) => <option key={m}>{m}</option>)}
                  </>
                )}
              </select>
            )}
          </Row>
          {s.provider === "claude" && (
            <>
              <Row label="Claude API Key" hint="Anthropic · sk-ant-…">
                <div className="flex items-center gap-2">
                  <input
                    type={showClaudeKey ? "text" : "password"}
                    value={s.claudeKey}
                    onChange={(e) => set("claudeKey", e.target.value)}
                    placeholder="sk-ant-…"
                    className="w-44 border border-obsidian-500 bg-obsidian-850 px-3 py-2 font-terminal text-[12px] tracking-widest text-gold-400 focus:border-gold-500 focus:outline-none"
                  />
                  <button
                    type="button"
                    onClick={() => setShowClaudeKey((v) => !v)}
                    className="grid h-9 w-9 place-items-center border border-obsidian-500 text-obsidian-200 transition-colors hover:border-ember-500 hover:text-ember-500"
                    aria-label={showClaudeKey ? "Hide key" : "Show key"}
                  >
                    {showClaudeKey ? <EyeOff size={14} /> : <Eye size={14} />}
                  </button>
                </div>
              </Row>
              <Row label="Claude Model" hint="Anthropic model">
                <select
                  value={s.claudeModel}
                  onChange={(e) => set("claudeModel", e.target.value)}
                  className="w-56 border border-obsidian-500 bg-obsidian-850 px-3 py-2 font-terminal text-[12px] text-ember-300 focus:border-ember-500 focus:outline-none"
                >
                  {CLAUDE_MODELS.map((m) => <option key={m}>{m}</option>)}
                </select>
              </Row>
            </>
          )}
          <Row label="Gemini API Key" hint="Remote fallback">
            <div className="flex items-center gap-2">
              <input
                type={showKey ? "text" : "password"}
                value={s.geminiKey}
                onChange={(e) => set("geminiKey", e.target.value)}
                className="w-44 border border-obsidian-500 bg-obsidian-850 px-3 py-2 font-terminal text-[12px] tracking-widest text-gold-400 focus:border-gold-500 focus:outline-none"
              />
              <button
                type="button"
                onClick={() => setShowKey((v) => !v)}
                className="grid h-9 w-9 place-items-center border border-obsidian-500 text-obsidian-200 transition-colors hover:border-ember-500 hover:text-ember-500"
                aria-label={showKey ? "Hide key" : "Show key"}
              >
                {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
            </div>
          </Row>
          <Row label="Gemini Enabled" hint="Enable remote fallback">
            <Switch on={s.geminiEnabled} onChange={(v) => set("geminiEnabled", v)} />
          </Row>
          <Row label="Context Window" hint="Tokens">
            <select
              value={s.ctx}
              onChange={(e) => set("ctx", e.target.value)}
              className="w-56 border border-obsidian-500 bg-obsidian-850 px-3 py-2 font-terminal text-[12px] text-ember-300 focus:border-ember-500 focus:outline-none"
            >
              {["8k", "16k", "32k", "64k", "128k"].map((v) => <option key={v}>{v}</option>)}
            </select>
          </Row>
        </Section>

        {/* VOICE */}
        <Section title="Voice Pipeline" hint="Speech I/O">
          <Row label="Wake Word" hint='"Bantz" · 95% threshold'>
            <Switch on={s.wakeWord} onChange={(v) => set("wakeWord", v)} />
          </Row>
          <Row label="Speech-to-Text" hint="Whisper.cpp · local">
            <Switch on={s.stt} onChange={(v) => set("stt", v)} />
          </Row>
          <Row label="Text-to-Speech" hint="Piper · British male">
            <Switch on={s.tts} onChange={(v) => set("tts", v)} />
          </Row>
        </Section>

        {/* LOCALIZATION */}
        <Section title="Localization" hint="Language">
          <Row label="Language" hint="Conversation locale">
            <Segmented
              value={s.lang}
              onChange={(v) => set("lang", v as SettingsState["lang"])}
              options={[
                { k: "TR", label: "Türkçe" },
                { k: "EN", label: "English" },
              ]}
            />
          </Row>
        </Section>

        {/* BONDING */}
        <Section title="Affinity" hint="Read-only · earned not set">
          <div className="space-y-3 px-5 py-4">
            <div className="flex items-baseline justify-between">
              <div className="font-ui text-[10px] font-bold uppercase tracking-widest text-obsidian-200">
                Bonding Score
              </div>
              <div className="font-display text-[24px] font-extrabold leading-none text-gold-400">
                {s.bonding}
                <span className="text-[12px] opacity-60">/100</span>
              </div>
            </div>
            <div className="flex gap-0.5">
              {Array.from({ length: 20 }).map((_, i) => {
                const filled = i < Math.round(s.bonding / 5);
                const intense = filled && i >= 14;
                return (
                  <span
                    key={i}
                    className={`h-3 flex-1 ${
                      filled
                        ? intense ? "bg-gold-300 shadow-gold" : "bg-gold-500"
                        : "bg-obsidian-700"
                    }`}
                  />
                );
              })}
            </div>
            <div className="flex items-center justify-between font-terminal text-[10px] text-obsidian-300">
              <span>BARELY TOLERATED</span>
              <span>TRUSTED</span>
              <span>FAMILIAR</span>
            </div>
          </div>
        </Section>

        {/* APPEARANCE */}
        <Section title="Appearance" hint="Accent">
          <Row label="Accent Color" hint="Affects highlights & focus rings">
            <div className="flex items-center gap-2">
              {ACCENTS.map((a) => {
                const sel = s.accent === a.key;
                return (
                  <button
                    key={a.key}
                    type="button"
                    onClick={() => set("accent", a.key)}
                    title={a.label}
                    className={`relative h-9 w-9 border-2 transition-all ${sel ? "border-fg-primary" : "border-transparent hover:border-obsidian-300"}`}
                    style={{ background: a.hex, boxShadow: sel ? `0 0 12px ${a.hex}` : "" }}
                  >
                    {sel && <Check size={14} className="absolute inset-0 m-auto text-obsidian-900" strokeWidth={2.5} />}
                  </button>
                );
              })}
            </div>
          </Row>
          <Row label="Night Shift" hint="Velvet palette · low-light hours">
            <Switch on={s.nightShift} onChange={(v) => set("nightShift", v)} />
          </Row>
          <Row label="CRT Effects" hint="Grain + scanlines">
            <Switch on={s.crt} onChange={(v) => set("crt", v)} />
          </Row>
        </Section>

        {/* BEHAVIOR */}
        <Section title="Bantz Behaviour" hint="Personality dials">
          <Row label="Verbosity" hint="Broadcast frequency">
            <Segmented
              value={s.verbosity}
              onChange={(v) => set("verbosity", v as SettingsState["verbosity"])}
              options={[
                { k: "silent",        label: "Silent" },
                { k: "standard",      label: "Standard" },
                { k: "insufferable",  label: "Insufferable" },
              ]}
            />
          </Row>
          <Row label="Autonomy" hint="Permission to act unbidden">
            <Segmented
              value={s.autonomy}
              onChange={(v) => set("autonomy", v as SettingsState["autonomy"])}
              options={[
                { k: "low",  label: "Low" },
                { k: "med",  label: "Medium" },
                { k: "high", label: "High" },
                { k: "abs",  label: "Absolute" },
              ]}
            />
          </Row>
          <Row label="Mood Bias" hint="Default disposition">
            <Segmented
              value={s.mood}
              onChange={(v) => set("mood", v as SettingsState["mood"])}
              options={[
                { k: "tolerant",  label: "Tolerant" },
                { k: "impatient", label: "Impatient" },
                { k: "resigned",  label: "Resigned" },
              ]}
            />
          </Row>
          <Row label="Session Distillation" hint="Summarise conversations">
            <Switch on={s.distillation} onChange={(v) => set("distillation", v)} />
          </Row>
          <Row label="Shell Confirmation" hint="Confirm destructive commands">
            <Switch on={s.shellConfirm} onChange={(v) => set("shellConfirm", v)} />
          </Row>
          <Row label="Background Observer" hint="Monitor screen activity">
            <Switch on={s.observerEnabled} onChange={(v) => set("observerEnabled", v)} />
          </Row>
        </Section>
      </div>
    </div>
  );
}

function Section({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  return (
    <section className="border border-obsidian-700 bg-obsidian-850/70">
      <PanelHeader title={title} subtitle={hint} />
      <div className="divide-y divide-obsidian-800">{children}</div>
    </section>
  );
}

function Row({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <div className="flex items-center justify-between px-5 py-3 transition-colors hover:bg-obsidian-800/30">
      <div>
        <div className="font-ui text-[12px] font-semibold uppercase tracking-wider text-fg-primary">
          {label}
        </div>
        {hint && <div className="mt-0.5 font-terminal text-[10px] text-obsidian-300">{hint}</div>}
      </div>
      {children}
    </div>
  );
}

function Switch({ on, onChange }: { on: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="bz-switch">
      <input type="checkbox" checked={on} onChange={(e) => onChange(e.target.checked)} />
      <span className="track">
        <span className="knob" />
      </span>
    </label>
  );
}

function Segmented<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T;
  onChange: (v: T) => void;
  options: { k: T; label: string }[];
}) {
  return (
    <div className="inline-flex border border-obsidian-500">
      {options.map((o) => (
        <button
          key={o.k}
          type="button"
          onClick={() => onChange(o.k)}
          className={`px-3 py-2 font-ui text-[10px] font-bold uppercase tracking-widest transition-colors ${
            value === o.k
              ? "bg-ember-500 text-obsidian-900"
              : "bg-obsidian-850 text-obsidian-200 hover:text-ember-300"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
