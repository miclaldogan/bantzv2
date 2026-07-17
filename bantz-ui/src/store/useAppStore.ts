import { create } from "zustand";

// Hero images surfaced by web_news / web_research ("images" WS frames).
export interface MediaItem {
  image: string; // hero image URL
  title: string;
  url: string;   // source article
}

export interface ChatTurn {
  id: string;
  role: "bantz" | "user" | "system";
  text: string;
  ts: number;
  images?: MediaItem[];
}

// Live voice pipeline state ("voice_state" WS frames) — drives the overlay
// and any listening/speaking indicators.
export type VoiceState = "idle" | "wake" | "listening" | "processing" | "speaking";

export interface VitalSample {
  t: number;
  cpu: number;        // %
  mem: number;        // % (ram_used/ram_total*100, for chart compat)
  disk: number;       // % (disk_used/disk_total*100)
  net: number;        // KB/s — still synthetic (backend doesn't push net I/O)
  ram_used: number;   // GB
  ram_total: number;  // GB
  disk_used: number;  // GB
  disk_total: number; // GB
  vram_used: number;  // MB
  vram_total: number; // MB
}

export type TaskPriority = "critical" | "high" | "medium" | "low";
export type TaskStatus   = "active" | "queued" | "done";

export interface Task {
  id: string;
  title: string;
  detail: string;
  status: TaskStatus;
  eta: string;
  priority: TaskPriority;
  progress: number;
}

export type LogSeverity = "INFO" | "WARN" | "ERROR" | "CRITICAL";

export interface LogEntry {
  id: string;
  t: number;
  sev: LogSeverity;
  src: string;
  msg: string;
}

export type AlertSeverity = "critical" | "warning" | "info";
export type AlertCategory =
  | "thermal" | "disk" | "session" | "service"
  | "network" | "package";

export interface AlertItem {
  id: string;
  severity: AlertSeverity;
  category: AlertCategory;
  title: string;
  detail: string;
  ts: number;
  source: string;
}

// Live anomalies derived by the backend (resource pressure + recent error
// logs) and pushed on the periodic vitals message. Replaced wholesale each
// tick — stable ids per condition mean no duplicates.
export interface Anomaly {
  id: string;
  title: string;
  severity: AlertSeverity;
  description: string;
  source: string;
  timestamp: number;
}

// Live deep-research (web_research) progress, pushed as structured
// "research_progress" frames (#490). Null when no run is active; the terminal
// states ("done"/"cancelled") clear it shortly after arriving.
export interface ResearchProgress {
  stage: string;   // "searching" | "working" | "done" | "cancelled"
  detail: string;  // readable one-liner
  elapsed: number; // seconds since the run started
  state: "running" | "done" | "cancelled";
}

export type ServiceStatus = "online" | "degraded" | "offline";

export interface ServiceItem {
  name: string;
  port: number | null;
  status: ServiceStatus;
  uptime: string;
  detail: string;
}

export interface ConfigValues {
  llm_provider: string;
  ollama_model: string;
  ollama_base_url: string;
  anthropic_api_key: string;
  anthropic_model: string;
  gemini_enabled: boolean;
  gemini_api_key: string;
  language: string;
  tts_enabled: boolean;
  stt_enabled: boolean;
  wake_word_enabled: boolean;
  distillation_enabled: boolean;
  shell_confirm_destructive: boolean;
  observer_enabled: boolean;
  verbosity: string;
  autonomy: string;
  mood_bias: string;
}

interface AppState {
  chat: ChatTurn[];
  vitals: VitalSample[];
  tasks: Task[];
  streamingText: string | null;
  logs: LogEntry[];
  alerts: AlertItem[];
  anomalies: Anomaly[];
  dismissedIds: Set<string>;
  activePage: string;
  services: ServiceItem[];
  configValues: ConfigValues | null;
  research: ResearchProgress | null;
  voiceState: VoiceState;
  wsSend: ((msg: Record<string, unknown>) => boolean) | null;

  pushChat: (turn: Omit<ChatTurn, "id" | "ts"> & Partial<Pick<ChatTurn, "ts">>) => void;
  pushVital: (sample: VitalSample) => void;
  setTasks: (tasks: Task[]) => void;
  setStreamingText: (text: string | null) => void;
  pushLog: (entry: Omit<LogEntry, "id">) => void;
  pushAlert: (alert: Omit<AlertItem, "id">) => void;
  dismissAlert: (id: string) => void;
  dismissAllAlerts: () => void;
  setAnomalies: (anomalies: Anomaly[]) => void;
  dismissAnomaly: (id: string) => void;
  snoozeAnomaly: (id: string) => void;
  setActivePage: (page: string) => void;
  setServices: (services: ServiceItem[]) => void;
  setConfigValues: (cv: ConfigValues) => void;
  setResearch: (progress: ResearchProgress | null) => void;
  setVoiceState: (state: VoiceState) => void;
  setWsSend: (fn: ((msg: Record<string, unknown>) => boolean) | null) => void;
}

let _id = 0;
const nid = () => `t${Date.now()}-${++_id}`;

// ── Seed data ─────────────────────────────────────────────────────────────

function seedVitals(): VitalSample[] {
  const now = Date.now();
  const out: VitalSample[] = [];
  for (let i = 29; i >= 0; i--) {
    const cpu = 10 + Math.round(Math.random() * 18 + Math.sin(i / 4) * 6);
    out.push({
      t: now - i * 1000,
      cpu,
      mem: 38 + Math.round(Math.random() * 4),
      disk: 91,
      net: 120 + Math.round(Math.random() * 240),
      ram_used: 0, ram_total: 0,
      disk_used: 0, disk_total: 0,
      vram_used: 0, vram_total: 0,
    });
  }
  return out;
}

const SEED_TASKS: Task[] = [
  { id: "monitor-disk",  title: "Monitor disk usage",         detail: "91% → alerting at 95%",       status: "active", eta: "watching",    priority: "high",   progress: 62 },
  { id: "email-triage",  title: "Email triage",               detail: "3 flagged for your attention", status: "active", eta: "continuous",  priority: "medium", progress: 28 },
  { id: "package-watch", title: "Package watch",              detail: "Tracking 2 deliveries",        status: "active", eta: "Thu delivery",priority: "low",    progress: 45 },
  { id: "weekly-report", title: "Weekly report compilation",  detail: "Scheduled · 18:00",            status: "queued", eta: "18:00",       priority: "high",   progress: 0  },
  { id: "cron-audit",    title: "Audit recurring cron jobs",  detail: "backup.sh — 14 misses",        status: "queued", eta: "this evening",priority: "critical", progress: 0 },
  { id: "schedule-reorg",title: "Schedule reorganisation",    detail: "3 conflicts resolved",         status: "done",   eta: "completed 1d",priority: "medium", progress: 100 },
];

const SEED_LOGS: Omit<LogEntry, "id">[] = (
  [
    { t: 0, sev: "INFO",     src: "kernel",    msg: "Bantz v0.1.0 initialised — kernel 6.8.0-rc4" },
    { t: 0, sev: "INFO",     src: "comms",     msg: "WebSocket carrier opened on :8765" },
    { t: 0, sev: "INFO",     src: "scheduler", msg: "loaded 8 directives from store" },
    { t: 0, sev: "WARN",     src: "disk-mon",  msg: "/home partition at 91% — alert threshold not yet breached" },
    { t: 0, sev: "INFO",     src: "ollama",    msg: "model warmed: llama3.1:70b (38s)" },
    { t: 0, sev: "ERROR",    src: "redis",     msg: "high eviction rate — 142 keys/s (threshold 50)" },
    { t: 0, sev: "INFO",     src: "telegram",  msg: "polling 2 chats · last update 4s ago" },
    { t: 0, sev: "WARN",     src: "thermal",   msg: "GPU junction temp climbing — 68°C, fan ramping" },
    { t: 0, sev: "CRITICAL", src: "systemd",   msg: "neo4j.service failed to start (exit 137 · OOM killer)" },
    { t: 0, sev: "INFO",     src: "kernel",    msg: "awaiting acknowledgement…" },
  ] as Omit<LogEntry, "id">[]
).map((l, i) => ({ ...l, t: Date.now() - (10 - i) * 1800 }));

const SEED_SERVICES: ServiceItem[] = [
  { name: "Ollama",   port: 11434, status: "offline", uptime: "—", detail: "awaiting probe" },
  { name: "Gemini",   port: null,  status: "offline", uptime: "—", detail: "awaiting probe" },
  { name: "Telegram", port: 443,   status: "offline", uptime: "—", detail: "awaiting probe" },
  { name: "Redis",    port: 6379,  status: "offline", uptime: "—", detail: "awaiting probe" },
  { name: "Neo4j",    port: 7687,  status: "offline", uptime: "—", detail: "awaiting probe" },
];

// ── Snooze persistence (survives reloads) ───────────────────────────────────
// bantz.snoozed = { [anomalyId]: expiresAtEpochMs }
const SNOOZE_LS_KEY = "bantz.snoozed";

function readSnoozed(): Record<string, number> {
  try {
    const raw = localStorage.getItem(SNOOZE_LS_KEY);
    const obj = raw ? JSON.parse(raw) : {};
    return obj && typeof obj === "object" ? (obj as Record<string, number>) : {};
  } catch {
    return {};
  }
}

function writeSnoozed(map: Record<string, number>): void {
  try {
    localStorage.setItem(SNOOZE_LS_KEY, JSON.stringify(map));
  } catch {
    /* storage unavailable — snooze just won't persist */
  }
}

function removeSnoozed(id: string): void {
  const map = readSnoozed();
  if (id in map) {
    delete map[id];
    writeSnoozed(map);
  }
}

// On module load: drop already-expired snoozes, persist the pruned map, and
// return the survivors so the store can pre-populate dismissedIds + schedule
// their remaining timers.
function restoreSnoozes(): Record<string, number> {
  const now = Date.now();
  const survivors: Record<string, number> = {};
  for (const [id, exp] of Object.entries(readSnoozed())) {
    if (typeof exp === "number" && exp > now) survivors[id] = exp;
  }
  writeSnoozed(survivors);
  return survivors;
}

const _SNOOZE_RESTORE = restoreSnoozes();

export const useAppStore = create<AppState>((set) => ({
  chat: [
    {
      id: nid(),
      role: "bantz",
      text: "Operations Center initialised. I am, as ever, mildly disappointed but operational.",
      ts: Date.now() - 4000,
    },
    {
      id: nid(),
      role: "system",
      text: "awaiting backend transmission on ws://localhost:8765 …",
      ts: Date.now() - 2000,
    },
  ],
  vitals:       seedVitals(),
  tasks:        SEED_TASKS,
  streamingText: null,
  logs:         SEED_LOGS.map((l) => ({ ...l, id: nid() })),
  alerts:       [],
  anomalies:    [],
  dismissedIds: new Set<string>(Object.keys(_SNOOZE_RESTORE)),
  activePage:   "chat",
  services:     SEED_SERVICES,
  configValues: null,
  research:     null,
  voiceState:   "idle",
  wsSend:       null,

  pushChat: (turn) =>
    set((s) => ({
      chat: [...s.chat, { id: nid(), ts: Date.now(), ...turn } as ChatTurn].slice(-200),
    })),

  pushVital: (sample) =>
    set((s) => ({ vitals: [...s.vitals, sample].slice(-60) })),

  setTasks: (tasks) => set({ tasks }),

  setStreamingText: (text) => set({ streamingText: text }),

  pushLog: (entry) =>
    set((s) => ({
      logs: [...s.logs, { ...entry, id: nid() }].slice(-200),
    })),

  pushAlert: (alert) =>
    set((s) => ({
      alerts: [{ ...alert, id: nid() }, ...s.alerts],
    })),

  dismissAlert: (id) =>
    set((s) => ({ alerts: s.alerts.filter((a) => a.id !== id) })),

  dismissAllAlerts: () => set({ alerts: [] }),

  // Live anomalies arrive on each vitals push and replace the list wholesale.
  // Dismissed ids stay hidden; one whose condition has cleared (no longer in
  // the incoming push) is auto-undismissed so it can surface again later.
  setAnomalies: (incoming) =>
    set((s) => {
      const incomingIds = new Set(incoming.map((a) => a.id));
      const dismissedIds = new Set(
        [...s.dismissedIds].filter((id) => incomingIds.has(id)),
      );
      return {
        dismissedIds,
        anomalies: incoming.filter((a) => !dismissedIds.has(a.id)),
      };
    }),
  // Dismiss persists via dismissedIds (survives re-pushes); also drop it from
  // the visible list now for immediate feedback.
  dismissAnomaly: (id) =>
    set((s) => {
      const dismissedIds = new Set(s.dismissedIds);
      dismissedIds.add(id);
      return {
        dismissedIds,
        anomalies: s.anomalies.filter((a) => a.id !== id),
      };
    }),
  // Snooze = dismiss that auto-expires after 1h, so the anomaly can resurface
  // if the condition still holds. Client-side only.
  snoozeAnomaly: (id) => {
    const map = readSnoozed();
    map[id] = Date.now() + 3_600_000;
    writeSnoozed(map);
    set((s) => {
      const dismissedIds = new Set(s.dismissedIds);
      dismissedIds.add(id);
      return { dismissedIds, anomalies: s.anomalies.filter((a) => a.id !== id) };
    });
    setTimeout(() => {
      set((s) => {
        const dismissedIds = new Set(s.dismissedIds);
        dismissedIds.delete(id);
        return { dismissedIds };
      });
      removeSnoozed(id);
    }, 3_600_000);
  },
  setActivePage: (page) => set({ activePage: page }),

  setServices: (services) => set({ services }),

  setConfigValues: (cv) => set({ configValues: cv }),

  setResearch: (progress) => set({ research: progress }),

  setVoiceState: (state) => set({ voiceState: state }),

  setWsSend: (fn) => set({ wsSend: fn }),
}));

// Re-arm timers for snoozes restored from localStorage, using each one's
// *remaining* duration (not a fresh hour). On fire, un-snooze + clear storage.
for (const [id, expiresAt] of Object.entries(_SNOOZE_RESTORE)) {
  setTimeout(() => {
    useAppStore.setState((s) => {
      const dismissedIds = new Set(s.dismissedIds);
      dismissedIds.delete(id);
      return { dismissedIds };
    });
    removeSnoozed(id);
  }, Math.max(0, expiresAt - Date.now()));
}
