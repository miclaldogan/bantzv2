import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Header } from "./components/Header";
import { Sidebar, PAGE_LABEL } from "./components/Sidebar";
import { PageHost } from "./components/PageHost";
import { ChatPage } from "./pages/ChatPage";
import { VitalsPage } from "./pages/VitalsPage";
import { TasksPage } from "./pages/TasksPage";
import { LogsPage } from "./pages/LogsPage";
import { AlertsPage } from "./pages/AlertsPage";
import { SettingsPage } from "./pages/SettingsPage";
import { useWebSocket } from "./hooks/useWebSocket";
import { useAppStore, type Anomaly, type ConfigValues, type MediaItem, type ServiceItem, type Task, type TaskPriority, type VoiceState } from "./store/useAppStore";

async function getWindow() {
  try {
    const m = await import("@tauri-apps/api/window");
    return m.getCurrentWindow();
  } catch {
    return null;
  }
}

const WS_URL = "ws://localhost:8765";

// Map Python log levels → UI severity labels.
const LEVEL_MAP: Record<string, "INFO" | "WARN" | "ERROR" | "CRITICAL"> = {
  debug: "INFO", info: "INFO", warning: "WARN", error: "ERROR", critical: "CRITICAL",
};

// ── Backend data mappers ──────────────────────────────────────────────────

interface BackendReminder {
  id: string;
  title: string;
  fire_at: string;
  repeat: string;
  fired: number;
  snoozed_until: string | null;
  trigger_place: string | null;
}

interface BackendJob {
  id: string;
  name: string;
  next_run: string;
  trigger: string;
}

// Safety net: strip model reasoning blocks that may leak into the stream.
function stripThinking(text: string): string {
  return text.replace(/<thinking>[\s\S]*?<\/thinking>/g, "").trimStart();
}

// Derive a real priority from a job's name/id so the CRITICAL/HIGH filter
// chips match live data instead of everything collapsing to "low".
function jobPriority(job: BackendJob): TaskPriority {
  const hay = `${job.name} ${job.id}`.toLowerCase();
  if (/\b(critical|urgent)\b/.test(hay)) return "critical";
  if (hay.includes("overnight") || hay.includes("poll")) return "high";
  if (hay.includes("maintenance") || hay.includes("reflection")) return "low";
  return "low";
}

function mapBackendTasks(reminders: BackendReminder[], jobs: BackendJob[]): Task[] {
  const now = Date.now();
  const result: Task[] = [];

  // System jobs from APScheduler
  for (const job of jobs) {
    const nextTs = job.next_run === "paused" ? null : new Date(job.next_run).getTime();
    let eta = "paused";
    if (nextTs) {
      const diff = nextTs - now;
      if (diff > 0) {
        const h = Math.floor(diff / 3600000);
        const m = Math.floor((diff % 3600000) / 60000);
        eta = h > 0 ? `in ${h}h ${m}m` : `in ${m}m`;
      } else {
        eta = "overdue";
      }
    }
    result.push({
      id: `job-${job.id}`,
      title: job.name,
      detail: job.trigger,
      status: "active",
      eta,
      priority: jobPriority(job),
      progress: 0,
    });
  }

  // User reminders
  for (const r of reminders) {
    const fireTs = r.fire_at ? new Date(r.fire_at).getTime() : null;
    const isRecurring = r.repeat && r.repeat !== "none";

    let status: Task["status"];
    if (r.fired) {
      status = "done";
    } else if (isRecurring) {
      status = "active";
    } else {
      status = "queued";
    }

    let eta = "—";
    if (fireTs && !r.fired) {
      const diff = fireTs - now;
      if (diff > 0) {
        const h = Math.floor(diff / 3600000);
        const m = Math.floor((diff % 3600000) / 60000);
        eta = h > 0 ? `in ${h}h ${m}m` : `in ${m}m`;
      } else {
        eta = isRecurring ? r.repeat : "overdue";
      }
    } else if (isRecurring) {
      eta = r.repeat;
    }

    let detail = "";
    if (r.trigger_place) detail = `location: ${r.trigger_place}`;
    else if (r.repeat !== "none") detail = `repeats: ${r.repeat}`;
    else detail = "one-time reminder";

    result.push({
      id: `rem-${r.id}`,
      title: r.title,
      detail,
      status,
      eta,
      priority: /\b(critical|urgent)\b/i.test(r.title) ? "critical" : "medium",
      progress: 0,
    });
  }

  return result;
}

// ─────────────────────────────────────────────────────────────────────────────

export default function App() {
  // Active page lives in the store so other pages (e.g. Anomaly Watch's
  // "Investigate") can navigate here.
  const active    = useAppStore((s) => s.activePage);
  const setActive = useAppStore((s) => s.setActivePage);
  const [clock, setClock] = useState(() => fmtClock(new Date()));

  const pushChat         = useAppStore((s) => s.pushChat);
  const pushVital        = useAppStore((s) => s.pushVital);
  const setStreamingText = useAppStore((s) => s.setStreamingText);
  const pushLog          = useAppStore((s) => s.pushLog);
  const pushAlert        = useAppStore((s) => s.pushAlert);
  const setTasks         = useAppStore((s) => s.setTasks);
  const setServices      = useAppStore((s) => s.setServices);
  const setConfigValues  = useAppStore((s) => s.setConfigValues);
  const setAnomalies     = useAppStore((s) => s.setAnomalies);
  const setResearch      = useAppStore((s) => s.setResearch);
  const setVoiceState    = useAppStore((s) => s.setVoiceState);
  const setWsSend        = useAppStore((s) => s.setWsSend);
  const alertCount       = useAppStore((s) => s.anomalies.length);

  // Accumulates streaming tokens between "token" and "done" messages.
  const streamAccumRef = useRef<string>("");

  // Process each WS frame synchronously (called from ws.onmessage directly).
  // This avoids React 18 automatic batching coalescing rapid token→done pairs
  // into a single render where the token effect never runs.
  const handleWsMessage = useCallback(
    (msg: { data?: unknown }) => {
      const d = msg.data as { type?: string; [k: string]: unknown } | undefined;
      if (!d || typeof d !== "object" || !d.type) return;

      switch (d.type) {

        case "vitals": {
          const v = d as {
            cpu: number; ram_used: number; ram_total: number;
            disk_used: number; disk_total: number;
            vram_used: number; vram_total: number;
            anomalies?: Anomaly[];
          };
          if (Array.isArray(v.anomalies)) setAnomalies(v.anomalies);
          const ramPct  = v.ram_total  > 0 ? (v.ram_used  / v.ram_total)  * 100 : 0;
          const diskPct = v.disk_total > 0 ? (v.disk_used / v.disk_total) * 100 : 0;
          pushVital({
            t:          Date.now(),
            cpu:        v.cpu,
            mem:        ramPct,
            disk:       diskPct,
            net:        0,
            ram_used:   v.ram_used,
            ram_total:  v.ram_total,
            disk_used:  v.disk_used,
            disk_total: v.disk_total,
            vram_used:  v.vram_used,
            vram_total: v.vram_total,
          });
          break;
        }

        case "token": {
          const tok = d as { text?: string };
          streamAccumRef.current += tok.text ?? "";
          setStreamingText(stripThinking(streamAccumRef.current));
          break;
        }

        case "done": {
          const final = stripThinking(streamAccumRef.current);
          streamAccumRef.current = "";
          setStreamingText(null);
          if (final.trim()) {
            pushChat({ role: "bantz", text: final });
          }
          break;
        }

        case "broadcast": {
          const b = d as { text?: string };
          pushChat({ role: "bantz", text: String(b.text ?? "") });
          break;
        }

        case "log": {
          const l = d as { msg: string; level: string };
          const colonIdx = l.msg.indexOf(": ");
          const src = colonIdx > 0 ? l.msg.slice(0, colonIdx) : "bantz";
          const msg2 = colonIdx > 0 ? l.msg.slice(colonIdx + 2) : l.msg;
          pushLog({
            t: Date.now(),
            sev: LEVEL_MAP[l.level] ?? "INFO",
            src,
            msg: msg2,
          });
          break;
        }

        case "alert": {
          const a = d as { title?: string; reason?: string; source?: string };
          const source = String(a.source ?? "bantz");
          pushAlert({
            severity: source === "observer" ? "critical" : "warning",
            category: "service",
            title:  String(a.title  ?? "Backend alert"),
            detail: String(a.reason ?? ""),
            ts:     Date.now(),
            source,
          });
          break;
        }

        case "tasks": {
          const td = d as { reminders?: BackendReminder[]; jobs?: BackendJob[] };
          setTasks(mapBackendTasks(td.reminders ?? [], td.jobs ?? []));
          break;
        }

        case "config": {
          const cd = d as { values?: Partial<ConfigValues> };
          if (cd.values && Object.keys(cd.values).length > 0) {
            setConfigValues(cd.values as ConfigValues);
          }
          break;
        }

        case "services": {
          const sd = d as { services?: ServiceItem[] };
          if (sd.services && Array.isArray(sd.services)) {
            setServices(sd.services);
          }
          break;
        }

        case "research_progress": {
          const r = d as {
            stage?: string; detail?: string; elapsed?: number; state?: string;
          };
          const state =
            r.state === "done" || r.state === "cancelled" ? r.state : "running";
          setResearch({
            stage:   String(r.stage ?? ""),
            detail:  String(r.detail ?? ""),
            elapsed: Number(r.elapsed ?? 0),
            state,
          });
          // Leave the terminal state visible briefly, then clear the indicator.
          if (state !== "running") {
            setTimeout(() => setResearch(null), 4000);
          }
          break;
        }

        case "voice_state": {
          const v = d as { state?: string };
          const s = String(v.state ?? "idle");
          const known: VoiceState[] = ["idle", "wake", "listening", "processing", "speaking"];
          setVoiceState(known.includes(s as VoiceState) ? (s as VoiceState) : "idle");
          break;
        }

        case "voice_transcript": {
          // What the user said to the wake word — echo it into the chat log
          // so the spoken conversation is visible in the Broadcast Channel.
          const v = d as { text?: string };
          const text = String(v.text ?? "").trim();
          if (text) pushChat({ role: "user", text: `🎙 ${text}` });
          break;
        }

        case "images": {
          const im = d as { topic?: string; items?: MediaItem[] };
          const items = Array.isArray(im.items)
            ? im.items.filter((x) => x && typeof x.image === "string" && x.image)
            : [];
          if (items.length) {
            pushChat({
              role: "bantz",
              text: im.topic ? `Visuals — ${im.topic}` : "Visuals",
              images: items,
            });
          }
          break;
        }

        default:
          break;
      }
    },
    [pushChat, pushVital, setStreamingText, pushLog, pushAlert,
     setTasks, setServices, setConfigValues, setAnomalies, setResearch,
     setVoiceState],
  );

  const { status, attempts, send } = useWebSocket({
    url: WS_URL,
    reconnectDelay: 2000,
    onMessage: handleWsMessage,
  });

  // Register send in store so any page can send WS messages.
  useEffect(() => {
    setWsSend(send as (msg: Record<string, unknown>) => boolean);
    return () => setWsSend(null);
  }, [send, setWsSend]);

  // Wall clock
  useEffect(() => {
    const id = window.setInterval(() => setClock(fmtClock(new Date())), 1000);
    return () => window.clearInterval(id);
  }, []);

  // Synthetic vitals — only run when the backend is not connected.
  useEffect(() => {
    if (status === "open") return;
    const id = window.setInterval(() => {
      pushVital({
        t: Date.now(),
        cpu: 12 + Math.round(Math.random() * 22),
        mem: 38 + Math.round(Math.random() * 4),
        disk: 91,
        net: 100 + Math.round(Math.random() * 300),
        ram_used: 0, ram_total: 0,
        disk_used: 0, disk_total: 0,
        vram_used: 0, vram_total: 0,
      });
    }, 1400);
    return () => window.clearInterval(id);
  }, [pushVital, status]);


  // Connection status announcements
  useEffect(() => {
    if (status === "open") {
      streamAccumRef.current = "";
      setStreamingText(null);
      pushChat({ role: "system", text: `link established · ${WS_URL}` });
      // Request initial data — backend also pushes on connect, this is a safety
      send({ type: "get_tasks" });
      send({ type: "get_config" });
    } else if (status === "closed" && attempts > 0) {
      pushChat({
        role: "system",
        text: `link severed · attempting reconnect (${attempts})`,
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  async function handleMinimize() { (await getWindow())?.minimize(); }
  async function handleMaximize() {
    const w = await getWindow();
    if (!w) return;
    (await w.isMaximized()) ? w.unmaximize() : w.maximize();
  }
  async function handleClose() { (await getWindow())?.close(); }

  function handleSend(text: string) {
    const ok = send({ type: "chat", text });
    if (!ok) {
      pushChat({
        role: "system",
        text: "backend unreachable — message queued locally only",
      });
    }
  }

  // Memoize so PageHost doesn't see a new map on every clock tick.
  const pages = useMemo(
    () => ({
      chat:     <ChatPage wsStatus={status} onSend={handleSend} />,
      vitals:   <VitalsPage />,
      tasks:    <TasksPage />,
      logs:     <LogsPage wsConnected={status === "open"} />,
      alerts:   <AlertsPage />,
      settings: <SettingsPage />,
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [status],
  );

  return (
    <div className="bantz-texture relative flex h-screen w-screen flex-col overflow-hidden bg-obsidian-900 ring-1 ring-obsidian-700">
      <Header
        wsStatus={status}
        wsAttempts={attempts}
        clock={clock}
        activeLabel={PAGE_LABEL[active]}
        onMinimize={handleMinimize}
        onMaximize={handleMaximize}
        onClose={handleClose}
      />

      <div className="flex min-h-0 flex-1">
        <Sidebar active={active} onSelect={setActive} alertCount={alertCount} />
        <main className="min-h-0 min-w-0 flex-1 overflow-hidden p-6">
          <PageHost active={active} pages={pages} />
        </main>
      </div>

      <footer className="flex h-6 items-center justify-between border-t border-obsidian-700 bg-obsidian-850/95 px-4 font-terminal text-[10px] tracking-wider text-obsidian-300">
        <span>BANTZ v0.1.0 · operations center · {PAGE_LABEL[active]?.toLowerCase()}</span>
        <span>
          {status === "open"
            ? "// transmission stable"
            : status === "connecting"
              ? "// dialing backend…"
              : "// awaiting transmission"}
        </span>
      </footer>
    </div>
  );
}

function fmtClock(d: Date) {
  return d.toLocaleTimeString([], { hour12: false });
}
