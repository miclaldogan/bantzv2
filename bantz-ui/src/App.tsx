import { useEffect, useMemo, useRef, useState } from "react";
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
import { useAppStore, type ConfigValues, type ServiceItem, type Task } from "./store/useAppStore";

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
      priority: "low",
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
      priority: "medium",
      progress: 0,
    });
  }

  return result;
}

// ─────────────────────────────────────────────────────────────────────────────

export default function App() {
  const [active, setActive] = useState<string>("chat");
  const [clock, setClock] = useState(() => fmtClock(new Date()));

  const pushChat         = useAppStore((s) => s.pushChat);
  const pushVital        = useAppStore((s) => s.pushVital);
  const setStreamingText = useAppStore((s) => s.setStreamingText);
  const pushLog          = useAppStore((s) => s.pushLog);
  const pushAlert        = useAppStore((s) => s.pushAlert);
  const setTasks         = useAppStore((s) => s.setTasks);
  const setServices      = useAppStore((s) => s.setServices);
  const setConfigValues  = useAppStore((s) => s.setConfigValues);
  const setWsSend        = useAppStore((s) => s.setWsSend);
  const alertCount       = useAppStore((s) => s.alerts.length);

  // Accumulates streaming tokens between "token" and "done" messages.
  const streamAccumRef = useRef<string>("");

  const { status, lastMessage, attempts, send } = useWebSocket({
    url: WS_URL,
    reconnectDelay: 2000,
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

  // Route incoming WS messages into the store.
  useEffect(() => {
    if (!lastMessage) return;
    const d = lastMessage.data as { type?: string; [k: string]: unknown } | undefined;
    if (!d || typeof d !== "object" || !d.type) return;

    switch (d.type) {

      case "vitals": {
        const v = d as {
          cpu: number; ram_used: number; ram_total: number;
          disk_used: number; disk_total: number;
          vram_used: number; vram_total: number;
        };
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
        setStreamingText(streamAccumRef.current);
        break;
      }

      case "done": {
        const final = streamAccumRef.current;
        streamAccumRef.current = "";
        setStreamingText(null);
        if (final.trim()) {
          pushChat({ role: "bantz", text: final });
        }
        break;
      }

      case "broadcast": {
        // Fallback for proactive server-push messages (health alerts,
        // observer notifications) that are not part of a chat request/
        // response cycle and therefore never go through token+done.
        const b = d as { text?: string };
        pushChat({ role: "bantz", text: String(b.text ?? "") });
        break;
      }

      case "log": {
        const l = d as { msg: string; level: string };
        const colonIdx = l.msg.indexOf(": ");
        const src = colonIdx > 0 ? l.msg.slice(0, colonIdx) : "bantz";
        const msg = colonIdx > 0 ? l.msg.slice(colonIdx + 2) : l.msg;
        pushLog({
          t: Date.now(),
          sev: LEVEL_MAP[l.level] ?? "INFO",
          src,
          msg,
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

      // "pong", "config_ack", "task_created", "alert_dismissed" — intentionally ignored
      default:
        break;
    }
  }, [lastMessage, pushChat, pushVital, setStreamingText, pushLog, pushAlert,
      setTasks, setServices, setConfigValues]);

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
