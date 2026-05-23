import { useState } from "react";
import { Plus, ChevronDown, ChevronRight, X } from "lucide-react";
import { PageTitle, SectionLabel, Empty } from "../components/primitives";
import { useAppStore, type Task, type TaskPriority } from "../store/useAppStore";

const PRIORITY: Record<TaskPriority, { color: string; border: string; bg: string; label: string }> = {
  critical: { color: "text-rose-300",     border: "border-rose-500",     bg: "bg-rose-500/10",  label: "CRITICAL" },
  high:     { color: "text-ember-500",    border: "border-ember-500",    bg: "bg-ember-500/10", label: "HIGH" },
  medium:   { color: "text-gold-400",     border: "border-gold-500",     bg: "bg-gold-500/10",  label: "MEDIUM" },
  low:      { color: "text-obsidian-200", border: "border-obsidian-500", bg: "bg-obsidian-800", label: "LOW" },
};

function TaskRow({ t }: { t: Task }) {
  const p = PRIORITY[t.priority];
  return (
    <div
      className={`group border ${
        t.status === "active"
          ? "border-ember-500/30"
          : t.status === "queued"
            ? "border-gold-500/30"
            : "border-obsidian-700"
      } bg-obsidian-800/60 px-5 py-4 transition-colors hover:border-ember-500 ${t.status === "done" ? "opacity-60" : ""}`}
    >
      <div className="flex items-start gap-4">
        <span
          className={`mt-1.5 block h-2 w-2 flex-shrink-0 rounded-full ${
            t.status === "active"
              ? "bg-ember-500 shadow-ember"
              : t.status === "queued"
                ? "bg-gold-400"
                : "bg-obsidian-400"
          }`}
        />
        <div className="min-w-0 flex-1">
          <div className="mb-1 flex items-center gap-3">
            <div className="font-ui text-[14px] font-semibold uppercase tracking-wide text-fg-primary">
              {t.title}
            </div>
            <span
              className={`border px-1.5 py-0.5 font-ui text-[8px] font-bold uppercase tracking-widest ${p.color} ${p.border} ${p.bg}`}
            >
              {p.label}
            </span>
          </div>
          <div className="font-terminal text-[12px] text-obsidian-200">{t.detail}</div>
          {t.status === "active" && t.progress > 0 && (
            <div className="mt-3 flex items-center gap-3">
              <div className="relative h-1.5 flex-1 overflow-hidden bg-obsidian-700">
                <div
                  className="absolute inset-y-0 left-0 bg-ember-500 shadow-ember transition-all"
                  style={{ width: `${t.progress}%` }}
                />
                {[25, 50, 75].map((v) => (
                  <span
                    key={v}
                    className="absolute inset-y-0 w-px bg-obsidian-900/80"
                    style={{ left: `${v}%` }}
                  />
                ))}
              </div>
              <span className="w-10 text-right font-terminal text-[11px] text-ember-300">
                {t.progress}%
              </span>
            </div>
          )}
        </div>
        <div className="flex-shrink-0 text-right">
          <div className="font-terminal text-[10px] uppercase tracking-widest text-obsidian-300">
            {t.eta}
          </div>
        </div>
      </div>
    </div>
  );
}

export function TasksPage() {
  const tasks    = useAppStore((s) => s.tasks);
  const wsSend   = useAppStore((s) => s.wsSend);

  const [filter, setFilter]       = useState<"all" | TaskPriority>("all");
  const [showDone, setShowDone]   = useState(false);
  const [addOpen, setAddOpen]     = useState(false);
  const [newText, setNewText]     = useState("");

  const visible = tasks.filter((t) => filter === "all" || t.priority === filter);
  const active  = visible.filter((t) => t.status === "active");
  const queued  = visible.filter((t) => t.status === "queued");
  const done    = visible.filter((t) => t.status === "done");

  function submitNew() {
    const text = newText.trim();
    if (!text) return;
    if (wsSend) {
      wsSend({ type: "new_task", text });
    }
    setNewText("");
    setAddOpen(false);
  }

  return (
    <div className="flex h-full flex-col">
      <PageTitle
        eyebrow={`${active.length} active · ${queued.length} queued`}
        title="DIRECTIVES"
        sub="Standing instructions Bantz operates without further consultation."
        right={
          <button
            type="button"
            onClick={() => setAddOpen((v) => !v)}
            className="flex items-center gap-2 border border-ember-500 bg-ember-500/10 px-4 py-2 font-ui text-[10px] font-bold uppercase tracking-widest text-ember-500 transition-all hover:bg-ember-500 hover:text-obsidian-900"
          >
            {addOpen ? <X size={14} strokeWidth={1.5} /> : <Plus size={14} strokeWidth={1.5} />}
            {addOpen ? "Cancel" : "New Directive"}
          </button>
        }
      />

      {/* Inline new-task input */}
      {addOpen && (
        <div className="mb-4 flex items-center gap-2 border border-ember-500/40 bg-obsidian-850/80 px-4 py-3">
          <span className="font-terminal text-[12px] text-ember-300">»</span>
          <input
            autoFocus
            value={newText}
            onChange={(e) => setNewText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") { e.preventDefault(); submitNew(); }
              if (e.key === "Escape") { setAddOpen(false); setNewText(""); }
            }}
            placeholder={wsSend ? "Describe the directive…" : "Backend offline — cannot create directives"}
            disabled={!wsSend}
            className="flex-1 bg-transparent font-terminal text-[13px] text-fg-primary placeholder:text-obsidian-300 focus:outline-none disabled:opacity-50"
          />
          <button
            type="button"
            onClick={submitNew}
            disabled={!newText.trim() || !wsSend}
            className="border border-ember-500 px-3 py-1 font-ui text-[9px] font-bold uppercase tracking-widest text-ember-500 transition-all hover:bg-ember-500 hover:text-obsidian-900 disabled:cursor-not-allowed disabled:border-obsidian-500 disabled:text-obsidian-300 disabled:hover:bg-transparent"
          >
            Add
          </button>
        </div>
      )}

      <div className="mb-4 flex items-center gap-2">
        {(["all", "critical", "high", "medium", "low"] as const).map((k) => (
          <button
            key={k}
            type="button"
            onClick={() => setFilter(k)}
            className={`border px-3 py-1.5 font-ui text-[10px] font-bold uppercase tracking-widest transition-all ${
              filter === k
                ? "border-ember-500 bg-ember-500/15 text-ember-500"
                : "border-obsidian-500 text-obsidian-200 hover:border-ember-500 hover:text-ember-300"
            }`}
          >
            {k === "all" ? "All" : k.charAt(0).toUpperCase() + k.slice(1)}
          </button>
        ))}
        <div className="flex-1" />
        <div className="font-terminal text-[10px] tracking-widest text-obsidian-300">
          {visible.length} OF {tasks.length} SHOWN
        </div>
      </div>

      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto pr-1">
        <section>
          <SectionLabel count={active.length} accent="ember">
            Active
          </SectionLabel>
          <div className="mt-2 space-y-2">
            {active.length === 0 && <Empty text="No active directives." />}
            {active.map((t) => <TaskRow key={t.id} t={t} />)}
          </div>
        </section>

        <section>
          <SectionLabel count={queued.length} accent="gold">
            Queued
          </SectionLabel>
          <div className="mt-2 space-y-2">
            {queued.length === 0 && <Empty text="Nothing queued." />}
            {queued.map((t) => <TaskRow key={t.id} t={t} />)}
          </div>
        </section>

        <section>
          <button
            type="button"
            onClick={() => setShowDone((s) => !s)}
            className="group flex items-center gap-2"
          >
            {showDone ? (
              <ChevronDown size={14} className="text-obsidian-200 group-hover:text-ember-500" />
            ) : (
              <ChevronRight size={14} className="text-obsidian-200 group-hover:text-ember-500" />
            )}
            <SectionLabel count={done.length} accent="muted">
              Completed
            </SectionLabel>
          </button>
          {showDone && (
            <div className="mt-2 space-y-2">
              {done.length === 0 && <Empty text="No completed directives." />}
              {done.map((t) => <TaskRow key={t.id} t={t} />)}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
