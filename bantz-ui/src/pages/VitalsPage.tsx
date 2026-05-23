import { useEffect, useState } from "react";
import {
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useAppStore } from "../store/useAppStore";
import { ArcGauge } from "../components/ArcGauge";
import { PageTitle, PanelHeader } from "../components/primitives";
import type { ServiceItem } from "../store/useAppStore";

interface NetSample {
  t: number;
  down: number;
  up: number;
}

function seedNet(): NetSample[] {
  const now = Date.now();
  const out: NetSample[] = [];
  for (let i = 39; i >= 0; i--) {
    out.push({
      t: now - i * 1000,
      down: 200 + Math.round(Math.random() * 900 + Math.sin(i / 3) * 120),
      up: 80 + Math.round(Math.random() * 200 + Math.cos(i / 4) * 40),
    });
  }
  return out;
}

const PROCESSES = [
  { pid: 4127, name: "ollama serve", cpu: 38.2, mem: 12.4, user: "bantz" },
  { pid: 1029, name: "bantz-daemon", cpu: 14.8, mem: 6.1, user: "bantz" },
  { pid: 8842, name: "firefox-bin", cpu: 9.3, mem: 4.7, user: "misa" },
  { pid: 812, name: "Xorg", cpu: 4.1, mem: 1.8, user: "root" },
  { pid: 2256, name: "redis-server", cpu: 2.7, mem: 0.9, user: "bantz" },
];


export function VitalsPage() {
  const vitals   = useAppStore((s) => s.vitals);
  const services = useAppStore((s) => s.services);
  const [net, setNet] = useState<NetSample[]>(seedNet);
  const [gpuTemp, setGpuTemp] = useState(68);

  useEffect(() => {
    const id = window.setInterval(() => {
      setNet((d) => [
        ...d.slice(-39),
        {
          t: Date.now(),
          down: 200 + Math.round(Math.random() * 900 + Math.sin(Date.now() / 2000) * 120),
          up: 80 + Math.round(Math.random() * 200 + Math.cos(Date.now() / 2500) * 40),
        },
      ]);
      setGpuTemp((t) => Math.max(55, Math.min(85, t + (Math.random() * 2 - 0.8))));
    }, 1200);
    return () => window.clearInterval(id);
  }, []);

  const latest  = vitals[vitals.length - 1] || { cpu: 0, mem: 0, disk: 0, ram_used: 0, ram_total: 0, disk_used: 0, disk_total: 0, vram_used: 0, vram_total: 0 };
  const lastNet = net[net.length - 1] || { down: 0, up: 0 };
  const temp    = Math.round(gpuTemp);

  // Use real backend values when available, fall back to synthetic.
  const ramUsed   = latest.ram_total  > 0 ? latest.ram_used  : null;
  const ramTotal  = latest.ram_total  > 0 ? latest.ram_total  : null;
  const diskUsed  = latest.disk_total > 0 ? latest.disk_used  : null;
  const diskTotal = latest.disk_total > 0 ? latest.disk_total : null;
  const vramUsed  = latest.vram_total > 0 ? latest.vram_used  : null;
  const vramTotal = latest.vram_total > 0 ? latest.vram_total : null;
  const ramPct    = ramTotal  ? Math.round((ramUsed!  / ramTotal!)  * 100) : Math.round(latest.mem);
  const diskPct   = diskTotal ? Math.round((diskUsed! / diskTotal!) * 100) : Math.round(latest.disk || 91);
  const vramPct   = vramTotal ? Math.round((vramUsed! / vramTotal!) * 100) : 38;

  return (
    <div className="flex h-full flex-col">
      <PageTitle
        eyebrow="Diagnostics"
        title="SYSTEM VITALS"
        sub="Hardware telemetry sampled at 1Hz. Anomalies surface in Anomaly Watch."
        right={
          <div className="flex items-center gap-2 font-terminal text-[10px] tracking-widest text-obsidian-200">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-ember-500 shadow-ember" />
            SAMPLING · 1 Hz
          </div>
        }
      />

      <div className="grid grid-cols-4 gap-3">
        <ArcGauge label="CPU"  value={latest.cpu} detail="i9-13900K · 24c/32t"       tone={latest.cpu > 75 ? "rose" : "ember"} />
        <ArcGauge label="RAM"  value={ramPct}     detail={ramTotal  ? `${ramUsed!.toFixed(1)} / ${ramTotal.toFixed(1)} GB`     : "— / — GB"}           tone="ember" />
        <ArcGauge label="VRAM" value={vramPct}    detail={vramTotal ? `${(vramUsed!/1024).toFixed(1)} / ${(vramTotal/1024).toFixed(1)} GB · GPU` : "— / — GB · GPU"} tone="gold" />
        <ArcGauge label="DISK" value={diskPct}    detail={diskTotal ? `${diskUsed!.toFixed(0)} / ${diskTotal.toFixed(0)} GB`   : "/home · — GB"}        tone={diskPct > 85 ? "rose" : "ember"} />
      </div>

      <div className="mt-3 grid min-h-[200px] grid-cols-[1fr_320px] gap-3">
        <section className="flex flex-col border border-obsidian-700 bg-obsidian-850/70">
          <PanelHeader
            title="Network I/O"
            subtitle="kB/s · 40s window"
            right={
              <div className="flex items-center gap-4 font-terminal text-[10px]">
                <span className="text-ember-500">▼ {lastNet.down} kB/s</span>
                <span className="text-velvet-200">▲ {lastNet.up} kB/s</span>
              </div>
            }
          />
          <div className="min-h-0 flex-1 p-3">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={net} margin={{ top: 8, right: 8, bottom: 0, left: -20 }}>
                <defs>
                  <linearGradient id="netDown" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#FF4500" stopOpacity={0.45} />
                    <stop offset="100%" stopColor="#FF4500" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="netUp" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#00BFFF" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#00BFFF" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="t" hide />
                <YAxis
                  domain={[0, 1400]}
                  tick={{ fill: "#555", fontSize: 10, fontFamily: "Share Tech Mono" }}
                  axisLine={false}
                  tickLine={false}
                  width={28}
                />
                <Tooltip
                  contentStyle={{
                    background: "#0D0A07",
                    border: "1px solid #FF4500",
                    fontFamily: "Share Tech Mono",
                    fontSize: 11,
                    color: "#F0E8D8",
                    borderRadius: 0,
                  }}
                  labelFormatter={(t) =>
                    new Date(t as number).toLocaleTimeString([], { hour12: false })
                  }
                  formatter={(v: number, n: string) => [`${v} kB/s`, n.toUpperCase()]}
                />
                <Area type="stepAfter" dataKey="down" stroke="#FF4500" strokeWidth={1.5} fill="url(#netDown)" isAnimationActive={false} name="down" />
                <Area type="stepAfter" dataKey="up"   stroke="#00BFFF" strokeWidth={1.2} fill="url(#netUp)"   isAnimationActive={false} name="up" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="flex flex-col border border-obsidian-700 bg-obsidian-850/70">
          <PanelHeader title="GPU Thermal" subtitle="°C" />
          <div className="flex flex-1 items-stretch gap-4 px-5 py-4">
            <div className="relative w-7 flex-shrink-0">
              <div className="absolute inset-0 border border-obsidian-500 bg-obsidian-900" />
              <div
                className="absolute bottom-0 left-0 right-0 transition-all"
                style={{
                  height: `${Math.min(100, temp)}%`,
                  background:
                    temp > 80
                      ? "linear-gradient(to top, #CC1111, #FF4500)"
                      : temp > 65
                        ? "linear-gradient(to top, #FF4500, #FF8C00)"
                        : "linear-gradient(to top, #00BFFF, #00BFFF)",
                  boxShadow: temp > 80 ? "0 0 12px #CC1111" : "0 0 10px #FF4500",
                }}
              />
              {[20, 40, 60, 80].map((v) => (
                <span
                  key={v}
                  className="absolute left-full ml-1 font-terminal text-[9px] text-obsidian-300"
                  style={{ bottom: `calc(${v}% - 5px)` }}
                >
                  {v}°
                </span>
              ))}
            </div>
            <div className="flex flex-1 flex-col justify-between pl-6">
              <div>
                <div
                  className="font-display text-[40px] font-extrabold leading-none"
                  style={{ color: temp > 80 ? "#CC1111" : temp > 65 ? "#FF4500" : "#00BFFF" }}
                >
                  {temp}°<span className="ml-0.5 text-[16px] opacity-70">C</span>
                </div>
                <div className="mt-2 font-ui text-[9px] font-bold uppercase tracking-widest text-obsidian-200">
                  Junction Temp
                </div>
              </div>
              <div className="space-y-1 font-terminal text-[11px]">
                <div className="flex justify-between"><span className="text-obsidian-300">FAN</span><span className="text-fg-primary">68%</span></div>
                <div className="flex justify-between"><span className="text-obsidian-300">POWER</span><span className="text-fg-primary">214 W</span></div>
                <div className="flex justify-between"><span className="text-obsidian-300">CLOCK</span><span className="text-fg-primary">2.41 GHz</span></div>
              </div>
            </div>
          </div>
        </section>
      </div>

      <div className="mt-3 grid min-h-0 flex-1 grid-cols-[1fr_1fr] gap-3">
        <section className="flex flex-col border border-obsidian-700 bg-obsidian-850/70">
          <PanelHeader
            title="Top Processes"
            subtitle="by CPU · top 5"
            right={<span className="font-terminal text-[10px] text-obsidian-300">42 total · 0 zombies</span>}
          />
          <div className="flex-1 overflow-y-auto">
            <table className="w-full font-terminal text-[12px]">
              <thead>
                <tr className="border-b border-obsidian-700 text-[9px] uppercase tracking-widest text-obsidian-300">
                  <th className="px-4 py-2 text-left font-ui font-bold">PID</th>
                  <th className="px-2 py-2 text-left font-ui font-bold">Process</th>
                  <th className="px-2 py-2 text-right font-ui font-bold">CPU</th>
                  <th className="px-2 py-2 text-right font-ui font-bold">MEM</th>
                  <th className="px-4 py-2 text-right font-ui font-bold">User</th>
                </tr>
              </thead>
              <tbody>
                {PROCESSES.map((p) => (
                  <tr key={p.pid} className="border-b border-obsidian-800">
                    <td className="px-4 py-2 text-obsidian-200">{p.pid}</td>
                    <td className="px-2 py-2 text-fg-primary">{p.name}</td>
                    <td className="px-2 py-2 text-right">
                      <span className={p.cpu > 20 ? "text-ember-500" : p.cpu > 10 ? "text-gold-400" : "text-obsidian-200"}>
                        {p.cpu.toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-2 py-2 text-right text-obsidian-200">{p.mem.toFixed(1)}%</td>
                    <td className="px-4 py-2 text-right text-obsidian-300">{p.user}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="flex flex-col border border-obsidian-700 bg-obsidian-850/70">
          <PanelHeader
            title="Service Status"
            subtitle={`${services.length} services`}
            right={
              <span className="font-terminal text-[10px]">
                <span className="text-ember-500">{services.filter((s) => s.status === "online").length} OK</span>{" "}·{" "}
                <span className="text-gold-400">{services.filter((s) => s.status === "degraded").length} DEGRADED</span>{" "}·{" "}
                <span className="text-rose-300">{services.filter((s) => s.status === "offline").length} OFFLINE</span>
              </span>
            }
          />
          <div className="grid flex-1 grid-cols-1 gap-2 overflow-y-auto p-3">
            {services.map((svc: ServiceItem) => {
              const sc =
                svc.status === "online"
                  ? { dot: "bg-ember-500 shadow-ember", text: "text-ember-500", border: "border-ember-500/40" }
                  : svc.status === "degraded"
                    ? { dot: "bg-gold-400 shadow-gold", text: "text-gold-400", border: "border-gold-500/40" }
                    : { dot: "bg-rose-500", text: "text-rose-300", border: "border-rose-500/40" };
              return (
                <div key={svc.name} className={`flex items-center gap-4 border ${sc.border} bg-obsidian-800/60 px-4 py-3`}>
                  <span className={`block h-2.5 w-2.5 rounded-full ${sc.dot}`} />
                  <div className="min-w-0 flex-1">
                    <div className="font-ui text-[12px] font-bold uppercase tracking-wider text-fg-primary">
                      {svc.name}
                      {svc.port && (
                        <span className="ml-2 font-terminal text-[10px] text-obsidian-300">:{svc.port}</span>
                      )}
                    </div>
                    <div className="font-terminal text-[10px] text-obsidian-200">{svc.detail}</div>
                  </div>
                  <div className="text-right">
                    <div className={`font-ui text-[9px] font-bold uppercase tracking-widest ${sc.text}`}>
                      {svc.status}
                    </div>
                    <div className="mt-0.5 font-terminal text-[10px] text-obsidian-300">{svc.uptime}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      </div>
    </div>
  );
}
