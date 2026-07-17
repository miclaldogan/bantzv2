import { useState } from "react";
import { useWebSocket } from "../hooks/useWebSocket";
import type { VoiceState } from "../store/useAppStore";

const WS_URL = "ws://localhost:8765";

const LABEL: Record<VoiceState, string> = {
  idle:       "STANDING BY",
  wake:       "YES?",
  listening:  "LISTENING",
  processing: "THINKING",
  speaking:   "SPEAKING",
};

const ACCENT: Record<VoiceState, string> = {
  idle:       "#4a4a52",
  wake:       "#e8b44a",
  listening:  "#e8b44a",
  processing: "#d97742",
  speaking:   "#e05d38",
};

const KNOWN: VoiceState[] = ["idle", "wake", "listening", "processing", "speaking"];

/**
 * Compact always-on-top voice indicator (second Tauri window, ?overlay=1).
 * Holds its own WS connection so it works even when the main Operations
 * Center window is closed. Position/pin on Hyprland comes from windowrules
 * matched on the "Bantz Voice" title.
 */
export function VoiceOverlay() {
  const [state, setState] = useState<VoiceState>("idle");
  const [line, setLine] = useState("");

  const { status } = useWebSocket({
    url: WS_URL,
    reconnectDelay: 2000,
    onMessage: (msg) => {
      const d = msg.data as { type?: string; state?: string; text?: string } | undefined;
      if (!d || typeof d !== "object" || !d.type) return;
      if (d.type === "voice_state") {
        const s = String(d.state ?? "idle") as VoiceState;
        setState(KNOWN.includes(s) ? s : "idle");
        if (s === "wake" || s === "listening") setLine("");
      } else if (d.type === "voice_transcript") {
        const t = String(d.text ?? "").trim();
        if (t) setLine(`“${t}”`);
      }
    },
  });

  const accent = ACCENT[state];
  const active = state !== "idle";
  const offline = status !== "open";

  return (
    <div
      data-tauri-drag-region
      style={{
        width: "100%",
        height: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "transparent",
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        userSelect: "none",
        cursor: "default",
      }}
    >
      <style>{`
        @keyframes bantz-bar {
          0%, 100% { transform: scaleY(0.25); }
          50%      { transform: scaleY(1); }
        }
        @keyframes bantz-pulse {
          0%, 100% { opacity: 0.55; }
          50%      { opacity: 1; }
        }
      `}</style>
      <div
        data-tauri-drag-region
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          width: "calc(100% - 16px)",
          padding: "10px 14px",
          borderRadius: 10,
          border: `1px solid ${active ? accent : "#2e2e34"}`,
          background: "rgba(16, 16, 20, 0.92)",
          boxShadow: active ? `0 0 18px ${accent}44` : "0 2px 10px rgba(0,0,0,0.5)",
          transition: "border-color 200ms, box-shadow 200ms",
          overflow: "hidden",
        }}
      >
        {/* waveform bars */}
        <div style={{ display: "flex", alignItems: "center", gap: 3, height: 26 }}>
          {[0, 1, 2, 3, 4].map((i) => (
            <div
              key={i}
              style={{
                width: 3,
                height: 22,
                borderRadius: 2,
                background: offline ? "#3a3a40" : accent,
                transformOrigin: "center",
                transform: "scaleY(0.25)",
                animation:
                  !offline && (state === "listening" || state === "speaking")
                    ? `bantz-bar ${0.7 + i * 0.13}s ease-in-out ${i * 0.09}s infinite`
                    : state === "processing" && !offline
                      ? `bantz-pulse 1.1s ease-in-out ${i * 0.12}s infinite`
                      : "none",
              }}
            />
          ))}
        </div>

        <div style={{ minWidth: 0, flex: 1 }}>
          <div
            style={{
              fontSize: 10,
              fontWeight: 700,
              letterSpacing: "0.18em",
              color: offline ? "#5a5a62" : accent,
            }}
          >
            {offline ? "LINK DOWN" : `BANTZ · ${LABEL[state]}`}
          </div>
          {line && !offline && (
            <div
              style={{
                fontSize: 11,
                color: "#b9b9c0",
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
                marginTop: 2,
              }}
            >
              {line}
            </div>
          )}
        </div>

        <div
          style={{
            width: 7,
            height: 7,
            borderRadius: "50%",
            background: offline ? "#7a3030" : active ? accent : "#3f6e3f",
            animation: active && !offline ? "bantz-pulse 1.4s ease-in-out infinite" : "none",
            flexShrink: 0,
          }}
        />
      </div>
    </div>
  );
}
