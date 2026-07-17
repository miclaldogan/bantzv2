import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { VoiceOverlay } from "./overlay/VoiceOverlay";
import "./index.css";

// The compact voice overlay window loads the same bundle with ?overlay=1
// (see src-tauri/tauri.conf.json "overlay" window).
const isOverlay = new URLSearchParams(window.location.search).has("overlay");

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    {isOverlay ? <VoiceOverlay /> : <App />}
  </React.StrictMode>,
);
