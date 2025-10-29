import { useState } from "react";
import { Menu, MenuButton, MenuItem, MenuItems } from "@headlessui/react";
import { ConnectionStatus } from "../types";
import type { Metrics } from "../types";

interface ControlsProps {
  status: ConnectionStatus;
  metrics: Metrics;
  onConnect: () => void;
  onDisconnect: () => void;
  onStart: () => void;
  onStop: () => void;
  isGenerating: boolean;
  sourcePrompt: string;
  onSourcePromptChange: (prompt: string) => void;
  targetPrompt: string;
  onTargetPromptChange: (prompt: string) => void;
  promptC: string;
  onPromptCChange: (prompt: string) => void;
  promptD: string;
  onPromptDChange: (prompt: string) => void;
  reconnectAttempts?: number;
}

export function Controls({
  status,
  metrics,
  onConnect,
  onDisconnect,
  onStart,
  onStop,
  isGenerating,
  sourcePrompt,
  onSourcePromptChange,
  targetPrompt,
  onTargetPromptChange,
  promptC,
  onPromptCChange,
  promptD,
  onPromptDChange,
  reconnectAttempts = 0,
}: ControlsProps) {
  const isConnected = status === ConnectionStatus.CONNECTED;
  const isConnecting = status === ConnectionStatus.CONNECTING;

  // Draggable state
  const [position, setPosition] = useState({ x: 24, y: 24 });
  const [isDragging, setIsDragging] = useState(false);

  const handlePointerDown = (e: React.PointerEvent) => {
    setIsDragging(true);
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!isDragging) return;
    setPosition((prev) => ({
      x: prev.x + e.movementX,
      y: prev.y + e.movementY,
    }));
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    setIsDragging(false);
    (e.target as HTMLElement).releasePointerCapture(e.pointerId);
  };

  const getStatusColor = () => {
    switch (status) {
      case ConnectionStatus.CONNECTED:
        return "bg-[#B5CC9A]"; // bright sage green
      case ConnectionStatus.CONNECTING:
        return "bg-yellow-500";
      case ConnectionStatus.ERROR:
        return "bg-red-500";
      default:
        return "bg-gray-500";
    }
  };

  const getStatusText = () => {
    switch (status) {
      case ConnectionStatus.CONNECTED:
        return "Connected";
      case ConnectionStatus.CONNECTING:
        return "Connecting...";
      case ConnectionStatus.ERROR:
        return "Error";
      default:
        return "Disconnected";
    }
  };

  return (
    <div
      className="fixed left-0 top-0 z-10 w-80 select-none font-mono"
      style={{
        transform: `translate(${position.x}px, ${position.y}px)`,
        cursor: isDragging ? "grabbing" : "default",
      }}
    >
      <div className="rounded-xl border border-[#8B9A7E]/20 bg-gradient-to-br from-[#8B9A7E]/10 via-gray-900/15 to-[#9CA986]/10 p-6 shadow-2xl backdrop-blur-lg">
        {/* Drag handle header */}
        <div
          className="mb-6 flex cursor-grab items-center gap-3 active:cursor-grabbing"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
        >
          <div className="relative">
            <div
              className={`h-3 w-3 rounded-full ${getStatusColor()} ${isConnected ? "animate-pulse" : ""}`}
            />
            {isConnected && (
              <div className="absolute inset-0 h-3 w-3 animate-ping rounded-full bg-[#B5CC9A] opacity-75" />
            )}
          </div>

          <h2 className="text-lg font-semibold text-white">
            {getStatusText()}
          </h2>

          {reconnectAttempts > 0 && (
            <span className="ml-auto text-xs text-white/50">
              Retry {reconnectAttempts}
            </span>
          )}
        </div>

        {/* Connection controls */}
        <div className="mb-6 space-y-3">
          <div className="flex gap-0">
            <button
              onClick={onConnect}
              disabled={isConnected || isConnecting}
              className="inline-flex flex-1 items-center justify-center gap-2 rounded-md bg-[#8B9A7E]/20 px-3 py-1.5 text-sm/6 font-semibold text-white shadow-inner shadow-[#8B9A7E]/20 transition focus:outline-none data-hover:bg-[#8B9A7E]/30 data-focus:outline data-focus:outline-1 data-focus:outline-[#8B9A7E] disabled:cursor-not-allowed disabled:opacity-40"
            >
              {isConnecting ? "Connecting..." : "Connect"}
            </button>

            <button
              onClick={onDisconnect}
              disabled={!isConnected && !isConnecting}
              className="inline-flex flex-1 items-center justify-center gap-2 rounded-md bg-[#8B9A7E]/20 px-3 py-1.5 text-sm/6 font-semibold text-white shadow-inner shadow-[#8B9A7E]/20 transition focus:outline-none data-hover:bg-[#8B9A7E]/30 data-focus:outline data-focus:outline-1 data-focus:outline-[#8B9A7E] disabled:cursor-not-allowed disabled:opacity-40"
            >
              Disconnect
            </button>
          </div>

          <div className="flex gap-0">
            <button
              onClick={onStart}
              disabled={!isConnected || isGenerating}
              className="inline-flex flex-1 items-center justify-center gap-2 rounded-md bg-green-600/80 px-3 py-1.5 text-sm/6 font-semibold text-white shadow-inner shadow-green-700/20 transition focus:outline-none data-hover:bg-green-600 data-focus:outline data-focus:outline-1 data-focus:outline-green-500 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <svg className="size-4" viewBox="0 0 16 16" fill="currentColor">
                <path d="M3 3.732a1.5 1.5 0 0 1 2.305-1.265l6.706 4.267a1.5 1.5 0 0 1 0 2.531l-6.706 4.268A1.5 1.5 0 0 1 3 12.267V3.732Z" />
              </svg>
              Start
            </button>

            <button
              onClick={onStop}
              disabled={!isConnected || !isGenerating}
              className="inline-flex flex-1 items-center justify-center gap-2 rounded-md bg-red-600/80 px-3 py-1.5 text-sm/6 font-semibold text-white shadow-inner shadow-red-700/20 transition focus:outline-none data-hover:bg-red-600 data-focus:outline data-focus:outline-1 data-focus:outline-red-500 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <svg className="size-4" viewBox="0 0 16 16" fill="currentColor">
                <rect x="4" y="4" width="8" height="8" rx="1" />
              </svg>
              Stop
            </button>
          </div>

          <div>
            <label className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-white/50">
              Source Prompt
            </label>
            <input
              type="text"
              value={sourcePrompt}
              onChange={(e) => onSourcePromptChange(e.target.value)}
              disabled={isGenerating}
              placeholder="moldy burger in sewer..."
              className="w-full rounded-md border border-[#8B9A7E]/20 bg-[#8B9A7E]/10 px-3 py-2 text-sm text-white placeholder-white/30 shadow-inner backdrop-blur-sm transition focus:border-[#8B9A7E]/40 focus:outline-none focus:ring-1 focus:ring-[#8B9A7E]/40 disabled:cursor-not-allowed disabled:opacity-40"
            />
          </div>

          <div>
            <label className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-white/50">
              Target Prompt
            </label>
            <input
              type="text"
              value={targetPrompt}
              onChange={(e) => onTargetPromptChange(e.target.value)}
              disabled={isGenerating}
              placeholder="steamy burger..."
              className="w-full rounded-md border border-[#8B9A7E]/20 bg-[#8B9A7E]/10 px-3 py-2 text-sm text-white placeholder-white/30 shadow-inner backdrop-blur-sm transition focus:border-[#8B9A7E]/40 focus:outline-none focus:ring-1 focus:ring-[#8B9A7E]/40 disabled:cursor-not-allowed disabled:opacity-40"
            />
          </div>

          <div>
            <label className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-white/50">
              Prompt C
            </label>
            <input
              type="text"
              value={promptC}
              onChange={(e) => onPromptCChange(e.target.value)}
              disabled={isGenerating}
              placeholder="crispy fries..."
              className="w-full rounded-md border border-[#8B9A7E]/20 bg-[#8B9A7E]/10 px-3 py-2 text-sm text-white placeholder-white/30 shadow-inner backdrop-blur-sm transition focus:border-[#8B9A7E]/40 focus:outline-none focus:ring-1 focus:ring-[#8B9A7E]/40 disabled:cursor-not-allowed disabled:opacity-40"
            />
          </div>

          <div>
            <label className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-white/50">
              Prompt D
            </label>
            <input
              type="text"
              value={promptD}
              onChange={(e) => onPromptDChange(e.target.value)}
              disabled={isGenerating}
              placeholder="fresh salad..."
              className="w-full rounded-md border border-[#8B9A7E]/20 bg-[#8B9A7E]/10 px-3 py-2 text-sm text-white placeholder-white/30 shadow-inner backdrop-blur-sm transition focus:border-[#8B9A7E]/40 focus:outline-none focus:ring-1 focus:ring-[#8B9A7E]/40 disabled:cursor-not-allowed disabled:opacity-40"
            />
          </div>
        </div>

        {/* Metrics */}
        <div className="space-y-3 border-t border-white/5 pt-6">
          <h3 className="text-xs font-medium uppercase tracking-wide text-white/50">
            Performance
          </h3>

          <MetricRow
            label="Client FPS"
            value={metrics.fps}
            unit=""
            highlight={isConnected}
          />

          {metrics.inferenceTimeEmaMs !== undefined && (
            <MetricRow
              label="Inference Time"
              value={metrics.inferenceTimeEmaMs.toFixed(1)}
              unit="ms"
              highlight={isConnected}
            />
          )}

          {metrics.latencyMs !== undefined && (
            <MetricRow
              label="Latency"
              value={metrics.latencyMs.toFixed(1)}
              unit="ms"
              highlight={isConnected}
            />
          )}
        </div>

        {/* Settings menu */}
        <div className="mt-6 border-t border-white/5 pt-6">
          <Menu>
            <MenuButton className="inline-flex w-full items-center justify-center gap-2 rounded-md bg-[#8B9A7E]/20 px-3 py-1.5 text-sm/6 font-semibold text-white shadow-inner shadow-[#8B9A7E]/20 focus:outline-none data-focus:outline data-focus:outline-1 data-focus:outline-[#8B9A7E] data-hover:bg-[#8B9A7E]/30 data-open:bg-[#8B9A7E]/30">
              Settings
              <svg
                className="size-4 fill-white/60"
                viewBox="0 0 16 16"
                fill="none"
              >
                <path d="M4.22 6.22a.75.75 0 0 1 1.06 0L8 8.94l2.72-2.72a.75.75 0 1 1 1.06 1.06l-3.25 3.25a.75.75 0 0 1-1.06 0L4.22 7.28a.75.75 0 0 1 0-1.06Z" />
              </svg>
            </MenuButton>

            <MenuItems
              transition
              anchor="bottom"
              className="w-52 origin-top rounded-xl border border-[#8B9A7E]/20 bg-[#8B9A7E]/10 p-1 text-sm/6 text-white shadow-xl backdrop-blur-lg transition duration-100 ease-out [--anchor-gap:4px] focus:outline-none data-closed:scale-95 data-closed:opacity-0"
            >
              <MenuItem>
                <button className="group flex w-full items-center gap-2 rounded-lg px-3 py-1.5 data-focus:bg-[#8B9A7E]/20">
                  <svg
                    className="size-4 fill-white/30"
                    viewBox="0 0 16 16"
                    fill="none"
                  >
                    <path d="M8.75 2.75a.75.75 0 0 0-1.5 0v5.69L5.03 6.22a.75.75 0 0 0-1.06 1.06l3.5 3.5a.75.75 0 0 0 1.06 0l3.5-3.5a.75.75 0 0 0-1.06-1.06L8.75 8.44V2.75Z" />
                    <path d="M3.5 9.75a.75.75 0 0 0-1.5 0v1.5A2.75 2.75 0 0 0 4.75 14h6.5A2.75 2.75 0 0 0 14 11.25v-1.5a.75.75 0 0 0-1.5 0v1.5c0 .69-.56 1.25-1.25 1.25h-6.5c-.69 0-1.25-.56-1.25-1.25v-1.5Z" />
                  </svg>
                  Quality Settings
                </button>
              </MenuItem>

              <MenuItem>
                <button className="group flex w-full items-center gap-2 rounded-lg px-3 py-1.5 data-focus:bg-[#8B9A7E]/20">
                  <svg
                    className="size-4 fill-white/30"
                    viewBox="0 0 16 16"
                    fill="none"
                  >
                    <path
                      fillRule="evenodd"
                      d="M15 8A7 7 0 1 1 1 8a7 7 0 0 1 14 0Zm-6 3.5a1 1 0 1 1-2 0 1 1 0 0 1 2 0ZM7.293 5.293a1 1 0 1 1 1.414 1.414L8.414 7H9a1 1 0 1 1 0 2H8a1 1 0 0 1-1-1V7a1 1 0 0 1 .293-.707Z"
                    />
                  </svg>
                  About
                </button>
              </MenuItem>
            </MenuItems>
          </Menu>
        </div>
      </div>
    </div>
  );
}

interface MetricRowProps {
  label: string;
  value: string | number;
  unit: string;
  highlight?: boolean;
}

function MetricRow({ label, value, unit, highlight = false }: MetricRowProps) {
  return (
    <div className="flex items-baseline justify-between text-sm">
      <span className="text-white/50">{label}</span>
      <span
        className={`font-semibold tabular-nums ${highlight ? "text-[#9CA986]" : "text-white/70"}`}
      >
        {value}
        {unit && <span className="ml-0.5 text-white/40">{unit}</span>}
      </span>
    </div>
  );
}
