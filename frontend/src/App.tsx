import { useRef, useCallback, useState } from "react";
import { Canvas } from "./components/Canvas";
import type { CanvasHandle } from "./components/Canvas";
import { Controls } from "./components/Controls";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { useWebSocket } from "./hooks/useWebSocket";
import { WS_CONFIG } from "./constants";
import type { Metrics } from "./types";

function App() {
  const canvasRef = useRef<CanvasHandle>(null);

  const [prompt, setPrompt] = useState(
    "Moldy Burger in a sopping wet sewer, grimy, high quality",
  );

  const handleFrame = useCallback(async (data: ArrayBuffer) => {
    await canvasRef.current?.renderFrame(data);
  }, []);

  const {
    connect,
    disconnect,
    sendStart,
    sendStop,
    status,
    fps,
    isGenerating,
    reconnectAttempts,
  } = useWebSocket({
    url: WS_CONFIG.URL,
    onFrame: handleFrame,
    autoConnect: false,
    enableReconnect: true,
  });

  const handleStart = useCallback(() => {
    sendStart(prompt);
  }, [sendStart, prompt]);

  const metrics: Metrics = {
    fps,
  };

  return (
    <ErrorBoundary>
      {/* Full-screen container */}
      <div className="relative w-screen h-screen bg-black overflow-hidden">
        {/* Canvas: fills entire screen */}
        <Canvas ref={canvasRef} className="w-full h-full" />

        <Controls
          status={status}
          metrics={metrics}
          onConnect={connect}
          onDisconnect={disconnect}
          onStart={handleStart}
          onStop={sendStop}
          isGenerating={isGenerating}
          prompt={prompt}
          onPromptChange={setPrompt}
          reconnectAttempts={reconnectAttempts}
        />
      </div>
    </ErrorBoundary>
  );
}

export default App;
