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

  const [sourcePrompt, setSourcePrompt] = useState(
    "Moldy Burger in a sopping wet sewer, grimy, high quality",
  );
  const [targetPrompt, setTargetPrompt] = useState("steamy burger");

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
    sendStart(sourcePrompt, targetPrompt);
  }, [sendStart, sourcePrompt, targetPrompt]);

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
          sourcePrompt={sourcePrompt}
          onSourcePromptChange={setSourcePrompt}
          targetPrompt={targetPrompt}
          onTargetPromptChange={setTargetPrompt}
          reconnectAttempts={reconnectAttempts}
        />
      </div>
    </ErrorBoundary>
  );
}

export default App;
