import { useRef, useCallback, useState } from "react";
import { Canvas } from "./components/Canvas";
import type { CanvasHandle } from "./components/Canvas";
import { Controls } from "./components/Controls";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { VectorField } from "./components/VectorField";
import { useWebSocket } from "./hooks/useWebSocket";
import { WS_CONFIG } from "./constants";
import type { Metrics } from "./types";

function App() {
  const canvasRef = useRef<CanvasHandle>(null);

  const [sourcePrompt, setSourcePrompt] = useState(
    "Moldy Burger in a sopping wet sewer, grimy, high quality",
  );
  const [targetPrompt, setTargetPrompt] = useState("steamy burger");

  const [promptEdgeProximity, setPromptEdgeProximity] = useState(0);
  const [latentEdgeProximity, setLatentEdgeProximity] = useState(0);

  const handleFrame = useCallback(async (data: ArrayBuffer) => {
    await canvasRef.current?.renderFrame(data);
  }, []);

  const handleEdgeProximity = useCallback(
    (prompt: number, latent: number) => {
      setPromptEdgeProximity(prompt);
      setLatentEdgeProximity(latent);
    },
    [],
  );

  const {
    connect,
    disconnect,
    sendStart,
    sendStop,
    sendDirectionUpdate,
    status,
    fps,
    isGenerating,
    reconnectAttempts,
  } = useWebSocket({
    url: WS_CONFIG.URL,
    onFrame: handleFrame,
    onEdgeProximity: handleEdgeProximity,
    autoConnect: false,
    enableReconnect: true,
  });

  const handleStart = useCallback(() => {
    sendStart(sourcePrompt, targetPrompt);
  }, [sendStart, sourcePrompt, targetPrompt]);

  const handlePromptVector = useCallback(
    (dx: number, dy: number, magnitude: number) => {
      sendDirectionUpdate?.(dx, dy, magnitude, 0, 0, 0);
    },
    [sendDirectionUpdate],
  );

  const handleLatentVector = useCallback(
    (dx: number, dy: number, magnitude: number) => {
      sendDirectionUpdate?.(0, 0, 0, dx, dy, magnitude);
    },
    [sendDirectionUpdate],
  );

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

        {/* Vector field controls */}
        <div className="absolute bottom-8 left-1/2 flex -translate-x-1/2 gap-8">
          <VectorField
            label="Prompt"
            edgeProximity={promptEdgeProximity}
            onVectorChange={handlePromptVector}
          />
          <VectorField
            label="Latent"
            edgeProximity={latentEdgeProximity}
            onVectorChange={handleLatentVector}
          />
        </div>
      </div>
    </ErrorBoundary>
  );
}

export default App;
