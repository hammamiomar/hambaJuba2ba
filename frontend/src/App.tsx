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
  const [promptC, setPromptC] = useState("crispy golden fries");
  const [promptD, setPromptD] = useState("fresh green salad");

  const [promptPos, setPromptPos] = useState<[number, number, number]>([
    0.5, 0.5, 0.5,
  ]);
  const [latentPos, setLatentPos] = useState<[number, number, number]>([
    0.5, 0.5, 0.5,
  ]);
  const [promptEdgeProximity, setPromptEdgeProximity] = useState(0);
  const [latentEdgeProximity, setLatentEdgeProximity] = useState(0);

  const handleFrame = useCallback(async (data: ArrayBuffer) => {
    await canvasRef.current?.renderFrame(data);
  }, []);

  const handlePositionUpdate = useCallback(
    (
      promptPosition: [number, number, number],
      latentPosition: [number, number, number],
      promptProx: number,
      latentProx: number,
    ) => {
      setPromptPos(promptPosition);
      setLatentPos(latentPosition);
      setPromptEdgeProximity(promptProx);
      setLatentEdgeProximity(latentProx);
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
    onPositionUpdate: handlePositionUpdate,
    autoConnect: false,
    enableReconnect: true,
  });

  const handleStart = useCallback(() => {
    sendStart(sourcePrompt, targetPrompt, promptC, promptD);
  }, [sendStart, sourcePrompt, targetPrompt, promptC, promptD]);

  const handlePromptVector = useCallback(
    (dx: number, dy: number, dz: number, magnitude: number) => {
      sendDirectionUpdate?.(dx, dy, dz, magnitude, 0, 0, 0, 0);
    },
    [sendDirectionUpdate],
  );

  const handleLatentVector = useCallback(
    (dx: number, dy: number, dz: number, magnitude: number) => {
      sendDirectionUpdate?.(0, 0, 0, 0, dx, dy, dz, magnitude);
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
          promptC={promptC}
          onPromptCChange={setPromptC}
          promptD={promptD}
          onPromptDChange={setPromptD}
          reconnectAttempts={reconnectAttempts}
        />

        {/* Vector field controls */}
        <div className="absolute bottom-8 left-1/2 flex -translate-x-1/2 gap-8">
          <VectorField
            label="Prompt"
            position={promptPos}
            edgeProximity={promptEdgeProximity}
            onVectorChange={handlePromptVector}
          />
          <VectorField
            label="Latent"
            position={latentPos}
            edgeProximity={latentEdgeProximity}
            onVectorChange={handleLatentVector}
          />
        </div>
      </div>
    </ErrorBoundary>
  );
}

export default App;
