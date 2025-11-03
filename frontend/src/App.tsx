import { useRef, useCallback, useState, useEffect } from "react";
import { Canvas } from "./components/Canvas";
import type { CanvasHandle } from "./components/Canvas";
import { Controls } from "./components/Controls";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { VectorField } from "./components/VectorField";
import { ControlDial } from "./components/ControlDial";
import { useWebSocket } from "./hooks/useWebSocket";
import { WS_CONFIG } from "./constants";
import type { Metrics } from "./types";
import type { Mode } from "./components/ModeSelector";

function App() {
  const canvasRef = useRef<CanvasHandle>(null);

  // mode state
  const [mode, setMode] = useState<Mode>("looping");

  // prompt state
  const [sourcePrompt, setSourcePrompt] = useState(
    "Moldy Burger in a sopping wet sewer, grimy, high quality",
  );
  const [targetPrompt, setTargetPrompt] = useState("steamy burger");

  // step sizes (for looping/audio modes)
  const [promptStep, setPromptStep] = useState(0.02);
  const [latentStep, setLatentStep] = useState(0.02);

  // audio state (for audio mode)
  const [audioId, setAudioId] = useState<string | null>(null);
  const [audioRms, setAudioRms] = useState<number[]>([]);
  const [audioTimestamps, setAudioTimestamps] = useState<number[]>([]);
  const [audioDuration, setAudioDuration] = useState(0);
  const [audioTime, setAudioTime] = useState(0); // backend-driven playback time

  // edge proximity (for four-corner mode)
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

  const handleInterpPos = useCallback(
    (promptT: number, latentT: number, audioTime?: number) => {
      if (audioTime !== undefined) {
        setAudioTime(audioTime);
      }
    },
    [],
  );

  const {
    connect,
    disconnect,
    sendStart,
    sendStop,
    sendDirectionUpdate,
    sendStepUpdate,
    sendAudioTime,
    sendAudioPlay,
    sendAudioPause,
    sendAudioSeek,
    status,
    fps,
    isGenerating,
    reconnectAttempts,
  } = useWebSocket({
    mode,
    onFrame: handleFrame,
    onEdgeProximity: handleEdgeProximity,
    onInterpPos: handleInterpPos,
    autoConnect: false,
    enableReconnect: true,
  });

  const handleStart = useCallback(() => {
    sendStart(sourcePrompt, targetPrompt, promptStep, latentStep, audioId || undefined);
  }, [sendStart, sourcePrompt, targetPrompt, promptStep, latentStep, audioId]);

  const handleAudioLoaded = useCallback(
    (id: string, rms: number[], timestamps: number[], duration: number) => {
      setAudioId(id);
      setAudioRms(rms);
      setAudioTimestamps(timestamps);
      setAudioDuration(duration);
    },
    [],
  );

  // send step updates when dials change (with debounce)
  useEffect(() => {
    if (!isGenerating) return;

    const timer = setTimeout(() => {
      sendStepUpdate(promptStep, latentStep);
    }, 100);

    return () => clearTimeout(timer);
  }, [promptStep, latentStep, isGenerating, sendStepUpdate]);

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
          mode={mode}
          onModeChange={setMode}
          onAudioLoaded={handleAudioLoaded}
          audioTime={audioTime}
          onAudioTimeUpdate={sendAudioTime}
          onAudioPlay={sendAudioPlay}
          onAudioPause={sendAudioPause}
          onAudioSeeked={sendAudioSeek}
        />

        {/* Bottom center controls */}
        <div className="absolute bottom-8 left-1/2 flex -translate-x-1/2 gap-8">
          {/* Dials (looping and audio modes) */}
          {(mode === "looping" || mode === "audio") && (
            <>
              <ControlDial
                label="Prompt Step"
                value={promptStep}
                onChange={setPromptStep}
              />
              <ControlDial
                label="Latent Step"
                value={latentStep}
                onChange={setLatentStep}
              />
            </>
          )}

          {/* Vector fields (four-corner mode) */}
          {mode === "fourcorner" && (
            <>
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
            </>
          )}
        </div>
      </div>
    </ErrorBoundary>
  );
}

export default App;
