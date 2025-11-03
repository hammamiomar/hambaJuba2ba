import { useRef, useState, useEffect } from "react";

interface AudioPlayerProps {
  onAudioLoaded: (audioId: string, rmsValues: number[], timestamps: number[], duration: number) => void;
  disabled?: boolean;
  audioTime?: number; // backend-driven time for progress visualization
  isGenerating?: boolean;

  // audio control callbacks
  onTimeUpdate?: (time: number) => void;
  onPlay?: () => void;
  onPause?: () => void;
  onSeeked?: (time: number) => void;
}

export function AudioPlayer({
  onAudioLoaded,
  disabled = false,
  audioTime = 0,
  isGenerating = false,
  onTimeUpdate,
  onPlay,
  onPause,
  onSeeked,
}: AudioPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [rmsValues, setRmsValues] = useState<number[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);

  const audioRef = useRef<HTMLAudioElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // use backend audio time during generation, otherwise local playback time
  const displayTime = isGenerating ? audioTime : currentTime;

  // draw waveform
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || rmsValues.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const WIDTH = canvas.width;
    const HEIGHT = canvas.height;

    ctx.clearRect(0, 0, WIDTH, HEIGHT);

    // background
    ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
    ctx.fillRect(0, 0, WIDTH, HEIGHT);

    // waveform
    ctx.strokeStyle = "#B5CC9A";
    ctx.lineWidth = 2;
    ctx.beginPath();

    const step = WIDTH / rmsValues.length;
    rmsValues.forEach((rms, i) => {
      const x = i * step;
      const y = HEIGHT - rms * HEIGHT;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // playback position
    if (duration > 0) {
      const progress = displayTime / duration;
      const x = progress * WIDTH;

      ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, HEIGHT);
      ctx.stroke();
    }
  }, [rmsValues, displayTime, duration]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setFileName(file.name);

    try {
      // upload to backend
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://localhost:8080/api/audio/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.error) {
        console.error("Upload failed:", data.error);
        setIsUploading(false);
        return;
      }

      // store audio analysis
      setRmsValues(data.rms);
      setDuration(data.duration);
      onAudioLoaded(data.audio_id, data.rms, data.timestamps, data.duration);

      // create audio element for playback
      const audioUrl = URL.createObjectURL(file);
      if (audioRef.current) {
        audioRef.current.src = audioUrl;
      }

      setIsUploading(false);
    } catch (error) {
      console.error("Upload error:", error);
      setIsUploading(false);
    }
  };

  const togglePlayPause = () => {
    if (!audioRef.current) return;

    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  // audio event handlers
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    let lastUpdateTime = 0;
    const UPDATE_THROTTLE = 33; // ~30fps

    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime);

      // throttle time updates to backend
      const now = Date.now();
      if (onTimeUpdate && now - lastUpdateTime > UPDATE_THROTTLE) {
        onTimeUpdate(audio.currentTime);
        lastUpdateTime = now;
      }
    };

    const handlePlay = () => {
      setIsPlaying(true);
      if (onPlay) onPlay();
    };

    const handlePause = () => {
      setIsPlaying(false);
      if (onPause) onPause();
    };

    const handleSeeked = () => {
      if (onSeeked) onSeeked(audio.currentTime);
    };

    const handleEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
      if (onPause) onPause();
    };

    audio.addEventListener("timeupdate", handleTimeUpdate);
    audio.addEventListener("play", handlePlay);
    audio.addEventListener("pause", handlePause);
    audio.addEventListener("seeked", handleSeeked);
    audio.addEventListener("ended", handleEnded);

    return () => {
      audio.removeEventListener("timeupdate", handleTimeUpdate);
      audio.removeEventListener("play", handlePlay);
      audio.removeEventListener("pause", handlePause);
      audio.removeEventListener("seeked", handleSeeked);
      audio.removeEventListener("ended", handleEnded);
    };
  }, [onTimeUpdate, onPlay, onPause, onSeeked]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="text-xs font-medium uppercase tracking-wide text-white/50">
        Audio Track
      </div>

      {/* File upload */}
      <div>
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*"
          onChange={handleFileUpload}
          disabled={disabled || isUploading}
          className="hidden"
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled || isUploading}
          className="w-full rounded-md bg-[#8B9A7E]/20 px-3 py-2 text-sm text-white hover:bg-[#8B9A7E]/30 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {isUploading
            ? "Uploading..."
            : fileName
              ? `Loaded: ${fileName}`
              : "Upload Audio"}
        </button>
      </div>

      {/* Waveform */}
      {rmsValues.length > 0 && (
        <>
          <canvas
            ref={canvasRef}
            width={280}
            height={80}
            className="rounded-md border border-white/10"
          />

          {/* Controls */}
          <div className="flex items-center gap-3">
            <button
              onClick={togglePlayPause}
              disabled={disabled}
              className="rounded-md bg-[#8B9A7E]/20 p-2 hover:bg-[#8B9A7E]/30 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {isPlaying ? (
                <svg
                  className="size-5"
                  viewBox="0 0 16 16"
                  fill="currentColor"
                >
                  <rect x="4" y="3" width="3" height="10" rx="1" />
                  <rect x="9" y="3" width="3" height="10" rx="1" />
                </svg>
              ) : (
                <svg
                  className="size-5"
                  viewBox="0 0 16 16"
                  fill="currentColor"
                >
                  <path d="M4 3.732a1.5 1.5 0 0 1 2.305-1.265l6.706 4.267a1.5 1.5 0 0 1 0 2.531l-6.706 4.268A1.5 1.5 0 0 1 4 12.267V3.732Z" />
                </svg>
              )}
            </button>

            <div className="flex-1 text-sm tabular-nums text-white/70">
              {formatTime(currentTime)} / {formatTime(duration)}
            </div>
          </div>
        </>
      )}

      {/* Hidden audio element */}
      <audio ref={audioRef} />
    </div>
  );
}
