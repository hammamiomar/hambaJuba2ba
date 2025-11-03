import { useRef, useState, useEffect } from "react";

interface ControlDialProps {
  label: string;
  value: number; // 0 to 1
  min?: number;
  max?: number;
  onChange: (value: number) => void;
  disabled?: boolean;
}

export function ControlDial({
  label,
  value,
  min = 0.001,
  max = 0.1,
  onChange,
  disabled = false,
}: ControlDialProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const SIZE = 120;
  const CENTER = SIZE / 2;
  const RADIUS = 45;

  // map value to angle (-135° to +135°)
  const valueToAngle = (v: number) => {
    const normalized = (v - min) / (max - min);
    return -135 + normalized * 270;
  };

  // map angle to value
  const angleToValue = (angle: number) => {
    const normalized = (angle + 135) / 270;
    return min + normalized * (max - min);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // clear
    ctx.clearRect(0, 0, SIZE, SIZE);

    // outer circle
    ctx.beginPath();
    ctx.arc(CENTER, CENTER, RADIUS, 0, 2 * Math.PI);
    ctx.strokeStyle = "rgba(139, 154, 126, 0.3)";
    ctx.lineWidth = 3;
    ctx.stroke();

    // arc for value
    const angle = valueToAngle(value);
    const startAngle = (-135 * Math.PI) / 180;
    const endAngle = (angle * Math.PI) / 180;

    ctx.beginPath();
    ctx.arc(CENTER, CENTER, RADIUS, startAngle, endAngle);
    ctx.strokeStyle = "#B5CC9A";
    ctx.lineWidth = 4;
    ctx.stroke();

    // pointer
    const pointerAngle = (angle * Math.PI) / 180;
    const pointerX = CENTER + Math.cos(pointerAngle) * (RADIUS - 10);
    const pointerY = CENTER + Math.sin(pointerAngle) * (RADIUS - 10);

    ctx.beginPath();
    ctx.arc(pointerX, pointerY, 6, 0, 2 * Math.PI);
    ctx.fillStyle = "#B5CC9A";
    ctx.fill();

    // center circle
    ctx.beginPath();
    ctx.arc(CENTER, CENTER, 8, 0, 2 * Math.PI);
    ctx.fillStyle = "rgba(139, 154, 126, 0.5)";
    ctx.fill();

    // value text
    ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
    ctx.font = "14px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(value.toFixed(3), CENTER, CENTER);
  }, [value, min, max]);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (disabled) return;
    setIsDragging(true);
    handleMouseMove(e);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging && e.buttons === 0) return;
    if (disabled) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left - CENTER;
    const y = e.clientY - rect.top - CENTER;

    const angle = (Math.atan2(y, x) * 180) / Math.PI;

    // clamp to -135 to +135
    let clampedAngle = angle;
    if (angle > 135) clampedAngle = -135;
    if (angle < -135) clampedAngle = -135;

    const newValue = Math.max(min, Math.min(max, angleToValue(clampedAngle)));
    onChange(newValue);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  return (
    <div className="flex flex-col items-center gap-2">
      <span className="text-xs font-medium uppercase tracking-wide text-white/50">
        {label}
      </span>
      <canvas
        ref={canvasRef}
        width={SIZE}
        height={SIZE}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className={`rounded-lg border border-white/10 bg-gray-900/20 backdrop-blur-lg ${disabled ? "cursor-not-allowed opacity-40" : "cursor-pointer"}`}
      />
      <div className="flex gap-2 text-xs text-white/40">
        <span>{min.toFixed(3)}</span>
        <span>→</span>
        <span>{max.toFixed(3)}</span>
      </div>
    </div>
  );
}
