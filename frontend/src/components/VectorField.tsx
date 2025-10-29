import { useRef, useEffect, useState } from "react";

interface VectorFieldProps {
  label: string;
  edgeProximity: number;
  position: [number, number, number]; // x, y, z in [0, 1]
  onVectorChange: (dx: number, dy: number, dz: number, magnitude: number) => void;
}

export function VectorField({
  label,
  edgeProximity,
  position,
  onVectorChange,
}: VectorFieldProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isActive, setIsActive] = useState(false);
  const [vector, setVector] = useState({ dx: 0, dy: 0, dz: 0, magnitude: 0 });
  const [zMode, setZMode] = useState<"none" | "up" | "down">("none");

  const SIZE = 200;
  const CENTER = SIZE / 2;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, SIZE, SIZE);

    // Current position (deep purple)
    const posX = position[0] * SIZE;
    const posY = position[1] * SIZE;

    // Calculate preview position if active
    let previewX = posX;
    let previewY = posY;
    if (isActive && vector.magnitude > 0.01) {
      const stepSize = 0.05;
      previewX = Math.max(
        0,
        Math.min(SIZE, posX + vector.dx * stepSize * SIZE),
      );
      previewY = Math.max(
        0,
        Math.min(SIZE, posY + vector.dy * stepSize * SIZE),
      );

      // Draw dotted line if positions are far apart
      const distance = Math.sqrt(
        (previewX - posX) ** 2 + (previewY - posY) ** 2,
      );
      if (distance > 20) {
        ctx.beginPath();
        ctx.setLineDash([3, 3]);
        ctx.moveTo(posX, posY);
        ctx.lineTo(previewX, previewY);
        ctx.strokeStyle = "rgba(156, 163, 175, 0.6)";
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Draw preview position (grey translucent)
      ctx.beginPath();
      ctx.arc(previewX, previewY, 5, 0, 2 * Math.PI);
      ctx.fillStyle = "rgba(156, 163, 175, 0.5)";
      ctx.fill();
    }

    // Draw current position (deep purple)
    ctx.beginPath();
    ctx.arc(posX, posY, 6, 0, 2 * Math.PI);
    ctx.fillStyle = "rgba(88, 28, 135, 1)";
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw arrow if active (smaller)
    if (isActive && vector.magnitude > 0.01) {
      const arrowLength = vector.magnitude * CENTER * 0.5;
      const angle =
        zMode === "none"
          ? Math.atan2(vector.dy, vector.dx)
          : vector.dy > 0
            ? Math.PI / 2
            : -Math.PI / 2;
      const endX = CENTER + Math.cos(angle) * arrowLength;
      const endY = CENTER + Math.sin(angle) * arrowLength;

      ctx.beginPath();
      ctx.moveTo(CENTER, CENTER);
      ctx.lineTo(endX, endY);
      ctx.strokeStyle = `rgba(147, 51, 234, ${0.5 + vector.magnitude * 0.5})`;
      ctx.lineWidth = 1.5 + vector.magnitude * 2;
      ctx.stroke();

      // Arrow head (smaller)
      const headSize = 6 + vector.magnitude * 4;
      ctx.beginPath();
      ctx.moveTo(endX, endY);
      ctx.lineTo(
        endX - headSize * Math.cos(angle - Math.PI / 6),
        endY - headSize * Math.sin(angle - Math.PI / 6),
      );
      ctx.lineTo(
        endX - headSize * Math.cos(angle + Math.PI / 6),
        endY - headSize * Math.sin(angle + Math.PI / 6),
      );
      ctx.closePath();
      ctx.fillStyle = `rgba(147, 51, 234, ${0.5 + vector.magnitude * 0.5})`;
      ctx.fill();
    }
  }, [isActive, vector, edgeProximity, position, zMode]);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsActive(true);
    handleMouseMove(e);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isActive && e.buttons === 0) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    let dx = 0,
      dy = 0,
      dz = 0;

    if (zMode === "none") {
      // XY mode
      dx = (x - CENTER) / CENTER;
      dy = (y - CENTER) / CENTER;
    } else {
      // Z mode (space bar)
      dz = (CENTER - y) / CENTER;
      dy = (y - CENTER) / CENTER; // For arrow visual
    }

    const magnitude = Math.min(
      Math.sqrt(dx * dx + dy * dy + dz * dz),
      1.0,
    );

    setVector({ dx, dy, dz, magnitude });
    onVectorChange(dx, dy, dz, magnitude);
  };

  const handleMouseUp = () => {
    setIsActive(false);
    setVector({ dx: 0, dy: 0, dz: 0, magnitude: 0 });
    onVectorChange(0, 0, 0, 0);
  };

  // Keyboard handling for space bar
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === " " && zMode === "none") {
        e.preventDefault();
        setZMode("up");
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === " ") {
        e.preventDefault();
        setZMode("none");
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [zMode]);

  // Border color based on edge proximity
  const getBorderClass = () => {
    if (edgeProximity > 0.95) {
      return "border-red-500";
    } else if (edgeProximity > 0.8) {
      return "border-red-500 animate-pulse";
    }
    return "border-white/10";
  };

  const getModeText = () => {
    if (zMode === "up") return "Z Mode (Space)";
    return "XY Mode";
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
        className={`cursor-crosshair rounded-lg border-2 bg-gray-900/20 backdrop-blur-lg ${getBorderClass()}`}
      />
      <div className="flex flex-col items-center gap-1 text-xs">
        <span className="font-mono text-white/70">
          ({position[0].toFixed(2)}, {position[1].toFixed(2)},{" "}
          {position[2].toFixed(2)})
        </span>
        <span className="text-white/50">{getModeText()}</span>
      </div>
    </div>
  );
}
