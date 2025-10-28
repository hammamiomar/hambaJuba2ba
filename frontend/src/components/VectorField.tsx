import { useRef, useEffect, useState } from "react";

interface VectorFieldProps {
  label: string;
  edgeProximity: number; // 0-1, where 1 is at edge
  onVectorChange: (dx: number, dy: number, magnitude: number) => void;
}

export function VectorField({
  label,
  edgeProximity,
  onVectorChange,
}: VectorFieldProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isActive, setIsActive] = useState(false);
  const [vector, setVector] = useState({ dx: 0, dy: 0, magnitude: 0 });

  const SIZE = 200;
  const CENTER = SIZE / 2;
  const GRID_SIZE = 4;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear
    ctx.clearRect(0, 0, SIZE, SIZE);

    // Draw 16 dots
    const spacing = SIZE / (GRID_SIZE + 1);
    for (let i = 1; i <= GRID_SIZE; i++) {
      for (let j = 1; j <= GRID_SIZE; j++) {
        const x = i * spacing;
        const y = j * spacing;

        ctx.beginPath();
        ctx.arc(x, y, 2, 0, 2 * Math.PI);

        // Color based on edge proximity
        if (edgeProximity > 0.95) {
          ctx.fillStyle = "rgba(239, 68, 68, 1)"; // Solid red
        } else if (edgeProximity > 0.8) {
          const flash = Math.sin(Date.now() / 200) * 0.5 + 0.5;
          ctx.fillStyle = `rgba(239, 68, 68, ${flash})`;
        } else {
          ctx.fillStyle = "rgba(255, 255, 255, 0.3)";
        }

        ctx.fill();
      }
    }

    // Draw arrow if active
    if (isActive && vector.magnitude > 0.01) {
      const arrowLength = vector.magnitude * CENTER * 0.8;
      const angle = Math.atan2(vector.dy, vector.dx);
      const endX = CENTER + Math.cos(angle) * arrowLength;
      const endY = CENTER + Math.sin(angle) * arrowLength;

      // Arrow shaft
      ctx.beginPath();
      ctx.moveTo(CENTER, CENTER);
      ctx.lineTo(endX, endY);
      ctx.strokeStyle = `rgba(147, 51, 234, ${0.5 + vector.magnitude * 0.5})`;
      ctx.lineWidth = 2 + vector.magnitude * 3;
      ctx.stroke();

      // Arrow head
      const headSize = 8 + vector.magnitude * 8;
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
  }, [isActive, vector, edgeProximity]);

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

    const dx = (x - CENTER) / CENTER;
    const dy = (y - CENTER) / CENTER;
    const magnitude = Math.min(
      Math.sqrt(dx * dx + dy * dy),
      1.0,
    );

    setVector({ dx, dy, magnitude });
    onVectorChange(dx, dy, magnitude);
  };

  const handleMouseUp = () => {
    setIsActive(false);
    setVector({ dx: 0, dy: 0, magnitude: 0 });
    onVectorChange(0, 0, 0);
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
        className="cursor-crosshair rounded-lg border border-white/10 bg-gray-900/20 backdrop-blur-lg"
      />
    </div>
  );
}
