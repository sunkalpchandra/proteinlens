"use client";

/** Compact SVG track of one value per residue (attention weights, Δ norms).
 *
 *  Line + area in a single accent hue; a marked position gets a reference
 *  line; hover shows a crosshair with the exact value. Height stays small —
 *  this sits under a sequence, it is not a standalone chart.
 */

import { useMemo, useRef, useState } from "react";

export interface ResidueTrackProps {
  values: number[];
  /** 1-based highlighted position (e.g. the mutated site). */
  marked?: number | null;
  label: string;
  color?: string;
  height?: number;
  onClickPos?: (position: number) => void;
  className?: string;
}

export function ResidueTrack({
  values,
  marked = null,
  label,
  color = "#3987e5",
  height = 64,
  onClickPos,
  className = "",
}: ResidueTrackProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoverPos, setHoverPos] = useState<number | null>(null); // 1-based
  const width = 720; // viewBox width; scales to container

  const { path, area, max } = useMemo(() => {
    const n = values.length;
    if (!n) return { path: "", area: "", max: 1 };
    const maxValue = Math.max(...values, 1e-12);
    const stepX = width / Math.max(n - 1, 1);
    const y = (v: number) => height - 4 - (v / maxValue) * (height - 12);
    let d = `M 0 ${y(values[0]).toFixed(1)}`;
    for (let i = 1; i < n; i++) d += ` L ${(i * stepX).toFixed(1)} ${y(values[i]).toFixed(1)}`;
    const a = `${d} L ${width} ${height} L 0 ${height} Z`;
    return { path: d, area: a, max: maxValue };
  }, [values, height]);

  const posToX = (pos: number) =>
    ((pos - 1) * width) / Math.max(values.length - 1, 1);

  const eventPos = (clientX: number): number | null => {
    const svg = svgRef.current;
    if (!svg || !values.length) return null;
    const rect = svg.getBoundingClientRect();
    const frac = (clientX - rect.left) / rect.width;
    const pos = Math.round(frac * (values.length - 1)) + 1;
    return Math.max(1, Math.min(values.length, pos));
  };

  return (
    <div className={className}>
      <div className="flex items-baseline justify-between pb-1">
        <span className="label-mono">{label}</span>
        {hoverPos !== null && (
          <span className="font-mono text-[11px] text-ink2 tabular">
            pos {hoverPos} · {values[hoverPos - 1]?.toPrecision(3)}
          </span>
        )}
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className={`block w-full ${onClickPos ? "cursor-pointer" : ""}`}
        style={{ height }}
        onMouseMove={(e) => setHoverPos(eventPos(e.clientX))}
        onMouseLeave={() => setHoverPos(null)}
        onClick={(e) => {
          const pos = eventPos(e.clientX);
          if (pos !== null) onClickPos?.(pos);
        }}
      >
        <line x1={0} y1={height - 4} x2={width} y2={height - 4} stroke="#2c2c2a" strokeWidth={1} />
        <path d={area} fill={color} opacity={0.12} />
        <path d={path} fill="none" stroke={color} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
        {marked !== null && marked >= 1 && marked <= values.length && (
          <line
            x1={posToX(marked)} y1={2} x2={posToX(marked)} y2={height - 4}
            stroke="#e66767" strokeWidth={1.25} strokeDasharray="3 2"
            vectorEffect="non-scaling-stroke"
          />
        )}
        {hoverPos !== null && (
          <line
            x1={posToX(hoverPos)} y1={2} x2={posToX(hoverPos)} y2={height - 4}
            stroke="rgba(242,242,239,0.45)" strokeWidth={1} vectorEffect="non-scaling-stroke"
          />
        )}
      </svg>
      <div className="flex justify-between pt-0.5 font-mono text-[9px] text-ink3 tabular">
        <span>1</span>
        <span>max {max.toPrecision(3)}</span>
        <span>{values.length}</span>
      </div>
    </div>
  );
}
