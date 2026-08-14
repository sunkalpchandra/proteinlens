"use client";

/** Canvas scatter of the protein representation map.
 *
 *  Renders ~12k points at 60fps: pixel coordinates are precomputed once per
 *  resize, points are drawn in per-color batches (one beginPath per color),
 *  and a d3 quadtree in screen space handles hover/click picking. Zoom/pan is
 *  d3-zoom on the canvas element; the parent supplies colors via `colorOf` so
 *  legend logic lives beside the mode selector, not in here.
 */

import { quadtree, type Quadtree } from "d3-quadtree";
import { select } from "d3-selection";
import { zoom, zoomIdentity, type ZoomTransform } from "d3-zoom";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MapPoint } from "@/lib/types";

export interface EmbeddingMapProps {
  points: MapPoint[];
  colorOf: (p: MapPoint) => string;
  selectedId?: string | null;
  /** accession → cosine similarity; highlighted and exempt from dimming */
  neighborIds?: ReadonlyMap<string, number> | null;
  onSelect?: (p: MapPoint) => void;
  /** When set, the view animates to center this accession. */
  focusId?: string | null;
  className?: string;
}

interface Hover {
  point: MapPoint;
  cx: number;
  cy: number;
}

const PAD = 24;

export function EmbeddingMap({
  points,
  colorOf,
  selectedId = null,
  neighborIds = null,
  onSelect,
  focusId = null,
  className = "",
}: EmbeddingMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const transformRef = useRef<ZoomTransform>(zoomIdentity);
  const rafRef = useRef<number>(0);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [hover, setHover] = useState<Hover | null>(null);

  // --- data-space → base pixel-space mapping (before zoom) ----------------
  const layout = useMemo(() => {
    if (!points.length || !size.w || !size.h) return null;
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const p of points) {
      if (p.x < xMin) xMin = p.x;
      if (p.x > xMax) xMax = p.x;
      if (p.y < yMin) yMin = p.y;
      if (p.y > yMax) yMax = p.y;
    }
    const spanX = xMax - xMin || 1;
    const spanY = yMax - yMin || 1;
    const scale = Math.min((size.w - 2 * PAD) / spanX, (size.h - 2 * PAD) / spanY);
    const offX = (size.w - spanX * scale) / 2;
    const offY = (size.h - spanY * scale) / 2;
    const px = new Float32Array(points.length);
    const py = new Float32Array(points.length);
    for (let i = 0; i < points.length; i++) {
      px[i] = offX + (points[i].x - xMin) * scale;
      // Flip y: UMAP coordinates are math-oriented, canvas y grows downward.
      py[i] = size.h - (offY + (points[i].y - yMin) * scale);
    }
    return { px, py };
  }, [points, size]);

  const tree: Quadtree<number> | null = useMemo(() => {
    if (!layout) return null;
    return quadtree<number>(
      points.map((_, i) => i),
      (i) => layout.px[i],
      (i) => layout.py[i],
    );
  }, [layout, points]);

  const colorGroups = useMemo(() => {
    const groups = new Map<string, number[]>();
    points.forEach((p, i) => {
      const c = colorOf(p);
      const bucket = groups.get(c);
      if (bucket) bucket.push(i);
      else groups.set(c, [i]);
    });
    return groups;
  }, [points, colorOf]);

  const indexOfId = useMemo(() => {
    const m = new Map<string, number>();
    points.forEach((p, i) => m.set(p.id, i));
    return m;
  }, [points]);

  // --- drawing --------------------------------------------------------------
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !layout) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const t = transformRef.current;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.w, size.h);

    const r = Math.max(1.6, Math.min(4.5, 1.9 * Math.sqrt(t.k)));
    const dimming = selectedId !== null;

    for (const [color, indices] of colorGroups) {
      // Pass 1: dimmed points (or all points when nothing is selected).
      ctx.beginPath();
      for (const i of indices) {
        const p = points[i];
        if (dimming && (p.id === selectedId || neighborIds?.has(p.id))) continue;
        const cx = t.applyX(layout.px[i]);
        const cy = t.applyY(layout.py[i]);
        if (cx < -8 || cy < -8 || cx > size.w + 8 || cy > size.h + 8) continue;
        ctx.moveTo(cx + r, cy);
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
      }
      ctx.globalAlpha = dimming ? 0.18 : 0.85;
      ctx.fillStyle = color;
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    // Pass 2: neighbors, brighter with a surface ring.
    if (neighborIds && selectedId) {
      for (const [id] of neighborIds) {
        const i = indexOfId.get(id);
        if (i === undefined) continue;
        const cx = t.applyX(layout.px[i]);
        const cy = t.applyY(layout.py[i]);
        ctx.beginPath();
        ctx.arc(cx, cy, r + 1.5, 0, Math.PI * 2);
        ctx.fillStyle = "#0d0d0d";
        ctx.fill();
        ctx.beginPath();
        ctx.arc(cx, cy, r + 0.5, 0, Math.PI * 2);
        ctx.fillStyle = colorOf(points[i]);
        ctx.fill();
      }
    }

    // Pass 3: selected point on top with a white ring.
    if (selectedId) {
      const i = indexOfId.get(selectedId);
      if (i !== undefined) {
        const cx = t.applyX(layout.px[i]);
        const cy = t.applyY(layout.py[i]);
        ctx.beginPath();
        ctx.arc(cx, cy, r + 3, 0, Math.PI * 2);
        ctx.strokeStyle = "#f2f2ef";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(cx, cy, r + 1, 0, Math.PI * 2);
        ctx.fillStyle = colorOf(points[i]);
        ctx.fill();
      }
    }

    // Hover ring.
    if (hover) {
      const i = indexOfId.get(hover.point.id);
      if (i !== undefined) {
        const cx = t.applyX(layout.px[i]);
        const cy = t.applyY(layout.py[i]);
        ctx.beginPath();
        ctx.arc(cx, cy, r + 2.5, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(242,242,239,0.7)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  }, [layout, size, colorGroups, points, selectedId, neighborIds, indexOfId, colorOf, hover]);

  const scheduleDraw = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(draw);
  }, [draw]);

  useEffect(scheduleDraw, [scheduleDraw]);

  // --- resize ---------------------------------------------------------------
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setSize({ w: Math.round(width), h: Math.round(height) });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !size.w) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size.w * dpr;
    canvas.height = size.h * dpr;
    canvas.style.width = `${size.w}px`;
    canvas.style.height = `${size.h}px`;
    scheduleDraw();
  }, [size, scheduleDraw]);

  // --- zoom / pan ------------------------------------------------------------
  const zoomBehavior = useMemo(
    () =>
      zoom<HTMLCanvasElement, unknown>()
        .scaleExtent([0.5, 40])
        .on("zoom", (event) => {
          transformRef.current = event.transform;
          setHover(null);
          scheduleDraw();
        }),
    [scheduleDraw],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const selection = select(canvas);
    selection.call(zoomBehavior);
    return () => {
      selection.on(".zoom", null);
    };
  }, [zoomBehavior]);

  // --- focus animation --------------------------------------------------------
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !layout || !focusId) return;
    const i = indexOfId.get(focusId);
    if (i === undefined) return;
    const k = Math.max(transformRef.current.k, 4);
    const target = zoomIdentity
      .translate(size.w / 2, size.h / 2)
      .scale(k)
      .translate(-layout.px[i], -layout.py[i]);
    // Tween via rAF: d3-transition is not a dependency, so animate the zoom
    // transform manually and feed each frame through zoomBehavior.transform.
    const from = transformRef.current;
    const duration = 650;
    const t0 = performance.now();
    const easeCubicInOut = (u: number) =>
      u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
    let raf = requestAnimationFrame(function step(now: number) {
      const u = Math.min(1, (now - t0) / duration);
      const e = easeCubicInOut(u);
      const frame = zoomIdentity
        .translate(
          from.x + (target.x - from.x) * e,
          from.y + (target.y - from.y) * e,
        )
        .scale(from.k + (target.k - from.k) * e);
      zoomBehavior.transform(select(canvas), frame);
      if (u < 1) raf = requestAnimationFrame(step);
    });
    return () => cancelAnimationFrame(raf);
  }, [focusId, layout, indexOfId, size, zoomBehavior]);

  // --- picking ---------------------------------------------------------------
  const pick = useCallback(
    (clientX: number, clientY: number): Hover | null => {
      const canvas = canvasRef.current;
      if (!canvas || !tree || !layout) return null;
      const rect = canvas.getBoundingClientRect();
      const sx = clientX - rect.left;
      const sy = clientY - rect.top;
      const t = transformRef.current;
      const found = tree.find(t.invertX(sx), t.invertY(sy), 12 / t.k);
      if (found === undefined) return null;
      return {
        point: points[found],
        cx: t.applyX(layout.px[found]),
        cy: t.applyY(layout.py[found]),
      };
    },
    [tree, layout, points],
  );

  return (
    <div ref={containerRef} className={`relative overflow-hidden ${className}`}>
      <canvas
        ref={canvasRef}
        className="block cursor-crosshair"
        onMouseMove={(e) => setHover(pick(e.clientX, e.clientY))}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => {
          const hit = pick(e.clientX, e.clientY);
          if (hit && onSelect) onSelect(hit.point);
        }}
      />
      {hover && (
        <div
          className="pointer-events-none absolute z-10 w-56 rounded border border-bds bg-surface2 p-2.5 shadow-xl"
          style={{
            left: Math.min(hover.cx + 14, size.w - 240),
            top: Math.min(hover.cy + 14, size.h - 150),
          }}
        >
          <div className="truncate text-[13px] font-medium text-ink">{hover.point.name}</div>
          <div className="mt-1 space-y-0.5 font-mono text-[11px] text-ink2">
            <div className="flex justify-between">
              <span className="text-ink3">{hover.point.id}</span>
              <span>{hover.point.org}</span>
            </div>
            <div className="flex justify-between">
              <span>{hover.point.len} aa</span>
              <span>cluster {hover.point.cluster}</span>
            </div>
            {hover.point.family && (
              <div className="truncate text-ink3">{hover.point.family}</div>
            )}
            {neighborIds?.has(hover.point.id) && (
              <div className="text-accent tabular">
                cos {neighborIds.get(hover.point.id)?.toFixed(3)}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
