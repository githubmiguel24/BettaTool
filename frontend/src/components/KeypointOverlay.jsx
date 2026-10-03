/**
 * SVG layers for the perceptual tier's output, drawn in ORIGINAL image pixels.
 * Rendered inside AnnotatedImage's <svg>, which owns the viewBox (and thus
 * zoom), so this component does no scaling math of its own. `unit` is the
 * current viewBox size / 400, keeping markers a constant on-screen size.
 *
 * Each landmark is a dot plus its 1-sigma uncertainty ellipse, so the
 * covariance head's output is visible rather than buried in the JSON. The
 * optional heatmap draws a warm blob per landmark whose radius follows that
 * same sigma: a confident landmark is a tight hot spot, an uncertain one blooms.
 *
 * Dots are colored by the API's `group` field (head / dorsal / caudal / anal
 * fin).
 *
 * `highlightIndices`, when set (even to an empty array), switches into
 * "spotlight" mode for a selected measurement: everything outside the set is
 * hidden entirely, and each remaining landmark gets a soft glow halo.
 */

import { useMemo, useState } from "react";
import { colorForGroup } from "../data/keypointGroups.js";
import { computeMarkerStyles, pickLineColor } from "../lib/adaptiveColor.js";
import { MEASUREMENT_SEGMENTS } from "../data/measurementSegments.js";

// Index-aligned with app/perception/keypoints.py SKELETON_EDGES.
const SKELETON_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4], [3, 5], [5, 6], [5, 7],
  [6, 8], [7, 9], [8, 9], [6, 10], [10, 11], [11, 12], [0, 12],
];

/** Covariance ellipse from sigma_x, sigma_y, rho -> (rx, ry, rotation deg). */
function ellipseParams(sigmaX, sigmaY, rho) {
  const vx = sigmaX * sigmaX;
  const vy = sigmaY * sigmaY;
  const cxy = rho * sigmaX * sigmaY;
  const tr = vx + vy;
  const det = vx * vy - cxy * cxy;
  const disc = Math.max(0, (tr * tr) / 4 - det);
  const l1 = tr / 2 + Math.sqrt(disc);
  const l2 = Math.max(1e-9, tr / 2 - Math.sqrt(disc));
  const angle =
    Math.abs(cxy) < 1e-12
      ? vx >= vy
        ? 0
        : 90
      : (Math.atan2(l1 - vx, cxy) * 180) / Math.PI;
  return { rx: Math.sqrt(l1), ry: Math.sqrt(l2), angle };
}

function KeypointLayers({
  keypoints,
  unit,
  idPrefix,
  showKeypoints = true,
  showHeatmap = false,
  showEllipses = true,
  showSkeleton = false,
  highlightIndices = null,
  criterionKey = null,
  sampler = null,
}) {
  const [hovered, setHovered] = useState(null);
  const spotlight = Array.isArray(highlightIndices);
  const isLit = (i) => !spotlight || highlightIndices.includes(i);
  // Resolve an endpoint (index or [a, b] midpoint) to a point, or null if missing.
  const resolve = (e) => {
    if (!Array.isArray(e)) return keypoints[e] ?? null;
    const a = keypoints[e[0]];
    const b = keypoints[e[1]];
    return a && b ? { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, mid: true } : null;
  };
  const segments = (criterionKey && MEASUREMENT_SEGMENTS[criterionKey]) || [];
  // Marker colors adapt to the photo under each landmark when it is readable.
  const markerStyles = useMemo(
    () => computeMarkerStyles(keypoints, sampler),
    [keypoints, sampler],
  );

  /** Line color from the photo along the segment (5 samples, averaged). */
  const lineStyle = (p, q) => {
    if (!sampler) return pickLineColor(null);
    const acc = [0, 0, 0];
    let n = 0;
    for (let t = 0; t <= 1.001; t += 0.25) {
      const c = sampler(p.x + (q.x - p.x) * t, p.y + (q.y - p.y) * t);
      if (c) {
        acc[0] += c[0];
        acc[1] += c[1];
        acc[2] += c[2];
        n++;
      }
    }
    return pickLineColor(n ? acc.map((v) => v / n) : null);
  };
  const glowId = `${idPrefix}-glow`;
  const heatId = `${idPrefix}-heat`;

  return (
    <g>
      <defs>
        {spotlight && showKeypoints && (
          <filter id={glowId} x="-200%" y="-200%" width="500%" height="500%">
            <feGaussianBlur stdDeviation={unit * 1.4} result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        )}
        {showHeatmap && (
          <radialGradient id={heatId}>
            <stop offset="0%" stopColor="#ff3b00" stopOpacity="0.85" />
            <stop offset="35%" stopColor="#ffb300" stopOpacity="0.55" />
            <stop offset="70%" stopColor="#ffeb3b" stopOpacity="0.2" />
            <stop offset="100%" stopColor="#ffeb3b" stopOpacity="0" />
          </radialGradient>
        )}
      </defs>

      {showHeatmap &&
        keypoints.map((kp) => {
          if (!isLit(kp.index)) return null;
          const radius = Math.min(
            Math.max(Math.max(kp.sigma_x, kp.sigma_y) * 3, unit * 6),
            unit * 22,
          );
          return (
            <circle
              key={`h-${kp.index}`}
              cx={kp.x}
              cy={kp.y}
              r={radius}
              fill={`url(#${heatId})`}
              opacity={kp.low_visibility ? 0.45 : 1}
            />
          );
        })}

      {showKeypoints && showSkeleton &&
        SKELETON_EDGES.map(([a, b]) => {
          const p = keypoints[a];
          const q = keypoints[b];
          if (!p || !q) return null;
          if (spotlight && !(isLit(a) && isLit(b))) return null;
          const dim = p.low_visibility || q.low_visibility;
          return (
            <line
              key={`${a}-${b}`}
              x1={p.x}
              y1={p.y}
              x2={q.x}
              y2={q.y}
              stroke={dim ? "#94a3b8" : "#22d3ee"}
              strokeWidth={unit * 0.9}
              strokeOpacity={spotlight ? 0.85 : dim ? 0.35 : 0.75}
              strokeLinecap="round"
              strokeDasharray={dim ? `${unit * 2} ${unit * 2}` : undefined}
            />
          );
        })}

      {showKeypoints &&
        segments.map(([ea, eb], i) => {
          const p = resolve(ea);
          const q = resolve(eb);
          if (!p || !q) return null;
          const { line, halo } = lineStyle(p, q);
          return (
            <g key={`seg-${i}`} pointerEvents="none">
              <line
                x1={p.x} y1={p.y} x2={q.x} y2={q.y}
                stroke={halo} strokeWidth={unit * 2} strokeOpacity={0.7}
                strokeLinecap="round"
              />
              <line
                x1={p.x} y1={p.y} x2={q.x} y2={q.y}
                stroke={line} strokeWidth={unit * 1}
                strokeLinecap="round"
              />
              {[p, q].map((pt, j) =>
                pt.mid ? (
                  <rect
                    key={j}
                    x={pt.x - unit * 1.2} y={pt.y - unit * 1.2}
                    width={unit * 2.4} height={unit * 2.4}
                    transform={`rotate(45 ${pt.x} ${pt.y})`}
                    fill={line} stroke={halo} strokeWidth={unit * 0.5}
                  />
                ) : null,
              )}
            </g>
          );
        })}

      {showKeypoints &&
        segments.map(([ea, eb], i) => {
          const p = resolve(ea);
          const q = resolve(eb);
          if (!p || !q) return null;
          return (
            <g key={`seg-${i}`} pointerEvents="none">
              <line
                x1={p.x} y1={p.y} x2={q.x} y2={q.y}
                stroke="#ffffff" strokeWidth={unit * 2} strokeOpacity={0.7}
                strokeLinecap="round"
              />
              <line
                x1={p.x} y1={p.y} x2={q.x} y2={q.y}
                stroke="#0891b2" strokeWidth={unit * 1}
                strokeLinecap="round"
              />
              {[p, q].map((pt, j) =>
                pt.mid ? (
                  <rect
                    key={j}
                    x={pt.x - unit * 1.2} y={pt.y - unit * 1.2}
                    width={unit * 2.4} height={unit * 2.4}
                    transform={`rotate(45 ${pt.x} ${pt.y})`}
                    fill="#0891b2" stroke="#ffffff" strokeWidth={unit * 0.5}
                  />
                ) : null,
              )}
            </g>
          );
        })}

      {showKeypoints && showEllipses &&
        keypoints.map((kp) => {
          if (!isLit(kp.index)) return null;
          const { rx, ry, angle } = ellipseParams(kp.sigma_x, kp.sigma_y, kp.rho);
          return (
            <ellipse
              key={`e-${kp.index}`}
              cx={kp.x}
              cy={kp.y}
              rx={Math.max(rx, unit * 0.5)}
              ry={Math.max(ry, unit * 0.5)}
              transform={`rotate(${angle} ${kp.x} ${kp.y})`}
              fill="#f59e0b"
              fillOpacity={0.16}
              stroke="#f59e0b"
              strokeWidth={unit * 0.4}
              strokeOpacity={0.6}
            />
          );
        })}

      {showKeypoints &&
        keypoints.map((kp) => {
          if (!isLit(kp.index)) return null;
          const { fill: color, stroke: ring } = markerStyles[kp.index];
          return (
            <g key={`p-${kp.index}`} opacity={kp.low_visibility ? 0.55 : 1}>
              {spotlight && (
                <circle
                  cx={kp.x}
                  cy={kp.y}
                  r={unit * 3.2}
                  fill={color}
                  fillOpacity={0.45}
                  filter={`url(#${glowId})`}
                />
              )}
              <circle
                cx={kp.x}
                cy={kp.y}
                r={(spotlight ? unit * 2.3 : unit * 1.7) * (hovered === kp.index ? 1.8 : 1)}
                fill={color}
                stroke={ring}
                strokeWidth={unit * 0.6}
                style={{ transition: "r 120ms ease-out" }}
              />
              {/* Oversized transparent target so small dots are easy to hover. */}
              <circle
                cx={kp.x}
                cy={kp.y}
                r={unit * 3.5}
                fill="transparent"
                style={{ cursor: "pointer" }}
                onPointerEnter={() => setHovered(kp.index)}
                onPointerLeave={() => setHovered((h) => (h === kp.index ? null : h))}
              />
            </g>
          );
        })}

      {showKeypoints &&
        keypoints.map((kp) => {
          if (kp.index !== hovered || !isLit(kp.index)) return null;
          const fontSize = unit * 5.5;
          const w = (kp.label ?? kp.name).length * fontSize * 0.56 + unit * 6;
          const h = fontSize + unit * 4;
          const y = kp.y - unit * 6 - h;
          return (
            <g key={`t-${kp.index}`} pointerEvents="none">
              <rect
                x={kp.x - w / 2}
                y={y}
                width={w}
                height={h}
                rx={unit * 1.5}
                fill="#0f172a"
                fillOpacity={0.92}
              />
              <text
                x={kp.x}
                y={y + h / 2}
                fill="#ffffff"
                fontSize={fontSize}
                textAnchor="middle"
                dominantBaseline="central"
                fontFamily="Inter, sans-serif"
              >
                {kp.label ?? kp.name}
              </text>
            </g>
          );
        })}
    </g>
  );
}

export default KeypointLayers;
