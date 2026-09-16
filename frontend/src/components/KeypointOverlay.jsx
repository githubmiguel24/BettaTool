/**
 * Draws the perceptual tier's output over the uploaded image.
 *
 * Coordinates arrive from the API in ORIGINAL uploaded-image pixels, so the
 * SVG simply declares a viewBox of the original image dimensions and lets
 * the browser handle scaling. No manual letterbox math on this side - that
 * belongs in app/perception/geometry.py and nowhere else.
 *
 * Each landmark is drawn as a dot plus its 1-sigma uncertainty ellipse, so
 * the covariance head's output is visible rather than buried in the JSON.
 * That ellipse is the whole point of the probabilistic tier: a confident
 * landmark shows a tight dot, an uncertain one visibly blooms.
 */

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

function KeypointOverlay({ report, showEllipses = true, showSkeleton = true }) {
  if (!report?.keypoints?.length) return null;

  const { keypoints, image_width: w, image_height: h } = report;
  // Stroke widths must scale with the image or they vanish on a 4000px photo.
  const unit = Math.max(w, h) / 400;

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="pointer-events-none absolute inset-0 h-full w-full"
      preserveAspectRatio="xMidYMid meet"
    >
      {showSkeleton &&
        SKELETON_EDGES.map(([a, b]) => {
          const p = keypoints[a];
          const q = keypoints[b];
          if (!p || !q) return null;
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
              strokeOpacity={dim ? 0.35 : 0.75}
              strokeLinecap="round"
              strokeDasharray={dim ? `${unit * 2} ${unit * 2}` : undefined}
            />
          );
        })}

      {showEllipses &&
        keypoints.map((kp) => {
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

      {keypoints.map((kp) => (
        <g key={`p-${kp.index}`}>
          <circle
            cx={kp.x}
            cy={kp.y}
            r={unit * 1.7}
            fill={kp.low_visibility ? "#94a3b8" : "#ef4444"}
            stroke="#ffffff"
            strokeWidth={unit * 0.6}
          />
        </g>
      ))}
    </svg>
  );
}

export default KeypointOverlay;
