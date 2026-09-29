/**
 * The annotated-image viewer shared by the Upload and Report pages.
 *
 * - While the pipeline runs (`loading`, no report yet) the plain photo is shown
 *   under a decorative "annotating" animation. It is NOT model output: the
 *   dots are fixed decoration and vanish when real results arrive.
 * - Once a report exists, the photo is drawn INSIDE an <svg> together with the
 *   keypoint layers, so zooming is just animating the viewBox and the overlay
 *   can never drift out of alignment with the image.
 * - Selecting a measurement zooms to the landmarks it reads; clicking the
 *   image opens a lightbox with wheel-zoom and drag-to-pan.
 */

import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import KeypointLayers from "./KeypointOverlay.jsx";
import { colorForGroup } from "../data/keypointGroups.js";

const ZOOM_MS = 380;
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

/** ViewBox that frames the given landmarks, at the image's aspect ratio. */
function focusBox(keypoints, indices, w, h) {
  const full = { x: 0, y: 0, w, h };
  if (!indices?.length) return full;
  const pts = keypoints.filter((k) => indices.includes(k.index));
  if (!pts.length) return full;

  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const [minX, maxX] = [Math.min(...xs), Math.max(...xs)];
  const [minY, maxY] = [Math.min(...ys), Math.max(...ys)];
  const aspect = w / h;

  let bw = Math.max((maxX - minX) * 1.6, (maxY - minY) * 1.6 * aspect, w * 0.25);
  bw = Math.min(bw, w);
  const bh = bw / aspect;
  return {
    x: clamp((minX + maxX) / 2 - bw / 2, 0, w - bw),
    y: clamp((minY + maxY) / 2 - bh / 2, 0, h - bh),
    w: bw,
    h: bh,
  };
}

function AnnotatedImage({
  imageUrl,
  report,
  highlightIndices,
  showKeypoints,
  showHeatmap,
  interactive = false,
}) {
  const w = report.image_width;
  const h = report.image_height;
  const idPrefix = useId().replace(/:/g, "");
  const svgRef = useRef(null);
  const dragRef = useRef(null);
  const rafRef = useRef(0);

  const focusKey = highlightIndices?.join(",") ?? "";
  const target = useMemo(
    () => focusBox(report.keypoints, highlightIndices, w, h),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [report.keypoints, focusKey, w, h],
  );

  const [view, setView] = useState(target);
  const viewRef = useRef(target);
  const applyView = useCallback((v) => {
    viewRef.current = v;
    setView(v);
  }, []);

  const animateTo = useCallback(
    (to) => {
      cancelAnimationFrame(rafRef.current);
      const from = viewRef.current;
      const t0 = performance.now();
      const step = (now) => {
        const t = Math.min(1, (now - t0) / ZOOM_MS);
        const e = 1 - (1 - t) ** 3;
        applyView({
          x: from.x + (to.x - from.x) * e,
          y: from.y + (to.y - from.y) * e,
          w: from.w + (to.w - from.w) * e,
          h: from.h + (to.h - from.h) * e,
        });
        if (t < 1) rafRef.current = requestAnimationFrame(step);
      };
      rafRef.current = requestAnimationFrame(step);
    },
    [applyView],
  );

  useEffect(() => {
    animateTo(target);
    return () => cancelAnimationFrame(rafRef.current);
  }, [target, animateTo]);

  // Wheel needs a non-passive native listener so the page doesn't scroll.
  useEffect(() => {
    const svg = svgRef.current;
    if (!interactive || !svg) return undefined;
    const onWheel = (e) => {
      e.preventDefault();
      cancelAnimationFrame(rafRef.current);
      const ctm = svg.getScreenCTM();
      if (!ctm) return;
      const px = (e.clientX - ctm.e) / ctm.a;
      const py = (e.clientY - ctm.f) / ctm.d;
      const v = viewRef.current;
      const nw = clamp(v.w * Math.exp(e.deltaY * 0.0015), w / 12, w);
      const nh = nw * (h / w);
      const s = nw / v.w;
      applyView({
        w: nw,
        h: nh,
        x: clamp(px - (px - v.x) * s, 0, w - nw),
        y: clamp(py - (py - v.y) * s, 0, h - nh),
      });
    };
    svg.addEventListener("wheel", onWheel, { passive: false });
    return () => svg.removeEventListener("wheel", onWheel);
  }, [interactive, w, h, applyView]);

  function onPointerDown(e) {
    if (!interactive) return;
    cancelAnimationFrame(rafRef.current);
    e.currentTarget.setPointerCapture(e.pointerId);
    dragRef.current = { x: e.clientX, y: e.clientY };
  }

  function onPointerMove(e) {
    if (!dragRef.current) return;
    const ctm = svgRef.current.getScreenCTM();
    if (!ctm) return;
    const dx = (e.clientX - dragRef.current.x) / ctm.a;
    const dy = (e.clientY - dragRef.current.y) / ctm.d;
    dragRef.current = { x: e.clientX, y: e.clientY };
    const v = viewRef.current;
    applyView({
      ...v,
      x: clamp(v.x - dx, 0, w - v.w),
      y: clamp(v.y - dy, 0, h - v.h),
    });
  }

  const unit = Math.max(view.w, view.h) / 400;

  return (
    <svg
      ref={svgRef}
      viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
      preserveAspectRatio="xMidYMid meet"
      className={`h-full w-full ${interactive ? "cursor-grab touch-none active:cursor-grabbing" : ""}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={() => (dragRef.current = null)}
      onPointerCancel={() => (dragRef.current = null)}
      onDoubleClick={() => interactive && animateTo(target)}
    >
      {imageUrl && <image href={imageUrl} x={0} y={0} width={w} height={h} />}
      <KeypointLayers
        keypoints={report.keypoints}
        unit={unit}
        idPrefix={idPrefix}
        showKeypoints={showKeypoints}
        showHeatmap={showHeatmap}
        highlightIndices={highlightIndices}
      />
    </svg>
  );
}

// Decorative only: [x%, y%, group], ordered head -> dorsal -> caudal -> anal.
const SCAN_POINTS = [
  [20, 48, "head"], [25, 40, "head"], [25, 58, "head"],
  [42, 28, "dorsal_fin"], [52, 22, "dorsal_fin"], [62, 30, "dorsal_fin"],
  [78, 38, "caudal_fin"], [84, 50, "caudal_fin"], [80, 62, "caudal_fin"], [88, 44, "caudal_fin"],
  [46, 68, "anal_fin"], [54, 74, "anal_fin"], [62, 68, "anal_fin"],
];

const SCAN_STEPS = [
  "Locating head landmarks",
  "Tracing dorsal fin",
  "Tracing caudal fin",
  "Tracing anal fin",
  "Computing measurements",
];

function AnalyzingOverlay() {
  const [step, setStep] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setStep((s) => (s + 1) % SCAN_STEPS.length), 1000);
    return () => clearInterval(timer);
  }, []);

  const bracket = "absolute h-6 w-6 border-cyan-300";
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden bg-slate-900/25">
      <div
        className="absolute inset-0 opacity-40"
        style={{
          backgroundImage:
            "linear-gradient(rgba(103,232,249,0.25) 1px, transparent 1px), linear-gradient(90deg, rgba(103,232,249,0.25) 1px, transparent 1px)",
          backgroundSize: "32px 32px",
        }}
      />
      <div className="animate-scan absolute left-0 right-0 h-1/5 bg-gradient-to-b from-transparent to-cyan-300/50">
        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-cyan-200 shadow-[0_0_12px_2px_rgba(103,232,249,0.9)]" />
      </div>

      <span className={`${bracket} left-3 top-3 border-l-2 border-t-2`} />
      <span className={`${bracket} right-3 top-3 border-r-2 border-t-2`} />
      <span className={`${bracket} bottom-3 left-3 border-b-2 border-l-2`} />
      <span className={`${bracket} bottom-3 right-3 border-b-2 border-r-2`} />

      {SCAN_POINTS.map(([x, y, group], i) => (
        <span
          key={i}
          className="absolute -translate-x-1/2 -translate-y-1/2"
          style={{ left: `${x}%`, top: `${y}%` }}
        >
          <span
            className="animate-kp-pop block h-3.5 w-3.5 rounded-full ring-2 ring-white"
            style={{
              backgroundColor: colorForGroup(group),
              animationDelay: `${i * 0.32}s`,
            }}
          />
        </span>
      ))}

      <div className="absolute bottom-5 left-1/2 flex -translate-x-1/2 items-center gap-2.5 rounded-full bg-slate-900/70 px-4 py-2 text-sm font-medium text-white backdrop-blur">
        <span className="h-2 w-2 animate-pulse rounded-full bg-cyan-300" />
        {SCAN_STEPS[step]}&hellip;
      </div>
    </div>
  );
}

export function OverlayToggles({
  showKeypoints,
  setShowKeypoints,
  showHeatmap,
  setShowHeatmap,
  dark = false,
}) {
  const box = "h-4 w-4 rounded border-slate-300";
  return (
    <div
      className={`flex items-center gap-4 text-sm ${dark ? "text-slate-200" : "text-slate-500"}`}
    >
      <label className="flex cursor-pointer items-center gap-2">
        <input
          type="checkbox"
          checked={showKeypoints}
          onChange={(e) => setShowKeypoints(e.target.checked)}
          className={box}
        />
        Keypoints
      </label>
      <label className="flex cursor-pointer items-center gap-2">
        <input
          type="checkbox"
          checked={showHeatmap}
          onChange={(e) => setShowHeatmap(e.target.checked)}
          className={box}
        />
        Heatmap
      </label>
    </div>
  );
}

function Lightbox({ onClose, toggles, ...viewerProps }) {
  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex flex-col bg-slate-950/90 p-6"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="mb-4 flex flex-wrap items-center justify-between gap-4">
        <OverlayToggles {...toggles} dark />
        <div className="flex items-center gap-5">
          <span className="hidden text-xs text-slate-400 sm:inline">
            Scroll to zoom &middot; drag to pan &middot; double-click to reset
          </span>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="flex h-10 w-10 items-center justify-center rounded-full bg-white/10 text-2xl leading-none text-white transition hover:bg-white/20"
          >
            &times;
          </button>
        </div>
      </div>
      <div
        className="min-h-0 flex-1"
        onClick={(e) => e.target === e.currentTarget && onClose()}
      >
        <AnnotatedImage {...viewerProps} interactive />
      </div>
    </div>,
    document.body,
  );
}

function ImageStage({
  imageUrl,
  alt,
  report,
  loading = false,
  highlightIndices = null,
  showKeypoints,
  setShowKeypoints,
  showHeatmap,
  setShowHeatmap,
}) {
  const [expanded, setExpanded] = useState(false);
  const closeLightbox = useCallback(() => setExpanded(false), []);

  const viewerProps = { imageUrl, report, highlightIndices, showKeypoints, showHeatmap };

  return (
    <div className="relative flex aspect-[4/3] w-full items-center justify-center overflow-hidden rounded-xl border border-slate-200 bg-slate-50">
      {report ? (
        <button
          type="button"
          onClick={() => setExpanded(true)}
          aria-label="Enlarge image"
          className="block h-full w-full cursor-zoom-in"
        >
          <AnnotatedImage key={report.id} {...viewerProps} />
        </button>
      ) : (
        <img src={imageUrl} alt={alt} className="h-full w-full object-contain" />
      )}

      {loading && <AnalyzingOverlay />}

      {report && (
        <span className="pointer-events-none absolute bottom-3 right-3 rounded-full bg-slate-900/60 px-3 py-1 text-xs text-white">
          Click to enlarge
        </span>
      )}

      {expanded && report && (
        <Lightbox
          {...viewerProps}
          onClose={closeLightbox}
          toggles={{ showKeypoints, setShowKeypoints, showHeatmap, setShowHeatmap }}
        />
      )}
    </div>
  );
}

export default ImageStage;
