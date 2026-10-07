/**
 * Photo + SVG overlay viewer with the same zoom behaviour as the Upload tab.
 *
 * The photo is drawn INSIDE the <svg> with the overlay, so zooming is just
 * changing the viewBox and the markers can never drift off the image. Clicking
 * the photo opens a lightbox with wheel-zoom, drag-to-pan and double-click to
 * reset. `children` is a render function that receives `r`, a marker radius in
 * image pixels that shrinks as you zoom in, so two nearby points can be told
 * apart instead of the markers swallowing the gap between them.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const MAX_ZOOM = 20;

function Viewer({ width: w, height: h, imageUrl, ariaLabel, onImageError, interactive = false, children }) {
  const svgRef = useRef(null);
  const dragRef = useRef(null);
  const full = { x: 0, y: 0, w, h };
  const [view, setView] = useState(full);
  const viewRef = useRef(full);
  const applyView = useCallback((v) => {
    viewRef.current = v;
    setView(v);
  }, []);

  // Wheel needs a non-passive native listener so the page doesn't scroll.
  useEffect(() => {
    const svg = svgRef.current;
    if (!interactive || !svg) return undefined;
    const onWheel = (e) => {
      e.preventDefault();
      const ctm = svg.getScreenCTM();
      if (!ctm) return;
      const px = (e.clientX - ctm.e) / ctm.a;
      const py = (e.clientY - ctm.f) / ctm.d;
      const v = viewRef.current;
      const nw = clamp(v.w * Math.exp(e.deltaY * 0.0015), w / MAX_ZOOM, w);
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
    applyView({ ...v, x: clamp(v.x - dx, 0, w - v.w), y: clamp(v.y - dy, 0, h - v.h) });
  }

  const r = Math.max(view.w, view.h) / 170;

  return (
    <svg
      ref={svgRef}
      viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
      preserveAspectRatio="xMidYMid meet"
      className={
        interactive
          ? "h-full w-full cursor-grab touch-none active:cursor-grabbing"
          : "block w-full rounded-xl bg-slate-100"
      }
      role="img"
      aria-label={ariaLabel}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={() => (dragRef.current = null)}
      onPointerCancel={() => (dragRef.current = null)}
      onDoubleClick={() => interactive && applyView(full)}
    >
      <image href={imageUrl} x="0" y="0" width={w} height={h} onError={onImageError} />
      {children(r)}
    </svg>
  );
}

function Lightbox({ onClose, toolbar, ...viewerProps }) {
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
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2">{toolbar?.(true)}</div>
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
      <div className="min-h-0 flex-1" onClick={(e) => e.target === e.currentTarget && onClose()}>
        <Viewer {...viewerProps} interactive />
      </div>
    </div>,
    document.body,
  );
}

/**
 * `toolbar(dark)` optionally renders the overlay toggles; it is called again
 * inside the lightbox so they stay reachable while zoomed in.
 */
function ZoomableImage({ toolbar, ...viewerProps }) {
  const [expanded, setExpanded] = useState(false);
  const close = useCallback(() => setExpanded(false), []);

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setExpanded(true)}
        aria-label="Enlarge image to zoom in"
        className="block w-full cursor-zoom-in"
      >
        <Viewer {...viewerProps} />
      </button>
      <span className="pointer-events-none absolute bottom-3 right-3 rounded-full bg-slate-900/60 px-3 py-1 text-xs text-white">
        Click to zoom
      </span>
      {expanded && <Lightbox {...viewerProps} toolbar={toolbar} onClose={close} />}
    </div>
  );
}

export default ZoomableImage;
