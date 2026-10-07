/**
 * Picks overlay colors that stay readable on top of the uploaded photo.
 * `useImageSampler` rasterizes the image once (downscaled) so the overlay can
 * ask "what's the average color around this point / along this line?", and the
 * pick* helpers choose the first preferred color with enough WCAG contrast
 * against it. If the image can't be read (cross-origin without CORS headers),
 * the sampler is null and callers fall back to the static palette.
 */

import { useEffect, useState } from "react";
import { colorForGroup } from "../data/keypointGroups.js";

const MAX_SIDE = 512;
const cache = new Map(); // url -> Promise<sampler | null>

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function luminance([r, g, b]) {
  const lin = (v) => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

export function contrast(a, b) {
  const la = luminance(a);
  const lb = luminance(b);
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

function mix(rgb, target, t) {
  return rgb.map((v, i) => Math.round(v + (target[i] - v) * t));
}

const toHex = (rgb) => `#${rgb.map((v) => v.toString(16).padStart(2, "0")).join("")}`;

/** Halo color that separates a marker from both itself and the photo. */
function halo(fillRgb) {
  return luminance(fillRgb) > 0.4 ? "#0f172a" : "#ffffff";
}

const MIN_CONTRAST = 3;
// Share of a part's points that must clear MIN_CONTRAST for a shade to be accepted.
const MIN_COVERAGE = 0.6;

/**
 * One fill color for a whole body part, so every point of a part always looks
 * the same. Keeps the part's base hue and only shifts its lightness (the halo
 * stroke handles any point that still sits on a tricky patch of photo): the
 * first shade that contrasts with most of the part's local backgrounds wins,
 * otherwise the shade that clears the most points.
 */
export function pickGroupColor(baseHex, bgSamples) {
  const bgs = bgSamples.filter(Boolean);
  if (!bgs.length) return baseHex;
  const base = hexToRgb(baseHex);
  const white = [255, 255, 255];
  const black = [0, 0, 0];
  const candidates = [
    base,
    mix(base, white, 0.35),
    mix(base, black, 0.35),
    mix(base, white, 0.6),
    mix(base, black, 0.6),
  ];
  let best = candidates[0];
  let bestScore = [-1, -1];
  for (const c of candidates) {
    const ratios = bgs.map((bg) => contrast(c, bg));
    const coverage = ratios.filter((r) => r >= MIN_CONTRAST).length / ratios.length;
    if (coverage >= MIN_COVERAGE) return toHex(c);
    const mean = ratios.reduce((a, b) => a + b, 0) / ratios.length;
    if (coverage > bestScore[0] || (coverage === bestScore[0] && mean > bestScore[1])) {
      best = c;
      bestScore = [coverage, mean];
    }
  }
  return toHex(best);
}

/**
 * { [kp.index]: { fill, stroke } } - the dot colors drawn on the photo. Every
 * keypoint in a group shares one fill (adapted to the photo under that group).
 */
export function computeMarkerStyles(keypoints, sampler) {
  const byGroup = new Map();
  for (const kp of keypoints ?? []) {
    if (!byGroup.has(kp.group)) byGroup.set(kp.group, []);
    byGroup.get(kp.group).push(kp);
  }
  const out = {};
  for (const [group, kps] of byGroup) {
    const fill = pickGroupColor(
      colorForGroup(group),
      kps.map((kp) => (sampler ? sampler(kp.x, kp.y) : null)),
    );
    const style = { fill, stroke: halo(hexToRgb(fill)) };
    for (const kp of kps) out[kp.index] = style;
  }
  return out;
}

/** Measurement-line color: preferred cyan, then alternatives, by contrast. */
export function pickLineColor(bgRgb) {
  const preferred = ["#0891b2", "#22d3ee", "#e11d48", "#facc15", "#ffffff", "#0f172a"];
  if (!bgRgb) return { line: preferred[0], halo: "#ffffff" };
  let best = preferred[0];
  let bestC = 0;
  for (const hex of preferred) {
    const ratio = contrast(hexToRgb(hex), bgRgb);
    if (ratio >= 3.5) return { line: hex, halo: halo(hexToRgb(hex)) };
    if (ratio > bestC) {
      best = hex;
      bestC = ratio;
    }
  }
  return { line: best, halo: halo(hexToRgb(best)) };
}

function buildSampler(img, w, h) {
  const k = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
  const cw = Math.max(1, Math.round(img.naturalWidth * k));
  const ch = Math.max(1, Math.round(img.naturalHeight * k));
  const canvas = document.createElement("canvas");
  canvas.width = cw;
  canvas.height = ch;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, cw, ch);
  const { data } = ctx.getImageData(0, 0, cw, ch); // throws if the canvas is tainted
  const sx = cw / w;
  const sy = ch / h;
  const radius = Math.max(w, h) / 80; // image px

  /** Mean [r,g,b] in a small window around (x, y), in original image pixels. */
  return function sample(x, y) {
    const cx = Math.round(x * sx);
    const cy = Math.round(y * sy);
    const rx = Math.max(1, Math.round(radius * sx));
    const ry = Math.max(1, Math.round(radius * sy));
    let r = 0;
    let g = 0;
    let b = 0;
    let n = 0;
    for (let yy = Math.max(0, cy - ry); yy <= Math.min(ch - 1, cy + ry); yy++) {
      for (let xx = Math.max(0, cx - rx); xx <= Math.min(cw - 1, cx + rx); xx++) {
        const i = (yy * cw + xx) * 4;
        r += data[i];
        g += data[i + 1];
        b += data[i + 2];
        n++;
      }
    }
    return n ? [r / n, g / n, b / n] : null;
  };
}

function loadSampler(url, w, h) {
  const key = `${url}|${w}x${h}`;
  if (!cache.has(key)) {
    cache.set(
      key,
      new Promise((resolve) => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
          try {
            resolve(buildSampler(img, w, h));
          } catch {
            resolve(null);
          }
        };
        img.onerror = () => resolve(null);
        img.src = url;
      }),
    );
  }
  return cache.get(key);
}

/** Returns sample(x, y) -> [r,g,b], or null until/unless the image is readable. */
export function useImageSampler(url, w, h) {
  const [sampler, setSampler] = useState(null);
  useEffect(() => {
    let live = true;
    setSampler(null);
    if (url && w && h) {
      loadSampler(url, w, h).then((s) => {
        if (live) setSampler(() => s);
      });
    }
    return () => {
      live = false;
    };
  }, [url, w, h]);
  return sampler;
}
