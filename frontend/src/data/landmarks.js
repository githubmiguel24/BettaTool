/**
 * Static landmark legend, index-aligned with the backend's
 * `app/perception/keypoints.py` Keypoint enum.
 *
 * Prefer rendering from an API response's `keypoints[]` (each carries its own
 * `label`) wherever a report is available - that cannot drift. This list is
 * only for views that render the legend before any analysis has run.
 *
 * NOTE: an earlier version of this file listed a "Caudal Fin Peduncle/
 * Anterior" landmark that the model does not predict, omitted
 * `peduncle_bottom` entirely, and ordered the dorsal/caudal points
 * differently from the model's output indices. Keep this list in the enum's
 * order.
 */
export const landmarkPoints = [
  { label: "Snout Tip", tone: "bg-amber-400" },
  { label: "Eye Center", tone: "bg-emerald-400" },
  { label: "Dorsal Fin Base (Anterior)", tone: "bg-amber-400" },
  { label: "Dorsal Fin Base (Posterior)", tone: "bg-emerald-400" },
  { label: "Dorsal Fin Tip", tone: "bg-red-400" },
  { label: "Caudal Peduncle Top", tone: "bg-amber-400" },
  { label: "Caudal Peduncle Bottom", tone: "bg-emerald-400" },
  { label: "Caudal Fin Tip (Upper)", tone: "bg-red-400" },
  { label: "Caudal Fin Tip (Lower)", tone: "bg-amber-400" },
  { label: "Caudal Fin Center", tone: "bg-emerald-400" },
  { label: "Anal Fin Base (Anterior)", tone: "bg-red-400" },
  { label: "Anal Fin Base (Posterior)", tone: "bg-amber-400" },
  { label: "Anal Fin Tip", tone: "bg-emerald-400" },
];

const mid = Math.ceil(landmarkPoints.length / 2);
export const landmarkColumns = [
  landmarkPoints.slice(0, mid),
  landmarkPoints.slice(mid),
];
