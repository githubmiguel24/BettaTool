/**
 * Colors for the anatomical `group` field the API attaches to each keypoint
 * (see backend `app/perception/keypoints.py::KeypointGroup`). Keyed by that
 * same string so this can never drift from the model's grouping - only the
 * color choice lives here.
 *
 * Palette: dataviz skill's default categorical slots 1/2/3/7 (blue/orange/
 * aqua/violet), chosen because that specific combination is the one that
 * clears the all-pairs CVD and normal-vision floors (`validate_palette.js
 * --pairs all`) required when every color can appear on screen at once, as
 * they do here.
 */
export const GROUP_COLORS = {
  head: "#2a78d6",
  dorsal_fin: "#eb6834",
  caudal_fin: "#1baf7a",
  anal_fin: "#4a3aa7",
};

export const GROUP_LABELS = {
  head: "Head",
  dorsal_fin: "Dorsal Fin",
  caudal_fin: "Caudal Fin",
  anal_fin: "Anal Fin",
};

export const DEFAULT_GROUP_COLOR = "#64748b"; // slate-500, fallback for an unrecognized group

export function colorForGroup(group) {
  return GROUP_COLORS[group] ?? DEFAULT_GROUP_COLOR;
}
