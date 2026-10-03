/**
 * Line segments each measurement is computed from, keyed by criterion_key
 * (same keys as backend MORPHOMETRIC_FUNCTIONS). Mirrors the geometry in
 * backend app/analytical/morphometrics.py. An endpoint is a keypoint index,
 * or [a, b] for the midpoint of two keypoints (peduncle / fin-base midpoints).
 */
const PED = [5, 6]; // caudal peduncle midpoint
const DORSAL_BASE = [2, 3];
const ANAL_BASE = [10, 11];

const BODY = [0, PED]; // snout tip -> peduncle midpoint
const DORSAL = [DORSAL_BASE, 4];
const ANAL = [ANAL_BASE, 12];
const CAUDAL = [PED, 9]; // peduncle midpoint -> caudal fin center
const ANAL_WIDTH = [10, 11]; // anal base front corner -> rear corner

export const MEASUREMENT_SEGMENTS = {
  "caudal-spread-angle": [[PED, 7], [PED, 8], CAUDAL],
  "dorsal-body-ratio": [DORSAL, BODY],
  "anal-body-ratio": [ANAL, BODY],
  "caudal-body-ratio": [CAUDAL, BODY],
  "anal-length-width-ratio": [ANAL, ANAL_WIDTH],
};
