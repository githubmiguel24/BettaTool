import { useMemo } from "react";
import { GROUP_LABELS, colorForGroup } from "../data/keypointGroups.js";
import { computeMarkerStyles, useImageSampler } from "../lib/adaptiveColor.js";

/**
 * Group legend + per-landmark list with its localization uncertainty. Dot
 * colors match the (photo-adaptive) markers drawn on the image.
 */
function KeypointSummary({ keypoints, imageUrl, width, height }) {
  const sampler = useImageSampler(imageUrl, width, height);
  const styles = useMemo(() => computeMarkerStyles(keypoints, sampler), [keypoints, sampler]);
  if (!keypoints?.length) return null;
  // Legend swatches use the exact shade the group's points are drawn with.
  const groupStyle = (group) => {
    const kp = keypoints.find((k) => k.group === group);
    return styles[kp?.index] ?? { fill: colorForGroup(group), stroke: "#fff" };
  };
  return (
    <>
      <div className="mt-6 flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-slate-400">
        {Object.entries(GROUP_LABELS).map(([group, label]) => {
          const { fill, stroke } = groupStyle(group);
          return (
            <div key={group} className="flex items-center gap-1.5">
              <span
                className="h-2 w-2 shrink-0 rounded-full"
                style={{ backgroundColor: fill, boxShadow: `0 0 0 1px ${stroke}` }}
              />
              {label}
            </div>
          );
        })}
      </div>
      <div className="mt-3 grid grid-cols-2 gap-x-5 gap-y-2 text-sm text-slate-500">
        {keypoints.map((kp) => (
          <div key={kp.index} className="flex items-start gap-2.5">
            <span
              className="mt-1.5 h-2 w-2 shrink-0 rounded-full"
              style={{
                backgroundColor: styles[kp.index]?.fill,
                boxShadow: `0 0 0 1px ${styles[kp.index]?.stroke ?? "#fff"}`,
              }}
            />
            <span className="min-w-0">{kp.label}</span>
            <span className="ml-auto shrink-0 tabular-nums text-xs leading-5 text-slate-400">
              &plusmn;{Math.max(kp.sigma_x, kp.sigma_y).toFixed(1)}px
            </span>
          </div>
        ))}
      </div>
    </>
  );
}

export default KeypointSummary;
