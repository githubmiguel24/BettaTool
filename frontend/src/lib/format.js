/** Angles read in degrees; every other criterion is a dimensionless ratio. */
export function formatValue(criterionKey, value, uncertainty) {
  const isAngle = criterionKey === "caudal-spread-angle";
  const digits = isAngle ? 1 : 3;
  const unit = isAngle ? "°" : "";
  return `${value.toFixed(digits)}${unit} ± ${uncertainty.toFixed(digits)}${unit}`;
}

/** Backend decision string -> "Pass" | "Defer" | "Fault" (unknown -> Defer). */
export function decisionKind(decision) {
  if (decision === "Confident Pass") return "Pass";
  if (decision === "Confident Fault") return "Fault";
  return "Defer";
}

export function formatDate(iso) {
  return new Date(iso).toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });
}
