/** Angles read in degrees; every other criterion is a dimensionless ratio. */
export function formatValue(criterionKey, value, uncertainty) {
  const isAngle = criterionKey === "caudal-spread-angle";
  const digits = isAngle ? 1 : 3;
  const unit = isAngle ? "°" : "";
  return `${value.toFixed(digits)}${unit} ± ${uncertainty.toFixed(digits)}${unit}`;
}

/** One verdict per report: any fault wins, then any deferral, else pass. */
export function overallStatus(decisions) {
  if (decisions.includes("Confident Fault")) return "Fault";
  if (decisions.includes("Defer to Judge")) return "Defer";
  return "Pass";
}

export function formatDate(iso) {
  return new Date(iso).toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });
}
