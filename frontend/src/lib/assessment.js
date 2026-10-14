import { measurements as CRITERIA } from "../data/measurements.js";
import { decisionKind } from "./format.js";

const QUALITY_TIERS = [
  "Poor Quality",
  "Low Quality",
  "Fair Quality",
  "Good Quality",
  "Great Quality",
  "Perfect Quality",
];

/** Summarizes confident criterion outcomes; confidence is decision coverage, not probability. */
export function getFinalAssessment(report) {
  const total = CRITERIA.length;
  const byKey = new Map(
    (report?.measurements ?? []).map((measurement) => [
      measurement.criterion_key,
      measurement,
    ]),
  );
  let passed = 0;
  let faulted = 0;
  let deferred = 0;

  for (const { key } of CRITERIA) {
    const measurement = byKey.get(key);
    const kind = measurement ? decisionKind(measurement.decision) : "Defer";
    if (kind === "Pass") passed += 1;
    else if (kind === "Fault") faulted += 1;
    else deferred += 1;
  }

  const confidence = total ? (passed + faulted) / total : 0;
  let quality;
  let summary;
  if (report && !report.model_trained) {
    quality = "Unavailable";
    summary = "A trained model is required before this result can be assessed.";
  } else if (deferred === total) {
    quality = "Inconclusive";
    summary = `${deferred} of ${total} criteria could not be confidently classified. Judge review is needed before a quality tier can be confirmed.`;
  } else {
    quality = QUALITY_TIERS[passed];
    summary = `${passed} of ${total} criteria passed confidently, ${faulted} had a confident fault, and ${deferred} need judge review. This project quality tier is based on the number of confidently passed criteria.`;
  }

  return { quality, summary, confidence, total, passed, faulted, deferred };
}
