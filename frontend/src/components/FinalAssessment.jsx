import { getFinalAssessment } from "../lib/assessment.js";

const TONES = {
  "Perfect Quality": "border-emerald-200 bg-emerald-50 text-emerald-800",
  "Great Quality": "border-emerald-200 bg-emerald-50 text-emerald-800",
  "Good Quality": "border-lime-200 bg-lime-50 text-lime-800",
  "Fair Quality": "border-amber-200 bg-amber-50 text-amber-800",
  "Low Quality": "border-orange-200 bg-orange-50 text-orange-800",
  "Poor Quality": "border-red-200 bg-red-50 text-red-800",
  Inconclusive: "border-amber-200 bg-amber-50 text-amber-800",
  Unavailable: "border-red-200 bg-red-50 text-red-800",
};

function FinalAssessment({ report }) {
  const result = getFinalAssessment(report);
  const tone = TONES[result.quality] ?? TONES.Inconclusive;
  const percentage = Math.round(result.confidence * 100);

  return (
    <section
      aria-labelledby="final-assessment-heading"
      className={`mb-5 rounded-2xl border p-5 shadow-sm ${tone}`}
    >
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2
            id="final-assessment-heading"
            className="text-sm font-bold uppercase tracking-wider opacity-75"
          >
            Final Assessment
          </h2>
          <p className="mt-2 text-3xl font-bold">{result.quality}</p>
          <p className="mt-2 max-w-3xl text-sm leading-6">{result.summary}</p>
        </div>
        <div className="min-w-44 rounded-xl bg-white/70 px-4 py-3">
          <p className="text-xs font-semibold uppercase tracking-wide opacity-70">
            Decision confidence
          </p>
          <p className="mt-1 text-2xl font-bold">{percentage}%</p>
          <p className="text-xs">
            {result.total - result.deferred}/{result.total} criteria confidently classified
          </p>
        </div>
      </div>
    </section>
  );
}

export default FinalAssessment;
