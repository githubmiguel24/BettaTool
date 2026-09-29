import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import DashboardLayout from "../../components/DashboardLayout.jsx";
import ImageStage, { OverlayToggles } from "../../components/ImageStage.jsx";
import { measurements as CRITERIA } from "../../data/measurements.js";
import { GROUP_LABELS, colorForGroup } from "../../data/keypointGroups.js";
import { getReport } from "../../api/reports.js";
import { formatDate, formatValue } from "../../lib/format.js";
import {
  PlusIcon,
  ImageIcon,
  RulerIcon,
  FilePdfIcon,
  FileCsvIcon,
  CheckIcon,
  WarningIcon,
  AlertCircleIcon,
} from "../../components/Icons.jsx";

const statusStyles = {
  "Confident Pass": { icon: CheckIcon, badge: "bg-emerald-100 text-emerald-600" },
  "Defer to Judge": { icon: WarningIcon, badge: "bg-amber-100 text-amber-600" },
  "Confident Fault": { icon: AlertCircleIcon, badge: "bg-red-100 text-red-600" },
};

function downloadCsv(report) {
  const rows = [
    ["criterion", "value", "uncertainty", "tsi_px", "rmse_px", "decision"],
    ...report.measurements.map((m) => [
      m.criterion_key,
      m.value,
      m.uncertainty,
      m.tsi,
      m.rmse,
      m.decision,
    ]),
  ];
  const blob = new Blob([rows.map((r) => r.join(",")).join("\n")], {
    type: "text/csv",
  });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `report-${report.id.slice(0, 8)}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function ReportView() {
  const { id } = useParams();
  const [report, setReport] = useState(undefined); // undefined = loading, null = not found
  const [error, setError] = useState(null);
  const [selectedCriterion, setSelectedCriterion] = useState(null);
  const [showKeypoints, setShowKeypoints] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(false);

  useEffect(() => {
    setReport(undefined);
    getReport(id).then(setReport).catch((e) => setError(e.message));
  }, [id]);

  const newAnalysis = (
    <Link
      to="/upload"
      className="flex items-center gap-2 rounded-full bg-betta-950 px-6 py-3 text-base font-semibold text-white shadow-glow transition hover:bg-betta-900"
    >
      New Analysis
      <PlusIcon className="h-5 w-5" />
    </Link>
  );

  if (error || report === null || report === undefined) {
    return (
      <DashboardLayout actions={newAnalysis}>
        <p className="text-base text-slate-400">
          {error ?? (report === null ? "Report not found." : "Loading report…")}
        </p>
      </DashboardLayout>
    );
  }

  const byKey = Object.fromEntries(report.measurements.map((m) => [m.criterion_key, m]));
  const highlightIndices = selectedCriterion
    ? (byKey[selectedCriterion]?.landmark_indices ?? [])
    : null;

  return (
    <DashboardLayout actions={newAnalysis}>
      {!report.model_trained && (
        <div className="mb-6 rounded-xl border-l-4 border-red-500 bg-red-50 p-5">
          <p className="text-base font-semibold text-red-800">False results.</p>
          <p className="mt-1 text-sm text-red-700">
            This report was produced without a trained model.
          </p>
        </div>
      )}
      {report.warnings.length > 0 && report.model_trained && (
        <div className="mb-6 rounded-xl border-l-4 border-amber-500 bg-amber-50 p-5">
          {report.warnings.map((w, i) => (
            <p key={i} className="text-sm text-amber-800">
              {w}
            </p>
          ))}
        </div>
      )}

      <div className="grid gap-8 lg:grid-cols-2">
        {/* Left: image + landmarks + info */}
        <div className="space-y-8">
          <div className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-2.5 text-base font-semibold text-slate-700">
                <ImageIcon className="h-5 w-5 text-betta-600" />
                Analyzed Image
              </div>
              <OverlayToggles
                showKeypoints={showKeypoints}
                setShowKeypoints={setShowKeypoints}
                showHeatmap={showHeatmap}
                setShowHeatmap={setShowHeatmap}
              />
            </div>

            <ImageStage
              imageUrl={report.imageUrl}
              alt="Analyzed betta"
              report={report}
              highlightIndices={highlightIndices}
              showKeypoints={showKeypoints}
              setShowKeypoints={setShowKeypoints}
              showHeatmap={showHeatmap}
              setShowHeatmap={setShowHeatmap}
            />

            <div className="mt-6 flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-slate-400">
              {Object.entries(GROUP_LABELS).map(([group, label]) => (
                <div key={group} className="flex items-center gap-1.5">
                  <span
                    className="h-2 w-2 shrink-0 rounded-full"
                    style={{ backgroundColor: colorForGroup(group) }}
                  />
                  {label}
                </div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-5 rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
            <div>
              <p className="text-sm text-slate-400">Fish type</p>
              <p className="mt-1 text-base font-medium text-slate-700">
                {report.fish_class}
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Image ID</p>
              <p className="mt-1 truncate text-base font-medium text-slate-700">
                {report.image_id}
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Analysis Date</p>
              <p className="mt-1 text-base font-medium text-slate-700">
                {formatDate(report.analysis_date)}
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Model</p>
              <p className="mt-1 text-base font-medium text-slate-700">
                {report.model_name}
              </p>
            </div>
          </div>
        </div>

        {/* Right: measurements with computed results */}
        <div className="flex flex-col self-start rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
          <div className="mb-5 flex items-center justify-between gap-2.5">
            <div className="flex items-center gap-2.5 text-base font-semibold text-slate-700">
              <RulerIcon className="h-5 w-5 text-betta-600" />
              Morphometric Measurements
            </div>
            <span className="text-xs text-slate-400">
              {selectedCriterion
                ? "Click again to clear"
                : "Click a row to highlight its landmarks"}
            </span>
          </div>

          <div className="flex-1 space-y-3.5">
            {CRITERIA.map(({ key, label, icon: Icon }) => {
              const m = byKey[key];
              if (!m) return null;
              const selected = selectedCriterion === key;
              const { icon: StatusIcon, badge } =
                statusStyles[m.decision] ?? statusStyles["Defer to Judge"];
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => {
                    setSelectedCriterion(selected ? null : key);
                    if (!selected) setShowKeypoints(true);
                  }}
                  className={`flex w-full items-center gap-4 rounded-xl px-5 py-4 text-left transition ${
                    selected
                      ? "bg-betta-50 ring-2 ring-betta-400"
                      : "bg-slate-50 ring-1 ring-transparent hover:bg-betta-50"
                  }`}
                >
                  <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-white text-betta-600 ring-1 ring-slate-200">
                    <Icon className="h-5 w-5" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-base font-medium text-slate-700">{label}</p>
                    <p className="mt-0.5 text-sm text-slate-500">
                      <span className="font-semibold text-slate-600">
                        {formatValue(key, m.value, m.uncertainty)}
                      </span>{" "}
                      <span className="text-slate-400">· {m.decision}</span>
                    </p>
                  </div>
                  <div
                    className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-full ${badge}`}
                  >
                    <StatusIcon className="h-5 w-5" />
                  </div>
                </button>
              );
            })}
          </div>

          <div className="mt-5 flex flex-wrap items-center justify-between gap-4 border-t border-slate-100 pt-5">
            <div className="flex items-center gap-5 text-sm text-slate-500">
              <span className="flex items-center gap-2">
                <CheckIcon className="h-5 w-5 text-emerald-600" />
                Pass
              </span>
              <span className="flex items-center gap-2">
                <WarningIcon className="h-5 w-5 text-amber-600" />
                Defer
              </span>
              <span className="flex items-center gap-2">
                <AlertCircleIcon className="h-5 w-5 text-red-600" />
                Fault
              </span>
            </div>

            <div className="flex items-center gap-3.5">
              <span className="text-base font-medium text-slate-500">Export</span>
              <div className="flex items-center gap-2.5">
                <span
                  title="PDF export not implemented yet - use CSV"
                  className="flex h-11 w-11 cursor-not-allowed items-center justify-center rounded-lg bg-slate-100 text-slate-300"
                  aria-label="Export as PDF (unavailable)"
                >
                  <FilePdfIcon className="h-5 w-5" />
                </span>
                <button
                  type="button"
                  onClick={() => downloadCsv(report)}
                  className="flex h-11 w-11 items-center justify-center rounded-lg bg-emerald-100 text-emerald-600 transition hover:bg-emerald-200"
                  aria-label="Export as CSV"
                >
                  <FileCsvIcon className="h-5 w-5" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}

export default ReportView;
