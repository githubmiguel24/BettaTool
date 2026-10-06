import { useRef, useState } from "react";
import DashboardLayout from "../../components/DashboardLayout.jsx";
import ConfirmDialog from "../../components/ConfirmDialog.jsx";
import ExportPdfButton from "../../components/ExportPdfButton.jsx";
import KeypointSummary from "../../components/KeypointSummary.jsx";
import CompareTab from "../../components/CompareTab.jsx";
import ImageStage, { OverlayToggles } from "../../components/ImageStage.jsx";
import { measurements as CRITERIA } from "../../data/measurements.js";
import { analyzeImage } from "../../api/client.js";
import { deleteReports, saveReport } from "../../api/reports.js";
import { useAuth } from "../../context/AuthContext.jsx";
import { decisionKind, formatValue } from "../../lib/format.js";
import { GROUP_LABELS, colorForGroup } from "../../data/keypointGroups.js";
import {
  PlusIcon,
  ImageIcon,
  UploadIcon,
  RulerIcon,
  TrashIcon,
  FileCsvIcon,
} from "../../components/Icons.jsx";

const DECISION_STYLES = {
  Pass: "bg-emerald-100 text-emerald-700",
  Fault: "bg-red-100 text-red-700",
  Defer: "bg-amber-100 text-amber-700",
};

function UploadView() {
  const { user } = useAuth();
  const [saveState, setSaveState] = useState(null); // null | "saved" | error message
  const fileInputRef = useRef(null);
  const captureRef = useRef(null);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [image, setImage] = useState(null); // { url, name, uploadedAt, file }
  const [isDragging, setIsDragging] = useState(false);
  const [report, setReport] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [error, setError] = useState(null);
  const [showKeypoints, setShowKeypoints] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [selectedCriterion, setSelectedCriterion] = useState(null);
  const [tab, setTab] = useState("analysis"); // "analysis" | "compare"

  async function loadFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    setImage({
      url: URL.createObjectURL(file),
      name: file.name,
      uploadedAt: new Date(),
      file,
    });
    setReport(null);
    setError(null);
    setStatus("loading");
    setSelectedCriterion(null);
    setSaveState(null);
    setTab("analysis");

    try {
      const result = await analyzeImage(file);
      setReport(result);
      setStatus("done");
      try {
        await saveReport(file, result, user.id);
        setSaveState("saved");
      } catch (saveErr) {
        setSaveState(`Not saved to history: ${saveErr.message}`);
      }
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  function handleDrop(e) {
    e.preventDefault();
    setIsDragging(false);
    loadFile(e.dataTransfer.files?.[0]);
  }

  function resetUpload() {
    setImage(null);
    setReport(null);
    setError(null);
    setStatus("idle");
    setSelectedCriterion(null);
    setSaveState(null);
    setTab("analysis");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  async function handleDelete() {
    setDeleting(true);
    try {
      await deleteReports([report.id]);
      resetUpload();
    } catch (err) {
      setSaveState(`Could not delete: ${err.message}`);
    }
    setConfirmingDelete(false);
    setDeleting(false);
  }

  const byKey = Object.fromEntries(
    (report?.measurements ?? []).map((m) => [m.criterion_key, m]),
  );
  const highlightIndices = selectedCriterion
    ? (byKey[selectedCriterion]?.landmark_indices ?? [])
    : null;

  return (
    <DashboardLayout
      actions={
        <div className="flex items-center gap-3">
          {saveState === "saved" && (
            <button
              type="button"
              onClick={() => setConfirmingDelete(true)}
              className="flex items-center gap-2 rounded-full bg-red-50 px-5 py-3 text-base font-medium text-red-600 transition hover:bg-red-100"
            >
              <TrashIcon className="h-5 w-5" />
              Delete
            </button>
          )}
          <button
            onClick={resetUpload}
            className="flex items-center gap-2 rounded-full bg-betta-950 px-6 py-3 text-base font-semibold text-white shadow-glow transition hover:bg-betta-900"
          >
            New Analysis
            <PlusIcon className="h-5 w-5" />
          </button>
        </div>
      }
    >
      {/* Integrity banner: an untrained backend must never be mistaken for a result. */}
      {report && !report.model_trained && (
        <div className="mb-6 rounded-xl border-l-4 border-red-500 bg-red-50 p-5">
          <p className="text-base font-semibold text-red-800">
            False results. 
          </p>
          <p className="mt-1 text-sm text-red-700">
            No Trained Model Found.
          </p>
        </div>
      )}

      {saveState && saveState !== "saved" && (
        <div className="mb-6 rounded-xl border-l-4 border-red-500 bg-red-50 p-5">
          <p className="text-sm text-red-700">{saveState}</p>
        </div>
      )}

      {report?.warnings?.length > 0 && report.model_trained && (
        <div className="mb-6 rounded-xl border-l-4 border-amber-500 bg-amber-50 p-5">
          {report.warnings.map((w, i) => (
            <p key={i} className="text-sm text-amber-800">
              {w}
            </p>
          ))}
        </div>
      )}

      {report && (
        <div className="mb-6 flex gap-2 border-b border-slate-200" role="tablist" data-html2canvas-ignore>
          {[
            ["analysis", "Analysis"],
            ["compare", "Compare with MFLD-Net"],
          ].map(([key, label]) => (
            <button
              key={key}
              type="button"
              role="tab"
              aria-selected={tab === key}
              onClick={() => setTab(key)}
              className={`-mb-px border-b-2 px-5 py-3 text-base font-medium transition ${
                tab === key
                  ? "border-betta-600 text-betta-700"
                  : "border-transparent text-slate-400 hover:text-slate-600"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      )}

      {report && image && (
        <div className={tab === "compare" ? "" : "hidden"}>
          <CompareTab file={image.file} imageUrl={image.url} active={tab === "compare"} />
        </div>
      )}

      <div ref={captureRef} className={`grid gap-8 lg:grid-cols-2 ${tab === "compare" ? "hidden" : ""}`}>
        {/* Left: image upload + overlay */}
        <div className="space-y-8">
          <div className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-2.5 text-base font-semibold text-slate-700">
                <ImageIcon className="h-5 w-5 text-betta-600" />
                Analyze Image
              </div>
              {report && (
                <div data-html2canvas-ignore>
                  <OverlayToggles
                    showKeypoints={showKeypoints}
                    setShowKeypoints={setShowKeypoints}
                    showHeatmap={showHeatmap}
                    setShowHeatmap={setShowHeatmap}
                  />
                </div>
              )}
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => loadFile(e.target.files?.[0])}
            />

            {image ? (
              <ImageStage
                imageUrl={image.url}
                alt="Uploaded betta"
                report={report}
                loading={status === "loading"}
                highlightIndices={highlightIndices}
                criterionKey={selectedCriterion}
                showKeypoints={showKeypoints}
                setShowKeypoints={setShowKeypoints}
                showHeatmap={showHeatmap}
                setShowHeatmap={setShowHeatmap}
              />
            ) : (
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                className={`relative flex aspect-[4/3] w-full flex-col items-center justify-center overflow-hidden rounded-xl border-2 border-dashed transition ${
                  isDragging
                    ? "border-betta-500 bg-betta-50"
                    : "border-slate-300 bg-slate-50 hover:border-betta-400 hover:bg-betta-50/50"
                }`}
              >
                <UploadIcon className="h-9 w-9 text-slate-400" />
                <span className="mt-3 text-base font-medium text-slate-400">
                  Upload Image
                </span>
                <span className="mt-1 text-sm text-slate-400">
                  Click or drag a photo here
                </span>
              </button>
            )}

            {error && (
              <p className="mt-4 rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700">
                {error}
              </p>
            )}

            {/* Landmark legend, driven by the API response rather than a
                hardcoded list that can drift from the model's schema. */}
            <KeypointSummary
              keypoints={report?.keypoints}
              imageUrl={image?.url}
              width={report?.image_width}
              height={report?.image_height}
            />
          </div>

          <div className="grid grid-cols-2 gap-5 rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
            <div>
              <p className="text-sm text-slate-400">Fish type</p>
              <p className="mt-1 text-base font-medium text-slate-700">
                {report?.fish_class ?? "Halfmoon Longfin"}
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Image ID</p>
              <p className="mt-1 truncate text-base font-medium text-slate-700">
                {image ? image.name : "—"}
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Analysis Date</p>
              <p className="mt-1 text-base font-medium text-slate-700">
                {image
                  ? image.uploadedAt.toLocaleDateString("en-US", {
                      month: "long",
                      day: "numeric",
                      year: "numeric",
                    })
                  : "—"}
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Model</p>
              <p className="mt-1 text-base font-medium text-slate-700">
                {report?.model_name ?? "HRNet-W32"}
              </p>
            </div>
          </div>
        </div>

        {/* Right: measurements */}
        <div className="flex flex-col rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
          <div className="mb-5 flex items-center justify-between gap-2.5">
            <div className="flex items-center gap-2.5 text-base font-semibold text-slate-700">
              <RulerIcon className="h-5 w-5 text-betta-600" />
              Morphometric Measurements
            </div>
            {report && (
              <span data-html2canvas-ignore className="text-xs text-slate-400">
                {selectedCriterion
                  ? "Click again to clear"
                  : "Click a row to highlight its landmarks"}
              </span>
            )}
          </div>

          <div className="flex-1 space-y-3.5">
            {CRITERIA.map(({ key, label, description, icon: Icon }) => {
              const m = byKey[key];
              const selected = selectedCriterion === key;
              return (
                <button
                  key={key}
                  type="button"
                  disabled={!m}
                  onClick={() => {
                    setSelectedCriterion(selected ? null : key);
                    if (!selected) setShowKeypoints(true);
                  }}
                  className={`flex w-full items-center gap-4 rounded-xl px-5 py-4 text-left transition ${
                    selected
                      ? "bg-betta-50 ring-2 ring-betta-400"
                      : "bg-slate-50 ring-1 ring-transparent hover:bg-betta-50"
                  } ${m ? "cursor-pointer" : "cursor-default"}`}
                >
                  <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-white text-betta-600 ring-1 ring-slate-200">
                    <Icon className="h-5 w-5" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-base font-medium text-slate-700">
                      {m?.label ?? label}
                    </p>
                    <p className="mt-0.5 text-sm text-slate-400">
                      {m
                        ? `${formatValue(key, m.value, m.uncertainty)} · TSI ${m.tsi.toFixed(2)}px · RMSE ${m.rmse.toFixed(2)}px`
                        : description}
                    </p>
                  </div>
                  <span
                    className={`shrink-0 rounded-full px-3.5 py-1.5 text-sm font-medium ${
                      m
                        ? (DECISION_STYLES[decisionKind(m.decision)] ??
                          "bg-slate-200/70 text-slate-500")
                        : "bg-slate-200/70 text-slate-500"
                    }`}
                  >
                    {status === "loading"
                      ? "Running…"
                      : (m?.decision ?? "Pending")}
                  </span>
                </button>
              );
            })}
          </div>

          <p className="mt-5 text-center text-sm text-slate-400">
            {status === "done"
              ? `Report ${report.id.slice(0, 8)} · ${report.keypoints.length} landmarks localized${saveState === "saved" ? " · saved to history" : ""}`
              : status === "loading"
                ? "Running the perception → analytical → decisional pipeline…"
                : "Upload a photo to run the full measurement suite."}
          </p>

          <div
            data-html2canvas-ignore
            className="mt-5 flex items-center justify-between border-t border-slate-100 pt-5"
          >
            <span className="text-base font-medium text-slate-500">Export</span>
            <div
              className={`flex items-center gap-2.5 transition ${
                report ? "" : "pointer-events-none opacity-40"
              }`}
            >
              {/* PDF is a client-side screenshot of the results (see
                  lib/exportPdf.js); the backend's PDF exporter is unused. */}
              <ExportPdfButton
                targetRef={captureRef}
                filename={`report-${report?.id.slice(0, 8) ?? "analysis"}.pdf`}
                disabled={!report}
              />
              <a
                href={
                  report
                    ? `/reports/${report.id}/export?format=csv`
                    : undefined
                }
                download
                className="flex h-11 w-11 items-center justify-center rounded-lg bg-emerald-100 text-emerald-600 transition hover:bg-emerald-200"
                aria-label="Export as CSV"
              >
                <FileCsvIcon className="h-5 w-5" />
              </a>
            </div>
          </div>
        </div>
      </div>

      <ConfirmDialog
        open={confirmingDelete}
        title="Delete this analysis?"
        message="This permanently deletes the image and its results."
        busy={deleting}
        onConfirm={handleDelete}
        onCancel={() => setConfirmingDelete(false)}
      />
    </DashboardLayout>
  );
}

export default UploadView;
