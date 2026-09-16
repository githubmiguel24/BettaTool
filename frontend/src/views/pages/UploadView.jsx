import { useRef, useState } from "react";
import DashboardLayout from "../../components/DashboardLayout.jsx";
import KeypointOverlay from "../../components/KeypointOverlay.jsx";
import { measurements as CRITERIA } from "../../data/measurements.js";
import { analyzeImage } from "../../api/client.js";
import {
  PlusIcon,
  ImageIcon,
  UploadIcon,
  RulerIcon,
  FilePdfIcon,
  FileCsvIcon,
} from "../../components/Icons.jsx";

const DECISION_STYLES = {
  "Confident Pass": "bg-emerald-100 text-emerald-700",
  "Confident Fault": "bg-red-100 text-red-700",
  "Defer to Judge": "bg-amber-100 text-amber-700",
};

/** Angles read in degrees; every other criterion is a dimensionless ratio. */
function formatValue(criterionKey, value, uncertainty) {
  const isAngle = criterionKey === "caudal-spread-angle";
  const digits = isAngle ? 1 : 3;
  const unit = isAngle ? "°" : "";
  return `${value.toFixed(digits)}${unit} ± ${uncertainty.toFixed(digits)}${unit}`;
}

function UploadView() {
  const fileInputRef = useRef(null);
  const [image, setImage] = useState(null); // { url, name, uploadedAt, file }
  const [isDragging, setIsDragging] = useState(false);
  const [report, setReport] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [error, setError] = useState(null);
  const [showOverlay, setShowOverlay] = useState(true);

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

    try {
      const result = await analyzeImage(file);
      setReport(result);
      setStatus("done");
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
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  const byKey = Object.fromEntries(
    (report?.measurements ?? []).map((m) => [m.criterion_key, m]),
  );

  return (
    <DashboardLayout
      actions={
        <button
          onClick={resetUpload}
          className="flex items-center gap-2 rounded-full bg-betta-950 px-6 py-3 text-base font-semibold text-white shadow-glow transition hover:bg-betta-900"
        >
          New Analysis
          <PlusIcon className="h-5 w-5" />
        </button>
      }
    >
      {/* Integrity banner: an untrained backend must never be mistaken for a result. */}
      {report && !report.model_trained && (
        <div className="mb-6 rounded-xl border-l-4 border-red-500 bg-red-50 p-5">
          <p className="text-base font-semibold text-red-800">
            Untrained model &mdash; these numbers are not measurements
          </p>
          <p className="mt-1 text-sm text-red-700">
            The backend found no trained checkpoint and is running randomly
            initialized weights. This confirms the integration path works
            end&#8209;to&#8209;end; the values below are arbitrary and must not be
            reported as results.
          </p>
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

      <div className="grid gap-8 lg:grid-cols-2">
        {/* Left: image upload + overlay */}
        <div className="space-y-8">
          <div className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-2.5 text-base font-semibold text-slate-700">
                <ImageIcon className="h-5 w-5 text-betta-600" />
                Analyze Image
              </div>
              {report && (
                <label className="flex cursor-pointer items-center gap-2 text-sm text-slate-500">
                  <input
                    type="checkbox"
                    checked={showOverlay}
                    onChange={(e) => setShowOverlay(e.target.checked)}
                    className="h-4 w-4 rounded border-slate-300"
                  />
                  Landmarks
                </label>
              )}
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => loadFile(e.target.files?.[0])}
            />

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
              {image ? (
                <>
                  <img
                    src={image.url}
                    alt="Uploaded betta"
                    className="h-full w-full object-contain"
                  />
                  {showOverlay && report && <KeypointOverlay report={report} />}
                  {status === "loading" && (
                    <div className="absolute inset-0 flex items-center justify-center bg-white/70">
                      <span className="text-base font-medium text-betta-700">
                        Running pipeline&hellip;
                      </span>
                    </div>
                  )}
                </>
              ) : (
                <>
                  <UploadIcon className="h-9 w-9 text-slate-400" />
                  <span className="mt-3 text-base font-medium text-slate-400">
                    Upload Image
                  </span>
                  <span className="mt-1 text-sm text-slate-400">
                    Click or drag a photo here
                  </span>
                </>
              )}
            </button>

            {error && (
              <p className="mt-4 rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700">
                {error}
              </p>
            )}

            {/* Landmark legend, driven by the API response rather than a
                hardcoded list that can drift from the model's schema. */}
            {report?.keypoints?.length > 0 && (
              <div className="mt-6 grid grid-cols-2 gap-x-5 gap-y-2 text-sm text-slate-500">
                {report.keypoints.map((kp) => (
                  <div key={kp.index} className="flex items-center gap-2.5">
                    <span
                      className={`h-2 w-2 shrink-0 rounded-full ${
                        kp.low_visibility ? "bg-slate-400" : "bg-red-500"
                      }`}
                    />
                    <span className="truncate">{kp.label}</span>
                    <span className="ml-auto shrink-0 tabular-nums text-xs text-slate-400">
                      &plusmn;{Math.max(kp.sigma_x, kp.sigma_y).toFixed(1)}px
                    </span>
                  </div>
                ))}
              </div>
            )}
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
          <div className="mb-5 flex items-center gap-2.5 text-base font-semibold text-slate-700">
            <RulerIcon className="h-5 w-5 text-betta-600" />
            Morphometric Measurements
          </div>

          <div className="flex-1 space-y-3.5">
            {CRITERIA.map(({ key, label, description, icon: Icon }) => {
              const m = byKey[key];
              return (
                <div
                  key={key}
                  className="flex items-center gap-4 rounded-xl bg-slate-50 px-5 py-4 transition hover:bg-betta-50"
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
                        ? (DECISION_STYLES[m.decision] ??
                          "bg-slate-200/70 text-slate-500")
                        : "bg-slate-200/70 text-slate-500"
                    }`}
                  >
                    {status === "loading"
                      ? "Running…"
                      : (m?.decision ?? "Pending")}
                  </span>
                </div>
              );
            })}
          </div>

          <p className="mt-5 text-center text-sm text-slate-400">
            {status === "done"
              ? `Report ${report.id.slice(0, 8)} · ${report.keypoints.length} landmarks localized`
              : status === "loading"
                ? "Running the perception → analytical → decisional pipeline…"
                : "Upload a photo to run the full measurement suite."}
          </p>

          <div className="mt-5 flex items-center justify-between border-t border-slate-100 pt-5">
            <span className="text-base font-medium text-slate-500">Export</span>
            <div
              className={`flex items-center gap-2.5 transition ${
                report ? "" : "pointer-events-none opacity-40"
              }`}
            >
              {/* PDF export is not implemented on the backend yet
                  (app/reports/exporters.py raises NotImplementedError, which
                  the route surfaces as a 501). Rendered disabled rather than
                  linked, so a demo click cannot produce an error toast. */}
              <span
                title="PDF export not implemented yet - use CSV"
                className="flex h-11 w-11 cursor-not-allowed items-center justify-center rounded-lg bg-slate-100 text-slate-300"
                aria-label="Export as PDF (unavailable)"
              >
                <FilePdfIcon className="h-5 w-5" />
              </span>
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
    </DashboardLayout>
  );
}

export default UploadView;
