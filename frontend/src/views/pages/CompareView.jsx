import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import DashboardLayout from "../../components/DashboardLayout.jsx";
import CompareTab, { RemoteCompareTab } from "../../components/CompareTab.jsx";
import bettaPhoto from "../../assets/betta-hero.png";
import { PlusIcon, UploadIcon } from "../../components/Icons.jsx";
import { useReportList } from "../../lib/useReportList.js";
import { formatDate } from "../../lib/format.js";

/**
 * Sidebar page: compare our model with MFLD-Net on any photo, either one from
 * History or a fresh upload (which is compared only, not saved).
 */
function CompareView() {
  const { reports, error } = useReportList();
  const [params, setParams] = useSearchParams();
  const [upload, setUpload] = useState(null); // { file, url }
  const selectedId = params.get("report");
  const selected = reports?.find((r) => r.id === selectedId) ?? null;

  function pickFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    setUpload({ file, url: URL.createObjectURL(file) });
    setParams({});
  }

  return (
    <DashboardLayout
      title="Compare with MFLD-Net"
      actions={
        <Link
          to="/upload"
          className="flex items-center gap-2 rounded-full bg-betta-950 px-6 py-3 text-base font-semibold text-white shadow-glow transition hover:bg-betta-900"
        >
          New Analysis
          <PlusIcon className="h-5 w-5" />
        </Link>
      }
    >
      {error && <p className="mb-5 rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700">{error}</p>}

      <section className="mb-8 rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-base font-semibold text-slate-700">Choose a photo</h2>
          <label className="flex cursor-pointer items-center gap-2 rounded-full bg-betta-50 px-4 py-2 text-sm font-medium text-betta-700 transition hover:bg-betta-100">
            <UploadIcon className="h-4 w-4" />
            Upload a new photo
            <input type="file" accept="image/*" className="hidden" onChange={(e) => pickFile(e.target.files?.[0])} />
          </label>
        </div>
        {!reports && !error && <p className="text-sm text-slate-400">Loading history…</p>}
        {reports?.length === 0 && <p className="text-sm text-slate-400">No saved analyses yet. Upload a photo to compare.</p>}
        <div className="flex gap-3 overflow-x-auto pb-1">
          {reports?.map((r) => (
            <button
              key={r.id}
              type="button"
              onClick={() => {
                setUpload(null);
                setParams({ report: r.id });
              }}
              className={`w-36 shrink-0 rounded-xl p-2 text-left transition hover:bg-betta-50 ${
                r.id === selectedId ? "bg-betta-50 ring-2 ring-betta-400" : "ring-1 ring-slate-100"
              }`}
            >
              <img src={r.thumbnailUrl ?? bettaPhoto} alt={r.imageId} className="h-24 w-full rounded-lg object-cover" />
              <p className="mt-2 truncate text-sm font-medium text-slate-700">{r.imageId}</p>
              <p className="truncate text-xs text-slate-400">{formatDate(r.analysisDate)}</p>
            </button>
          ))}
        </div>
      </section>

      {upload && <CompareTab file={upload.file} imageUrl={upload.url} active />}
      {!upload && selected && <RemoteCompareTab key={selected.id} imageUrl={selected.thumbnailUrl} name={selected.imageId} />}
      {!upload && !selected && (
        <p className="text-base text-slate-400">Pick a photo above to see both models' keypoints side by side.</p>
      )}
    </DashboardLayout>
  );
}

export default CompareView;
