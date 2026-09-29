import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import DashboardLayout from "../../components/DashboardLayout.jsx";
import bettaPhoto from "../../assets/betta-hero.png";
import { PlusIcon } from "../../components/Icons.jsx";
import { listReports } from "../../api/reports.js";
import { formatDate } from "../../lib/format.js";

const statusClasses = {
  Pass: "text-emerald-600",
  Defer: "text-amber-600",
  Fault: "text-red-600",
};

function HistoryView() {
  const [history, setHistory] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    listReports().then(setHistory).catch((e) => setError(e.message));
  }, []);

  return (
    <DashboardLayout
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
      {error && (
        <p className="mb-5 rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </p>
      )}
      {!history && !error && (
        <p className="text-base text-slate-400">Loading history…</p>
      )}
      {history?.length === 0 && (
        <p className="text-base text-slate-400">
          No analyses yet. Upload a photo to create your first report.
        </p>
      )}

      <div className="grid gap-5 sm:grid-cols-2">
        {history?.map(({ id, imageId, analysisDate, status, thumbnailUrl }) => (
          <Link
            key={id}
            to={`/report/${id}`}
            className="flex items-center gap-5 rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-100 transition hover:-translate-y-0.5 hover:shadow-md"
          >
            <img
              src={thumbnailUrl ?? bettaPhoto}
              alt={imageId}
              className="h-24 w-24 shrink-0 rounded-xl object-cover"
            />
            <div className="min-w-0">
              <p className="truncate text-lg font-semibold text-slate-800">
                {imageId}
              </p>
              <p className="mt-1.5 text-base text-slate-400">
                {formatDate(analysisDate)} ·{" "}
                <span className={`font-medium ${statusClasses[status]}`}>
                  {status}
                </span>
              </p>
            </div>
          </Link>
        ))}
      </div>
    </DashboardLayout>
  );
}

export default HistoryView;
