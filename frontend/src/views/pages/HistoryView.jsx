import { Link } from "react-router-dom";
import DashboardLayout from "../../components/DashboardLayout.jsx";
import ConfirmDialog from "../../components/ConfirmDialog.jsx";
import SelectionBar from "../../components/SelectionBar.jsx";
import bettaPhoto from "../../assets/betta-hero.png";
import { PlusIcon, TrashIcon } from "../../components/Icons.jsx";
import { deleteMessage, useReportList } from "../../lib/useReportList.js";
import { formatDate } from "../../lib/format.js";

function HistoryView() {
  const {
    reports: history,
    error,
    selected,
    toggle,
    toggleAll,
    pending,
    deleting,
    requestDelete,
    cancelDelete,
    confirmDelete,
  } = useReportList();

  const ids = history?.map((r) => r.id) ?? [];

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

      {history?.length > 0 && (
        <SelectionBar
          visibleIds={ids}
          selected={selected}
          onToggleAll={toggleAll}
          onDelete={() => requestDelete(ids.filter((id) => selected.has(id)))}
        />
      )}

      <div className="grid gap-5 sm:grid-cols-2">
        {history?.map(({ id, imageId, analysisDate, thumbnailUrl }) => (
          <div
            key={id}
            className={`flex items-center gap-4 rounded-2xl bg-white p-5 shadow-sm transition hover:-translate-y-0.5 hover:shadow-md ${
              selected.has(id) ? "ring-2 ring-betta-400" : "ring-1 ring-slate-100"
            }`}
          >
            <input
              type="checkbox"
              checked={selected.has(id)}
              onChange={() => toggle(id)}
              aria-label={`Select ${imageId}`}
              className="h-5 w-5 shrink-0 rounded border-slate-300 accent-betta-600"
            />
            <Link to={`/report/${id}`} className="flex min-w-0 flex-1 items-center gap-5">
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
                  {formatDate(analysisDate)}
                </p>
              </div>
            </Link>
            <button
              type="button"
              onClick={() => requestDelete([id])}
              aria-label={`Delete ${imageId}`}
              title="Delete"
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg text-slate-400 transition hover:bg-red-50 hover:text-red-600"
            >
              <TrashIcon className="h-5 w-5" />
            </button>
          </div>
        ))}
      </div>

      <ConfirmDialog
        open={pending !== null}
        title={pending?.length === 1 ? "Delete this analysis?" : `Delete ${pending?.length} analyses?`}
        message={deleteMessage(pending?.length ?? 0)}
        busy={deleting}
        onConfirm={confirmDelete}
        onCancel={cancelDelete}
      />
    </DashboardLayout>
  );
}

export default HistoryView;
