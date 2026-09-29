import { Link, useNavigate } from "react-router-dom";
import DashboardLayout from "../../components/DashboardLayout.jsx";
import ConfirmDialog from "../../components/ConfirmDialog.jsx";
import SelectionBar from "../../components/SelectionBar.jsx";
import { useAuth } from "../../context/AuthContext.jsx";
import { summarize } from "../../api/reports.js";
import { deleteMessage, useReportList } from "../../lib/useReportList.js";
import { formatDate } from "../../lib/format.js";
import {
  UserIcon,
  PlusIcon,
  ArrowRightIcon,
  ClockIcon,
  ChartIcon,
  CheckIcon,
  WarningIcon,
  AlertCircleIcon,
} from "../../components/Icons.jsx";

const toneClasses = {
  sky: "bg-sky-100 text-sky-600",
  emerald: "bg-emerald-100 text-emerald-600",
  amber: "bg-amber-100 text-amber-600",
  red: "bg-red-100 text-red-600",
};

function DashboardView() {
  const navigate = useNavigate();
  const { displayName, email, signOut } = useAuth();
  const {
    reports,
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

  async function handleSignOut() {
    await signOut();
    navigate("/");
  }

  // Pass/Defer/Fault count individual measurements (6 per image), not images.
  const counts = summarize(reports ?? []);
  const stats = [
    { label: "Total Analyses", note: "images", value: counts.total, icon: ChartIcon, tone: "sky" },
    { label: "Pass", note: "measurements", value: counts.Pass, icon: CheckIcon, tone: "emerald" },
    { label: "Defer", note: "measurements", value: counts.Defer, icon: WarningIcon, tone: "amber" },
    { label: "Fault", note: "measurements", value: counts.Fault, icon: AlertCircleIcon, tone: "red" },
  ];
  const recentAnalyses = (reports ?? []).slice(0, 3);
  const recentIds = recentAnalyses.map((r) => r.id);

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
      <div className="space-y-8">
        <div className="flex flex-wrap items-center justify-between gap-4 rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
          <div className="flex items-center gap-4">
            <div className="flex h-14 w-14 items-center justify-center rounded-full bg-betta-950 text-white">
              <UserIcon className="h-6 w-6" />
            </div>
            <div>
              <p className="text-lg font-semibold text-slate-800">
                {displayName}
              </p>
              <p className="text-base text-slate-400">{email}</p>
            </div>
          </div>
          <button
            type="button"
            onClick={handleSignOut}
            className="flex items-center gap-2 text-base text-slate-400 transition hover:text-slate-600"
          >
            Sign out
            <ArrowRightIcon className="h-5 w-5" />
          </button>
        </div>

        <div className="grid grid-cols-2 gap-5 lg:grid-cols-4">
          {stats.map(({ label, note, value, icon: Icon, tone }) => (
            <div
              key={label}
              className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100"
            >
              <div
                className={`mb-4 flex h-11 w-11 items-center justify-center rounded-lg ${toneClasses[tone]}`}
              >
                <Icon className="h-6 w-6" />
              </div>
              <p className="text-3xl font-bold text-slate-800">{value}</p>
              <p className="text-base text-slate-400">{label}</p>
              <p className="text-sm text-slate-300">{note}</p>
            </div>
          ))}
        </div>

        <div className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-slate-800">
              Recent Analyses
            </h2>
            <Link
              to="/history"
              className="text-base font-medium text-betta-600 transition hover:text-betta-700"
            >
              view all »
            </Link>
          </div>
          {error && (
            <p className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700">
              {error}
            </p>
          )}
          {reports && recentAnalyses.length === 0 && (
            <p className="py-4 text-base text-slate-400">
              No analyses yet. Upload a photo to get started.
            </p>
          )}
          {recentAnalyses.length > 0 && (
            <SelectionBar
              visibleIds={recentIds}
              selected={selected}
              onToggleAll={toggleAll}
              onDelete={() =>
                requestDelete(recentIds.filter((id) => selected.has(id)))
              }
            />
          )}
          <div className="divide-y divide-slate-100">
            {recentAnalyses.map(({ id, imageId, analysisDate }) => (
              <div
                key={id}
                className="flex flex-wrap items-center justify-between gap-3 py-4"
              >
                <div className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    checked={selected.has(id)}
                    onChange={() => toggle(id)}
                    aria-label={`Select ${imageId}`}
                    className="h-5 w-5 rounded border-slate-300 accent-betta-600"
                  />
                  <span className="text-base font-medium text-slate-700">
                    {imageId}
                  </span>
                </div>
                <div className="flex items-center gap-7">
                  <span className="flex items-center gap-2 text-base text-slate-400">
                    <ClockIcon className="h-5 w-5" />
                    {formatDate(analysisDate)}
                  </span>
                  <Link
                    to={`/report/${id}`}
                    className="text-base font-medium text-betta-600 transition hover:text-betta-700"
                  >
                    view
                  </Link>
                  <button
                    type="button"
                    onClick={() => requestDelete([id])}
                    className="text-base font-medium text-red-500 transition hover:text-red-600"
                  >
                    delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
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

export default DashboardView;
