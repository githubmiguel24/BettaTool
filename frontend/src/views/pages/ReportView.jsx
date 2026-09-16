import { useParams, Link } from "react-router-dom";
import DashboardLayout from "../../components/DashboardLayout.jsx";
import { measurements } from "../../data/measurements.js";
import { landmarkColumns } from "../../data/landmarks.js";
import bettaPhoto from "../../assets/betta-hero.png";
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
  Pass: { icon: CheckIcon, badge: "bg-emerald-100 text-emerald-600" },
  Defer: { icon: WarningIcon, badge: "bg-amber-100 text-amber-600" },
  Fault: { icon: AlertCircleIcon, badge: "bg-red-100 text-red-600" },
};

// mock results for the demo — a real backend would return these per image
const mockResults = [
  { value: "170°", tolerance: "±1.5°", status: "Fault" },
  { value: "0.45", tolerance: "±0.03", status: "Fault" },
  { value: "0.56", tolerance: "±0.02", status: "Pass" },
  { value: "0.49", tolerance: "±0.05", status: "Defer" },
  { value: "1.12", tolerance: "±0.05", status: "Defer" },
  { value: "1.04", tolerance: "±0.04", status: "Pass" },
];

function ReportView() {
  const { id } = useParams();

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
      <div className="grid gap-8 lg:grid-cols-2">
        {/* Left: image + landmarks + info */}
        <div className="space-y-8">
          <div className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
            <div className="mb-4 flex items-center gap-2.5 text-base font-semibold text-slate-700">
              <ImageIcon className="h-5 w-5 text-betta-600" />
              Analyze Image
            </div>

            <div className="flex aspect-[4/3] w-full items-center justify-center overflow-hidden rounded-xl border border-slate-200 bg-slate-50">
              <img
                src={bettaPhoto}
                alt="Analyzed betta"
                className="h-full w-full object-contain"
              />
            </div>

            <div className="mt-6 grid grid-cols-2 gap-x-5 gap-y-2.5 text-sm text-slate-500">
              {landmarkColumns.map((column, i) => (
                <ul key={i} className="space-y-2.5">
                  {column.map(({ label, tone }) => (
                    <li key={label} className="flex items-center gap-2.5">
                      <span
                        className={`h-2 w-2 shrink-0 rounded-full ${tone}`}
                      />
                      {label}
                    </li>
                  ))}
                </ul>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-5 rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
            <div>
              <p className="text-sm text-slate-400">Fish type</p>
              <p className="mt-1 text-base font-medium text-slate-700">
                Halfmoon Lengthii
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Image ID</p>
              <p className="mt-1 truncate text-base font-medium text-slate-700">
                {id}
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Analysis Date</p>
              <p className="mt-1 text-base font-medium text-slate-700">
                May 30, 2026
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Model</p>
              <p className="mt-1 text-base font-medium text-slate-700">
                HB Net
              </p>
            </div>
          </div>
        </div>

        {/* Right: measurements with computed results */}
        <div className="flex flex-col self-start rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
          <div className="mb-5 flex items-center gap-2.5 text-base font-semibold text-slate-700">
            <RulerIcon className="h-5 w-5 text-betta-600" />
            Morphometric Measurements
          </div>

          <div className="flex-1 space-y-3.5">
            {measurements.map(({ key, label, icon: Icon }, i) => {
              const result = mockResults[i];
              const { icon: StatusIcon, badge } = statusStyles[result.status];
              return (
                <div
                  key={key}
                  className="flex items-center gap-4 rounded-xl bg-slate-50 px-5 py-4"
                >
                  <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-white text-betta-600 ring-1 ring-slate-200">
                    <Icon className="h-5 w-5" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-base font-medium text-slate-700">
                      {label}
                    </p>
                    <p className="mt-0.5 text-sm">
                      <span className="font-semibold text-slate-600">
                        {result.value}
                      </span>{" "}
                      <span className="text-slate-400">
                        {result.tolerance}
                      </span>
                    </p>
                  </div>
                  <div
                    className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-full ${badge}`}
                  >
                    <StatusIcon className="h-5 w-5" />
                  </div>
                </div>
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
              <span className="text-base font-medium text-slate-500">
                Export
              </span>
              <div className="flex items-center gap-2.5">
                <button
                  type="button"
                  className="flex h-11 w-11 items-center justify-center rounded-lg bg-red-100 text-red-600 transition hover:bg-red-200"
                  aria-label="Export as PDF"
                >
                  <FilePdfIcon className="h-5 w-5" />
                </button>
                <button
                  type="button"
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
