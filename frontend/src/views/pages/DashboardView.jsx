import { Link } from "react-router-dom";
import DashboardLayout from "../../components/DashboardLayout.jsx";
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

const stats = [
  { label: "Total Analyses", value: 48, icon: ChartIcon, tone: "sky" },
  { label: "Pass", value: 31, icon: CheckIcon, tone: "emerald" },
  { label: "Defer", value: 12, icon: WarningIcon, tone: "amber" },
  { label: "Fault", value: 5, icon: AlertCircleIcon, tone: "red" },
];

const toneClasses = {
  sky: "bg-sky-100 text-sky-600",
  emerald: "bg-emerald-100 text-emerald-600",
  amber: "bg-amber-100 text-amber-600",
  red: "bg-red-100 text-red-600",
};

const recentAnalyses = [
  { id: "IMG_2026_0001", date: "May 30, 2026", status: "Pass" },
  { id: "IMG_2026_0002", date: "May 30, 2026", status: "Defer" },
  { id: "IMG_2026_0003", date: "May 30, 2026", status: "Fault" },
];

const statusClasses = {
  Pass: { dot: "bg-emerald-500", text: "text-emerald-600" },
  Defer: { dot: "bg-amber-500", text: "text-amber-600" },
  Fault: { dot: "bg-red-500", text: "text-red-600" },
};

function DashboardView() {
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
                Lebron James
              </p>
              <p className="text-base text-slate-400">goat@email.com</p>
            </div>
          </div>
          <Link
            to="/"
            className="flex items-center gap-2 text-base text-slate-400 transition hover:text-slate-600"
          >
            Sign out
            <ArrowRightIcon className="h-5 w-5" />
          </Link>
        </div>

        <div className="grid grid-cols-2 gap-5 lg:grid-cols-4">
          {stats.map(({ label, value, icon: Icon, tone }) => (
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
          <div className="divide-y divide-slate-100">
            {recentAnalyses.map(({ id, date, status }) => (
              <div
                key={id}
                className="flex flex-wrap items-center justify-between gap-3 py-4"
              >
                <div className="flex items-center gap-3">
                  <span
                    className={`h-2.5 w-2.5 rounded-full ${statusClasses[status].dot}`}
                  />
                  <span className="text-base font-medium text-slate-700">
                    {id}
                  </span>
                </div>
                <div className="flex items-center gap-7">
                  <span className="flex items-center gap-2 text-base text-slate-400">
                    <ClockIcon className="h-5 w-5" />
                    {date}
                  </span>
                  <span
                    className={`text-base font-medium ${statusClasses[status].text}`}
                  >
                    {status}
                  </span>
                  <Link
                    to={`/report/${id}`}
                    className="text-base font-medium text-betta-600 transition hover:text-betta-700"
                  >
                    view
                  </Link>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}

export default DashboardView;
