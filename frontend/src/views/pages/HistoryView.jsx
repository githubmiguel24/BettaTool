import { Link } from "react-router-dom";
import DashboardLayout from "../../components/DashboardLayout.jsx";
import bettaPhoto from "../../assets/betta-hero.png";
import { PlusIcon } from "../../components/Icons.jsx";

const statusClasses = {
  Pass: "text-emerald-600",
  Defer: "text-amber-600",
  Fault: "text-red-600",
};

const history = [
  { id: "IMG_2026_0001", date: "May 30, 2026", status: "Pass" },
  { id: "IMG_2026_0002", date: "May 30, 2026", status: "Defer" },
  { id: "IMG_2026_0003", date: "May 30, 2026", status: "Fault" },
  { id: "IMG_2026_0004", date: "May 27, 2026", status: "Pass" },
  { id: "IMG_2026_0005", date: "May 25, 2026", status: "Pass" },
  { id: "IMG_2026_0006", date: "May 22, 2026", status: "Defer" },
  { id: "IMG_2026_0007", date: "May 19, 2026", status: "Pass" },
  { id: "IMG_2026_0008", date: "May 14, 2026", status: "Fault" },
  { id: "IMG_2026_0009", date: "May 10, 2026", status: "Pass" },
];

function HistoryView() {
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
      <div className="grid gap-5 sm:grid-cols-2">
        {history.map(({ id, date, status }) => (
          <Link
            key={id}
            to={`/report/${id}`}
            className="flex items-center gap-5 rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-100 transition hover:-translate-y-0.5 hover:shadow-md"
          >
            <img
              src={bettaPhoto}
              alt={id}
              className="h-24 w-24 shrink-0 rounded-xl object-cover"
            />
            <div className="min-w-0">
              <p className="truncate text-lg font-semibold text-slate-800">
                {id}
              </p>
              <p className="mt-1.5 text-base text-slate-400">
                {date} ·{" "}
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
