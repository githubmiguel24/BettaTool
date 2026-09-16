import { Link, useLocation } from "react-router-dom";
import bettaLogo from "../assets/betta-tool-cutout.png";
import { UserIcon, UploadIcon, ClockIcon, LogoutIcon } from "./Icons.jsx";

const navItems = [
  { to: "/dashboard", label: "Profile", icon: UserIcon },
  { to: "/upload", label: "Upload", icon: UploadIcon },
  { to: "/history", label: "History", icon: ClockIcon },
];

function DashboardLayout({
  title = "Betta Quality Assessment Tool",
  actions,
  children,
}) {
  const { pathname } = useLocation();

  return (
    <div className="flex h-screen overflow-hidden bg-slate-100">
      <aside className="flex h-screen w-64 shrink-0 flex-col overflow-y-auto bg-betta-hero px-5 py-8">
        <div className="flex items-center gap-2.5 px-1">
          <img
            src={bettaLogo}
            alt="Betta fish"
            className="animate-swim h-11 w-11 object-contain"
          />
          <span className="font-display text-lg font-bold text-betta-950">
            Betta<span className="text-betta-500">Tool</span>
          </span>
        </div>

        <nav className="mt-12 flex flex-1 flex-col gap-2">
          {navItems.map(({ to, label, icon: Icon }) => {
            const active = pathname === to;
            return (
              <Link
                key={to}
                to={to}
                className={`flex items-center gap-3.5 rounded-xl px-4 py-3.5 text-base font-medium transition ${
                  active
                    ? "bg-betta-950 text-white shadow-lg"
                    : "text-betta-900/80 hover:bg-betta-950/5 hover:text-betta-950"
                }`}
              >
                <Icon className="h-5 w-5" />
                {label}
              </Link>
            );
          })}
        </nav>

        <div className="border-t border-betta-950/10 pt-4">
          <div className="flex items-center gap-3 rounded-xl px-3 py-2.5">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-betta-950 text-white">
              <UserIcon className="h-5 w-5" />
            </div>
            <div className="min-w-0">
              <p className="truncate text-base font-semibold text-betta-950">
                Lebron James
              </p>
              <p className="truncate text-sm text-betta-900/60">
                goat@email.com
              </p>
            </div>
          </div>
          <Link
            to="/"
            className="mt-2 flex items-center gap-3.5 rounded-xl px-4 py-3 text-base font-medium text-betta-900/70 transition hover:bg-betta-950/5 hover:text-betta-950"
          >
            <LogoutIcon className="h-5 w-5" />
            Sign out
          </Link>
        </div>
      </aside>

      <div className="flex-1 overflow-y-auto px-8 py-8 lg:px-10">
        <header className="mb-7 flex flex-wrap items-center justify-between gap-4">
          <h1 className="font-display text-2xl font-bold text-betta-950">
            {title}
          </h1>
          {actions}
        </header>
        {children}
      </div>
    </div>
  );
}

export default DashboardLayout;
