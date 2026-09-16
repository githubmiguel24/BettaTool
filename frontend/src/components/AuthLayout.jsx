import { Link } from "react-router-dom";
import bettaLogo from "../assets/betta-tool-cutout.png";

function AuthLayout({ mode, children }) {
  return (
    <div className="flex min-h-screen flex-col bg-white lg:flex-row">
      <aside className="relative flex flex-col items-center justify-between overflow-hidden bg-betta-hero px-8 py-10 text-center lg:w-[38%] lg:py-16">
        <div className="pointer-events-none absolute -left-16 top-10 h-56 w-56 animate-blob rounded-full bg-betta-300/30 blur-3xl" />
        <div className="pointer-events-none absolute -right-10 bottom-10 h-56 w-56 animate-blob animation-delay-2000 rounded-full bg-betta-400/20 blur-3xl" />

        <div />

        <div className="relative z-10 flex flex-col items-center">
          <div className="relative flex h-56 w-56 items-center justify-center rounded-full border border-betta-950/15">
            <img
              src={bettaLogo}
              alt="Betta fish"
              className="animate-float h-48 w-48 object-contain"
            />
          </div>
          <h2 className="mt-10 font-display text-3xl font-bold leading-snug text-betta-950 lg:text-4xl">
            Assess your
            <br />
            Betta Fish now!
          </h2>
        </div>

        <Link
          to="/"
          className="relative z-10 text-base text-betta-900/70 transition hover:text-betta-950"
        >
          ← Go back to homepage
        </Link>
      </aside>

      <main className="flex flex-1 items-center justify-center bg-slate-50 px-6 py-16 lg:px-16">
        <div className="w-full max-w-lg">
          <div className="mb-8 flex rounded-full bg-slate-200/70 p-1.5">
            <Link
              to="/login"
              className={`flex-1 rounded-full py-3 text-center text-base font-semibold transition ${
                mode === "signin"
                  ? "bg-gradient-to-r from-betta-500 to-betta-600 text-white shadow"
                  : "text-slate-500 hover:text-slate-700"
              }`}
            >
              Sign in
            </Link>
            <Link
              to="/register"
              className={`flex-1 rounded-full py-3 text-center text-base font-semibold transition ${
                mode === "signup"
                  ? "bg-gradient-to-r from-betta-500 to-betta-600 text-white shadow"
                  : "text-slate-500 hover:text-slate-700"
              }`}
            >
              Sign up
            </Link>
          </div>

          <div className="rounded-2xl bg-white p-10 shadow-xl shadow-slate-200/60 ring-1 ring-slate-100">
            {children}
          </div>
        </div>
      </main>
    </div>
  );
}

export default AuthLayout;
