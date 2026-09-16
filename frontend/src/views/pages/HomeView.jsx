import { Link } from "react-router-dom";
import bettaLogo from "../../assets/betta-tool-cutout.png";

function BoltIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" {...props}>
      <path
        d="M13 2 3 14h7l-1 8 10-12h-7l1-8z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

function TrendIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" {...props}>
      <path
        d="M3 17l6-6 4 4 8-8M21 7h-6v6"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ShieldIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" {...props}>
      <path
        d="M12 3l7 3v6c0 4.5-3 8-7 9-4-1-7-4.5-7-9V6l7-3z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
      <path
        d="M9 12l2 2 4-4"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

const features = [
  {
    icon: BoltIcon,
    title: "Instant AI Analysis",
    description:
      "Upload a photo and get a morphometric assessment back in seconds.",
  },
  {
    icon: TrendIcon,
    title: "Track Over Time",
    description:
      "Log every assessment to a per-fish history so you can the past assessment.",
  },
  {
    icon: ShieldIcon,
    title: "Consistent, Objective Scoring",
    description:
      "The same model checks every fish the same way, so results stay consistent across breeders and shows.",
  },
];

function HomeView() {
  return (
    <div className="min-h-screen bg-white font-sans">
      {/* Navbar */}
      <header className="fixed top-0 z-20 w-full border-b border-betta-950/10 bg-white/50 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5 lg:px-10">
          <div className="flex items-center gap-3">
            <img
              src={bettaLogo}
              alt="Betta fish"
              className="animate-swim h-12 w-12 object-contain sm:h-14 sm:w-14"
            />
            <span className="font-display text-2xl font-bold text-betta-950 sm:text-3xl">
              Betta<span className="text-betta-500">Tool</span>
            </span>
          </div>
          <Link
            to="/login"
            className="rounded-full border-2 border-betta-950/15 px-7 py-3 text-base font-semibold text-betta-950 transition hover:border-betta-950/30 hover:bg-betta-950/5"
          >
            Login
          </Link>
        </div>
      </header>

      {/* Hero */}
      <section className="relative flex flex-col items-center overflow-hidden bg-betta-hero px-6 pb-24 pt-40 lg:px-10 lg:pb-32 lg:pt-48">
        <div className="pointer-events-none absolute -left-24 top-20 h-72 w-72 animate-blob rounded-full bg-betta-300/12 blur-3xl" />
        <div className="pointer-events-none absolute -right-16 top-1/3 h-80 w-80 animate-blob animation-delay-2000 rounded-full bg-white/50 blur-3xl" />
        <div className="pointer-events-none absolute bottom-0 left-1/3 h-64 w-64 animate-blob animation-delay-4000 rounded-full bg-betta-400/10 blur-3xl" />

        <div className="relative z-10 flex w-full max-w-7xl flex-1 flex-col items-center justify-center gap-14 lg:flex-row lg:gap-16">
          <div className="flex flex-col items-center text-center lg:items-start lg:text-left">
            <h1 className="font-display text-5xl font-extrabold leading-tight text-betta-950 sm:text-6xl lg:text-7xl">
              Betta Fish{" "}
              <span className="bg-gradient-to-r from-betta-600 via-betta-500 to-betta-700 bg-clip-text text-transparent">
                Quality Assessment
              </span>
            </h1>

            <p className="mt-6 max-w-xl text-lg text-betta-900 sm:text-xl">
              Upload a photo of your betta and get an instant, AI-backed read
              on morphometric measurements — no guesswork
              required.
            </p>

            <div className="mt-10 flex flex-col items-center gap-5 sm:flex-row">
              <Link
                to="/login"
                className="rounded-full bg-gradient-to-r from-betta-500 to-betta-600 px-10 py-4 text-base font-semibold text-white shadow-glow transition hover:scale-105 hover:shadow-xl"
              >
                Get Started
              </Link>
              <a
                href="#features"
                className="text-base font-medium text-betta-900 underline underline-offset-4 transition hover:text-betta-950"
              >
                See how it works
              </a>
            </div>
          </div>

          <div className="relative shrink-0">
            <div className="pointer-events-none absolute inset-0 -z-10 rounded-full bg-betta-300/30 blur-3xl" />
            <img
              src={bettaLogo}
              alt="Blue halfmoon betta fish flaring its fins"
              className="animate-float h-72 w-72 object-contain drop-shadow-[0_0_20px_rgba(63,195,202,0.35)] sm:h-96 sm:w-96 lg:h-[26rem] lg:w-[26rem] xl:h-[30rem] xl:w-[30rem]"
            />
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="bg-white px-6 py-24 lg:px-10">
        <div className="mx-auto max-w-7xl">
          <div className="mx-auto max-w-xl text-center">
            <h2 className="font-display text-3xl font-bold text-betta-950">
              Everything you need to keep your betta thriving
            </h2>
            <p className="mt-3 text-slate-500">
              Built for hobbyists and breeders who want a quick, reliable
              second opinion on fish condition.
            </p>
          </div>

          <div className="mt-14 grid gap-6 md:grid-cols-3">
            {features.map(({ icon: Icon, title, description }) => (
              <div
                key={title}
                className="rounded-2xl border border-slate-100 p-7 shadow-sm transition hover:-translate-y-1 hover:shadow-lg"
              >
                <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-betta-500/10 text-betta-600">
                  <Icon className="h-5 w-5" />
                </div>
                <h3 className="mt-5 font-display text-lg font-semibold text-betta-950">
                  {title}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-slate-500">
                  {description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/10 bg-betta-950 py-8 text-center text-sm text-white/50">
        © {new Date().getFullYear()} BettaTool. Built for the betta community.
      </footer>
    </div>
  );
}

export default HomeView;
