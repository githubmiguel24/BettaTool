import { useState } from "react";

function EyeIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" {...props}>
      <path
        d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
      <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.8" />
    </svg>
  );
}

function EyeOffIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" {...props}>
      <path
        d="M3 3l18 18M10.6 10.6a3 3 0 004.2 4.2M9.9 5.1A10.6 10.6 0 0112 5c6.5 0 10 7 10 7a13.7 13.7 0 01-3.1 3.9M6.5 6.6C4 8.3 2 12 2 12a13.9 13.9 0 004 4.9"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

const inputClasses =
  "w-full rounded-lg border border-slate-200 bg-slate-50 px-5 py-3.5 text-base text-slate-800 placeholder:text-slate-400 focus:border-betta-400 focus:bg-white focus:outline-none focus:ring-2 focus:ring-betta-400/30";

export function TextField({ label, ...props }) {
  return (
    <label className="block">
      <span className="mb-2 block text-base font-medium text-slate-700">
        {label}
      </span>
      <input className={inputClasses} {...props} />
    </label>
  );
}

export function PasswordField({ label = "Password", ...props }) {
  const [visible, setVisible] = useState(false);

  return (
    <label className="block">
      <span className="mb-2 block text-base font-medium text-slate-700">
        {label}
      </span>
      <div className="relative">
        <input
          type={visible ? "text" : "password"}
          placeholder="••••••••••"
          className={`${inputClasses} pr-12`}
          {...props}
        />
        <button
          type="button"
          onClick={() => setVisible((v) => !v)}
          className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 transition hover:text-slate-600"
          aria-label={visible ? "Hide password" : "Show password"}
        >
          {visible ? (
            <EyeOffIcon className="h-5 w-5" />
          ) : (
            <EyeIcon className="h-5 w-5" />
          )}
        </button>
      </div>
    </label>
  );
}
