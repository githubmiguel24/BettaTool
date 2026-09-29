import { useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import AuthLayout from "../../components/AuthLayout.jsx";
import { TextField, PasswordField } from "../../components/AuthInputs.jsx";
import { useAuth } from "../../context/AuthContext.jsx";

function LoginView() {
  const navigate = useNavigate();
  const { user, signIn } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  if (user) return <Navigate to="/dashboard" replace />;

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    const { error: authError } = await signIn(email, password);
    setSubmitting(false);
    if (authError) {
      setError(authError.message);
      return;
    }
    navigate("/dashboard");
  }

  return (
    <AuthLayout mode="signin">
      <h1 className="font-display text-3xl font-bold text-betta-950">
        Welcome Back!
      </h1>
      <p className="mt-2 text-base text-slate-500">
        Sign in to continue to your account
      </p>

      <form onSubmit={handleSubmit} className="mt-8 space-y-6">
        <TextField
          label="Email Address"
          type="email"
          placeholder="you@email.com"
          autoComplete="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />

        <div>
          <PasswordField
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <div className="mt-2 text-right">
            <a
              href="#"
              className="text-sm font-medium text-betta-600 hover:text-betta-700"
            >
              Forgot Password?
            </a>
          </div>
        </div>

        {error && (
          <p className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={submitting}
          className="w-full rounded-full bg-gradient-to-r from-betta-500 to-betta-600 py-3.5 text-base font-semibold text-white shadow-glow transition hover:opacity-90 disabled:opacity-60"
        >
          {submitting ? "Signing in…" : "Sign in"}
        </button>
      </form>

      <p className="mt-8 text-center text-base text-slate-500">
        Do not have an account?{" "}
        <Link
          to="/register"
          className="font-semibold text-betta-600 hover:text-betta-700"
        >
          Sign up
        </Link>
      </p>
    </AuthLayout>
  );
}

export default LoginView;
