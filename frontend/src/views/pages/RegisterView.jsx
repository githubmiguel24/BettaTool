import { useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import AuthLayout from "../../components/AuthLayout.jsx";
import { TextField, PasswordField } from "../../components/AuthInputs.jsx";
import { useAuth } from "../../context/AuthContext.jsx";

function RegisterView() {
  const navigate = useNavigate();
  const { user, signUp } = useAuth();
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  if (user) return <Navigate to="/dashboard" replace />;

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setNotice(null);
    if (password.length < 6) {
      setError("Password must be at least 6 characters.");
      return;
    }
    if (password !== confirm) {
      setError("Passwords do not match.");
      return;
    }

    setSubmitting(true);
    const { data, error: authError } = await signUp(fullName, email, password);
    setSubmitting(false);
    if (authError) {
      setError(authError.message);
      return;
    }
    // With email confirmation enabled in Supabase there is no session yet.
    if (data.session) navigate("/dashboard");
    else setNotice("Account created. Check your email to confirm it, then sign in.");
  }

  return (
    <AuthLayout mode="signup">
      <h1 className="font-display text-3xl font-bold text-betta-950">
        Create an account
      </h1>
      <p className="mt-2 text-base text-slate-500">Join us!</p>

      <form onSubmit={handleSubmit} className="mt-8 space-y-6">
        <TextField
          label="Full Name"
          type="text"
          placeholder="Your name"
          autoComplete="name"
          value={fullName}
          onChange={(e) => setFullName(e.target.value)}
          required
        />
        <TextField
          label="Email Address"
          type="email"
          placeholder="you@email.com"
          autoComplete="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <PasswordField
          autoComplete="new-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <PasswordField
          label="Confirm Password"
          autoComplete="new-password"
          value={confirm}
          onChange={(e) => setConfirm(e.target.value)}
          required
        />

        {error && (
          <p className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </p>
        )}
        {notice && (
          <p className="rounded-lg bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
            {notice}
          </p>
        )}

        <button
          type="submit"
          disabled={submitting}
          className="w-full rounded-full bg-gradient-to-r from-betta-500 to-betta-600 py-3.5 text-base font-semibold text-white shadow-glow transition hover:opacity-90 disabled:opacity-60"
        >
          {submitting ? "Creating account…" : "Create Account"}
        </button>
      </form>

      <p className="mt-8 text-center text-base text-slate-500">
        Already have an account?{" "}
        <Link
          to="/login"
          className="font-semibold text-betta-600 hover:text-betta-700"
        >
          Login
        </Link>
      </p>
    </AuthLayout>
  );
}

export default RegisterView;
