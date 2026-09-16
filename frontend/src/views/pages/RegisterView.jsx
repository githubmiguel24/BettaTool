import { Link, useNavigate } from "react-router-dom";
import AuthLayout from "../../components/AuthLayout.jsx";
import { TextField, PasswordField } from "../../components/AuthInputs.jsx";

function RegisterView() {
  const navigate = useNavigate();

  function handleSubmit(e) {
    e.preventDefault();
    navigate("/dashboard");
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
          required
        />
        <TextField
          label="Email Address"
          type="email"
          placeholder="you@email.com"
          autoComplete="email"
          required
        />
        <PasswordField autoComplete="new-password" required />
        <PasswordField label="Confirm Password" autoComplete="new-password" required />

        <button
          type="submit"
          className="w-full rounded-full bg-gradient-to-r from-betta-500 to-betta-600 py-3.5 text-base font-semibold text-white shadow-glow transition hover:opacity-90"
        >
          Create Account
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
