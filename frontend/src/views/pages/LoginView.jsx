import { Link, useNavigate } from "react-router-dom";
import AuthLayout from "../../components/AuthLayout.jsx";
import { TextField, PasswordField } from "../../components/AuthInputs.jsx";

function LoginView() {
  const navigate = useNavigate();

  function handleSubmit(e) {
    e.preventDefault();
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
          required
        />

        <div>
          <PasswordField autoComplete="current-password" required />
          <div className="mt-2 text-right">
            <a
              href="#"
              className="text-sm font-medium text-betta-600 hover:text-betta-700"
            >
              Forgot Password?
            </a>
          </div>
        </div>

        <button
          type="submit"
          className="w-full rounded-full bg-gradient-to-r from-betta-500 to-betta-600 py-3.5 text-base font-semibold text-white shadow-glow transition hover:opacity-90"
        >
          Sign in
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
