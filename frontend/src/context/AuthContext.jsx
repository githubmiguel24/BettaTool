import { createContext, useContext, useEffect, useState } from "react";
import { supabase } from "../lib/supabase.js";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  // undefined = still restoring the session from storage
  const [session, setSession] = useState(undefined);

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => setSession(data.session));
    const { data } = supabase.auth.onAuthStateChange((_event, next) =>
      setSession(next),
    );
    return () => data.subscription.unsubscribe();
  }, []);

  const user = session?.user ?? null;
  const value = {
    user,
    loading: session === undefined,
    displayName: user?.user_metadata?.full_name || user?.email || "",
    email: user?.email ?? "",
    signIn: (email, password) =>
      supabase.auth.signInWithPassword({ email, password }),
    signUp: (fullName, email, password) =>
      supabase.auth.signUp({
        email,
        password,
        options: { data: { full_name: fullName } },
      }),
    signOut: () => supabase.auth.signOut(),
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
