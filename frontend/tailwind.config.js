/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["Sora", "sans-serif"],
        sans: ["Inter", "sans-serif"],
      },
      colors: {
        betta: {
          950: "#0a2540",
          900: "#0f3a5f",
          800: "#155a82",
          700: "#1a7aa8",
          600: "#2196c9",
          500: "#3aa8e0",
          400: "#6bc4ec",
          300: "#9ad9f4",
          200: "#c8ecfa",
        },
      },
      backgroundImage: {
        "betta-hero":
          "linear-gradient(135deg, #ffffff 0%, #eaf6fe 30%, #cfe9fb 60%, #ade0fa 100%)",
      },
      boxShadow: {
        glow: "0 0 60px -10px rgba(58, 168, 224, 0.45)",
      },
    },
  },
  plugins: [],
};
