import { useState } from "react";
import { FilePdfIcon } from "./Icons.jsx";
import { exportElementToPdf } from "../lib/exportPdf.js";

/** Screenshots `targetRef`'s element into a PDF named `filename`. */
function ExportPdfButton({ targetRef, filename, disabled = false }) {
  const [busy, setBusy] = useState(false);

  async function handleClick() {
    if (!targetRef.current) return;
    setBusy(true);
    try {
      await exportElementToPdf(targetRef.current, {
        filename,
        title: "Betta Quality Assessment Report",
      });
    } catch (err) {
      window.alert(`PDF export failed: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={disabled || busy}
      title={busy ? "Creating PDF…" : "Export as PDF"}
      aria-label="Export as PDF"
      className="flex h-11 w-11 items-center justify-center rounded-lg bg-red-100 text-red-600 transition hover:bg-red-200 disabled:cursor-not-allowed disabled:opacity-60"
    >
      {busy ? (
        <span className="h-5 w-5 animate-spin rounded-full border-2 border-red-300 border-t-red-600" />
      ) : (
        <FilePdfIcon className="h-5 w-5" />
      )}
    </button>
  );
}

export default ExportPdfButton;
