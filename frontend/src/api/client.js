/**
 * Backend API client.
 *
 * Base URL resolution order:
 *   1. VITE_API_BASE (set in frontend/.env.local) - use this to point at a
 *      backend on another host, e.g. a Kaggle/Colab tunnel.
 *   2. "" (empty) - relative URLs, which Vite's dev proxy forwards to
 *      http://localhost:8000 (see vite.config.js).
 */
const API_BASE = import.meta.env.VITE_API_BASE ?? "";

/** Thrown for any non-2xx response, carrying the backend's `detail` string. */
export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function parseError(response) {
  let detail = `Request failed (HTTP ${response.status})`;
  try {
    const body = await response.json();
    if (body?.detail) {
      detail =
        typeof body.detail === "string"
          ? body.detail
          : JSON.stringify(body.detail);
    }
  } catch {
    // Non-JSON error body (e.g. a proxy's HTML 502 page) - keep the default.
  }
  return new ApiError(detail, response.status);
}

/**
 * POSTs one image to /analyze and returns the AssessmentReport.
 *
 * The caller MUST check `report.model_trained` before presenting any value
 * in the response as a measurement: when it is false the backend is running
 * randomly initialized weights and every number is arbitrary.
 */
export async function analyzeImage(file, { signal } = {}) {
  const form = new FormData();
  form.append("file", file);

  let response;
  try {
    response = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      body: form,
      signal,
    });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError(
      "Could not reach the backend. Is it running on http://localhost:8000? " +
        "Start it with: uvicorn app.main:app --reload",
      0,
    );
  }

  if (!response.ok) throw await parseError(response);
  return response.json();
}

export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
