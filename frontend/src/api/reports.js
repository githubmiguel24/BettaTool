import { supabase } from "../lib/supabase.js";
import { overallStatus } from "../lib/format.js";

const BUCKET = "betta-images";

function fail(error) {
  if (error) throw new Error(error.message);
}

/** Persists an AssessmentReport from the backend plus its source image. */
export async function saveReport(file, report, userId) {
  const ext = (file.name.split(".").pop() || "jpg").toLowerCase();
  const imagePath = `${userId}/${report.id}.${ext}`;

  const upload = await supabase.storage
    .from(BUCKET)
    .upload(imagePath, file, { contentType: file.type });
  fail(upload.error);

  const inserted = await supabase.from("reports").insert({
    id: report.id,
    user_id: userId,
    image_id: report.image_id,
    image_path: imagePath,
    fish_class: report.fish_class,
    analysis_date: new Date().toISOString(),
    model_name: report.model_name,
    model_trained: report.model_trained,
    image_width: report.image_width,
    image_height: report.image_height,
  });
  if (inserted.error) {
    await supabase.storage.from(BUCKET).remove([imagePath]);
    fail(inserted.error);
  }

  const children = await Promise.all([
    supabase.from("measurements").insert(
      report.measurements.map((m) => ({
        report_id: report.id,
        criterion_key: m.criterion_key,
        label: m.label,
        value: m.value,
        uncertainty: m.uncertainty,
        tsi: m.tsi,
        rmse: m.rmse,
        decision: m.decision,
        landmark_indices: m.landmark_indices,
      })),
    ),
    supabase.from("keypoints").insert(
      report.keypoints.map((k) => ({
        report_id: report.id,
        index: k.index,
        name: k.name,
        label: k.label,
        group: k.group,
        x: k.x,
        y: k.y,
        sigma_x: k.sigma_x,
        sigma_y: k.sigma_y,
        rho: k.rho,
        visibility: k.visibility,
        low_visibility: k.low_visibility,
      })),
    ),
    report.warnings.length
      ? supabase
          .from("warnings")
          .insert(report.warnings.map((message) => ({ report_id: report.id, message })))
      : Promise.resolve({ error: null }),
  ]);

  const childError = children.find((r) => r.error)?.error;
  if (childError) {
    // ON DELETE CASCADE clears any child rows that did get written.
    await supabase.from("reports").delete().eq("id", report.id);
    await supabase.storage.from(BUCKET).remove([imagePath]);
    fail(childError);
  }
}

async function signedUrlMap(paths) {
  const valid = paths.filter(Boolean);
  if (!valid.length) return {};
  const { data, error } = await supabase.storage
    .from(BUCKET)
    .createSignedUrls(valid, 3600);
  fail(error);
  return Object.fromEntries(data.map((d) => [d.path, d.signedUrl]));
}

/** Newest first. Each item carries its overall Pass/Defer/Fault status. */
export async function listReports() {
  const { data, error } = await supabase
    .from("reports")
    .select("id, image_id, image_path, analysis_date, model_trained, measurements(decision)")
    .order("analysis_date", { ascending: false });
  fail(error);

  const urls = await signedUrlMap(data.map((r) => r.image_path));
  return data.map((r) => ({
    id: r.id,
    imageId: r.image_id,
    analysisDate: r.analysis_date,
    modelTrained: r.model_trained,
    status: overallStatus(r.measurements.map((m) => m.decision)),
    thumbnailUrl: urls[r.image_path] ?? null,
  }));
}

export function summarize(reports) {
  const count = (s) => reports.filter((r) => r.status === s).length;
  return {
    total: reports.length,
    Pass: count("Pass"),
    Defer: count("Defer"),
    Fault: count("Fault"),
  };
}

/** Returns a report shaped like the backend's AssessmentReport, plus imageUrl. */
export async function getReport(id) {
  const { data, error } = await supabase
    .from("reports")
    .select("*, measurements(*), keypoints(*), warnings(message)")
    .eq("id", id)
    .maybeSingle();
  fail(error);
  if (!data) return null;

  const urls = await signedUrlMap([data.image_path]);
  return {
    id: data.id,
    image_id: data.image_id,
    fish_class: data.fish_class,
    analysis_date: data.analysis_date,
    model_name: data.model_name,
    model_trained: data.model_trained,
    image_width: data.image_width,
    image_height: data.image_height,
    imageUrl: urls[data.image_path] ?? null,
    measurements: data.measurements,
    keypoints: [...data.keypoints].sort((a, b) => a.index - b.index),
    warnings: data.warnings.map((w) => w.message),
  };
}
