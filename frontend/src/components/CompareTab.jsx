import { useEffect, useMemo, useState } from "react";
import { compareWithMfld, exampleImageUrl, getMfldBenchmark } from "../api/client.js";
import ZoomableImage from "./ZoomableImage.jsx";

/**
 * "Compare with MFLD-Net" tab of the analysis page.
 *
 * Part 1 draws both models' keypoints for the uploaded photo and the distance between them. A single
 * photo has no ground truth, so this shows where the models DISAGREE, not which one is right.
 * Part 2 shows the measured, labelled test-set benchmark (written by training/export_mfld_benchmark.py).
 * Every number on this tab comes from the backend; nothing is hardcoded here.
 */

const OURS = "#2a78d6"; // blue circle
const MFLD = "#eb6834"; // orange diamond
const GT = "#16a34a"; // green: the human label
const DISAGREE_PCT = 5;

const fmt = (v, d = 1) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "—");

function useAbortableLoad(load, deps, enabled) {
  const [state, setState] = useState({ status: "idle", data: null, error: null });
  useEffect(() => {
    if (!enabled) return undefined;
    const ctrl = new AbortController();
    setState({ status: "loading", data: null, error: null });
    load(ctrl.signal)
      .then((data) => setState({ status: "done", data, error: null }))
      .catch((err) => {
        if (err.name !== "AbortError") setState({ status: "error", data: null, error: err.message });
      });
    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, enabled]);
  return state;
}

/** Both models' keypoints. Lines are ONLY drawn when hovering over a specific keypoint in the table. */
function OverlayLayers({ r, keypoints, showOurs, showMfld, hovered, setHovered }) {
  return (
    <>
      <defs>
        <marker id="arrow-alert" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M 0 0 L 6 3 L 0 6 z" fill="#dc2626" opacity="0.95" />
        </marker>
        <marker id="arrow-normal" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M 0 0 L 6 3 L 0 6 z" fill="#ffffff" opacity="0.8" />
        </marker>
      </defs>

      {/* Interactive Hover Line: Only shows when a user hovers a specific row */}
      {keypoints.map((k) => (
        hovered === k.index && (
          <line
            key={`l${k.index}`}
            x1={k.mfld_x} y1={k.mfld_y} x2={k.ours_x} y2={k.ours_y}
            stroke={k.distance_pct_body > DISAGREE_PCT ? "#dc2626" : "#ffffff"}
            strokeWidth={r * 0.8}
            strokeDasharray={`${r * 1.5} ${r * 1.5}`}
            opacity={0.95}
            markerEnd={k.distance_pct_body > DISAGREE_PCT ? "url(#arrow-alert)" : "url(#arrow-normal)"}
          />
        )
      ))}

      {showMfld &&
        keypoints.map((k) => (
          <rect
            key={`m${k.index}`}
            x={k.mfld_x - r} y={k.mfld_y - r} width={2 * r} height={2 * r}
            transform={`rotate(45 ${k.mfld_x} ${k.mfld_y})`}
            fill={MFLD} stroke="#ffffff" strokeWidth={r * 0.3}
            opacity={hovered == null || hovered === k.index ? 1 : 0.2}
            onMouseEnter={() => setHovered(k.index)} onMouseLeave={() => setHovered(null)}
            className="transition-opacity duration-200"
          />
        ))}
      {showOurs &&
        keypoints.map((k) => (
          <circle
            key={`o${k.index}`}
            cx={k.ours_x} cy={k.ours_y} r={r * (hovered === k.index ? 1.5 : 1.1)}
            fill={OURS} stroke="#ffffff" strokeWidth={r * 0.3}
            opacity={hovered == null || hovered === k.index ? 1 : 0.2}
            onMouseEnter={() => setHovered(k.index)} onMouseLeave={() => setHovered(null)}
            className="transition-opacity duration-200"
          />
        ))}
    </>
  );
}

function Toggle({ checked, onChange, children, dark = false }) {
  return (
    <label className={`flex cursor-pointer items-center gap-2 text-sm ${dark ? "text-slate-200" : "text-slate-600"}`}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="h-4 w-4 accent-betta-600" />
      {children}
    </label>
  );
}

function LivePanel({ imageUrl, state }) {
  const [showOurs, setShowOurs] = useState(true);
  const [showMfld, setShowMfld] = useState(true);
  const [hovered, setHovered] = useState(null);

  if (state.status === "loading") {
    return <p className="rounded-xl bg-slate-50 px-5 py-8 text-center text-sm text-slate-500">Running both models on this photo…</p>;
  }
  if (state.status === "error") {
    return <p className="rounded-xl bg-red-50 px-5 py-4 text-sm text-red-700">{state.error}</p>;
  }
  const data = state.data;
  if (!data) return null;
  const s = data.summary;
  const toolbar = (dark) => (
    <>
      <Toggle dark={dark} checked={showOurs} onChange={setShowOurs}>
        <span className="inline-block h-3 w-3 rounded-full" style={{ backgroundColor: OURS }} /> BettaTool
      </Toggle>
      <Toggle dark={dark} checked={showMfld} onChange={setShowMfld}>
        <span className="inline-block h-3 w-3 rotate-45" style={{ backgroundColor: MFLD }} /> MFLD-Net
      </Toggle>
    </>
  );
  const sorted = [...data.keypoints].sort((a, b) => b.distance_pct_body - a.distance_pct_body);
  const maxPct = Math.max(...data.keypoints.map((k) => k.distance_pct_body), DISAGREE_PCT * 2);

  return (
    <div className="grid gap-8 lg:grid-cols-2">
      <div>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-x-5 gap-y-2">
          <div className="flex gap-4">{toolbar(false)}</div>
          <span className="text-xs italic text-slate-400">Hover table to see difference lines</span>
        </div>
        <ZoomableImage
          width={data.image_width}
          height={data.image_height}
          imageUrl={imageUrl}
          ariaLabel="Keypoints of both models on the uploaded photo"
          toolbar={toolbar}
        >
          {(r) => (
            <OverlayLayers r={r} keypoints={data.keypoints} showOurs={showOurs} showMfld={showMfld} hovered={hovered} setHovered={setHovered} />
          )}
        </ZoomableImage>
        <p className="mt-3 text-xs text-slate-400">
          MFLD-Net: {data.mfld.note}.
        </p>
      </div>

      <div>
        <div className="grid grid-cols-3 gap-3">
          <div className="rounded-xl bg-slate-50 p-4">
            <p className="text-xs text-slate-400">Mean disagreement</p>
            <p className="mt-1 text-xl font-semibold text-slate-700">{fmt(s.mean_distance_pct_body)}%</p>
            <p className="text-xs text-slate-400">of body length</p>
          </div>
          <div className="rounded-xl bg-slate-50 p-4">
            <p className="text-xs text-slate-400">Keypoints &gt;{s.disagree_threshold_pct}% apart</p>
            <p className="mt-1 text-xl font-semibold text-slate-700">{s.n_disagree} / {data.keypoints.length}</p>
          </div>
          <div className="rounded-xl bg-slate-50 p-4">
            <p className="text-xs text-slate-400">Most different</p>
            <p className="mt-1 text-sm font-medium leading-snug text-slate-700">{s.most_different.join(", ")}</p>
          </div>
        </div>

        <div className="mt-5 max-h-[26rem] overflow-y-auto rounded-xl ring-1 ring-slate-100">
          <table className="w-full text-left text-sm">
            <thead className="sticky top-0 bg-white text-xs text-slate-400">
              <tr><th className="px-3 py-2 font-medium">Hover Keypoint to view gap</th><th className="px-3 py-2 font-medium">Distance between models</th><th className="px-3 py-2 text-right font-medium">px</th></tr>
            </thead>
            <tbody>
              {sorted.map((k) => (
                <tr key={k.index} onMouseEnter={() => setHovered(k.index)} onMouseLeave={() => setHovered(null)} className={`cursor-pointer transition-colors ${hovered === k.index ? "bg-betta-50" : ""}`}>
                  <td className="px-3 py-1.5 font-medium text-slate-700">{k.label}</td>
                  <td className="px-3 py-1.5">
                    <div className="flex items-center gap-2">
                      <div className="h-2 flex-1 rounded-full bg-slate-100">
                        <div className="h-2 rounded-full transition-all" style={{ width: `${Math.min(100, (k.distance_pct_body / maxPct) * 100)}%`, backgroundColor: k.distance_pct_body > DISAGREE_PCT ? "#dc2626" : "#94a3b8" }} />
                      </div>
                      <span className="w-12 text-right tabular-nums text-xs text-slate-500">{fmt(k.distance_pct_body)}%</span>
                    </div>
                  </td>
                  <td className="px-3 py-1.5 text-right tabular-nums text-slate-500">{fmt(k.distance_px, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function Bar({ value, max, color, isPercentage }) {
  const widthPct = isPercentage ? value : Math.max(2, (value / max) * 100);
  return (
    <div className={`h-2.5 rounded-full ${isPercentage ? "bg-slate-200" : ""}`}>
      <div className="h-full rounded-full" style={{ width: `${widthPct}%`, backgroundColor: color }} />
    </div>
  );
}

// Explicitly separated metrics into two clear themes: "Higher is Better" and "Lower is Better"
const METRIC_GROUPS = [
  { 
    title: "Performance Scores", 
    description: "Metrics where a longer bar means better performance (e.g., Accuracy, Precision). Positive = Fault Present, Negative = Pass/No Fault.",
    keys: ["criterion_accuracy", "precision", "recall", "f1_score", "win_rate", "pck_5", "pck_10"],
    isPercentage: true
  },
  { 
    title: "Error Measurements", 
    description: "Metrics measuring pixel deviation or error rates. A shorter bar means a better result.",
    keys: ["rmse_all", "mean_all", "mean_tips", "mean_body", "median_all", "criterion_error"],
    isPercentage: false
  },
];

const valueText = (v, unit) => `${fmt(v, v < 10 ? 2 : 1)}${unit === "px" ? " px" : "%"}`;

/** ONLY used for Higher is Better metrics. */
function MetricBarChart({ r }) {
  // Use deferral data if available; otherwise fallback to the base 'ours'
  const bettaToolValue = r.ours_deferral != null ? r.ours_deferral : r.ours;
  
  const bars = [
    { name: "BettaTool", value: bettaToolValue, color: OURS },
    { name: "MFLD-Net", value: r.mfld, color: MFLD },
  ].filter((b) => typeof b.value === "number");
  
  const max = 100; // Since all "Higher is Better" metrics are percentages

  return (
    <div className="py-4">
      <div className="flex items-start justify-between gap-3">
        <p className="text-sm font-medium text-slate-700">{r.label}</p>
        {r.better && (
          <span className={`shrink-0 rounded-full px-2.5 py-1 text-xs font-medium ${r.better === "ours" ? "bg-emerald-100 text-emerald-700" : "bg-slate-200/70 text-slate-600"}`}>
            {r.better === "ours" ? "BettaTool" : "MFLD-Net"}{r.times_better ? ` ${fmt(r.times_better)}×` : ""}
          </span>
        )}
      </div>
      <div className="mt-2 space-y-1.5">
        {bars.map((b) => (
          <div key={b.name} className="flex items-center gap-3">
            <span className="w-32 shrink-0 text-xs font-medium" style={{ color: b.color }}>{b.name}</span>
            <div className="flex-1"><Bar value={b.value} max={max} color={b.color} isPercentage={true} /></div>
            <span className="w-16 shrink-0 text-right text-sm font-semibold tabular-nums text-slate-700">{valueText(b.value, r.unit)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function BenchmarkPanel({ state }) {
  const rows = state.data?.rows ?? [];
  const perKp = state.data?.per_keypoint ?? [];
  
  if (state.status === "loading") return <p className="text-sm text-slate-500">Loading the benchmark…</p>;
  if (state.status === "error") return <p className="rounded-xl bg-amber-50 px-5 py-4 text-sm text-amber-800">{state.error}</p>;
  if (!state.data) return null;

  // STRICT SEPARATION: Graphs only for Positive (Higher is Better), Tables only for Negative (Lower is Better)
  const performanceKeys = ["criterion_accuracy", "precision", "recall", "f1_score", "win_rate", "pck_5", "pck_10"];
  const errorKeys = ["rmse_all", "mean_all", "mean_tips", "mean_body", "median_all", "criterion_error"];

  const performanceMetrics = performanceKeys.map(key => rows.find(r => r.key === key)).filter(Boolean);
  const errorMetrics = errorKeys.map(key => rows.find(r => r.key === key)).filter(Boolean);

  return (
    <div className="space-y-8">
      {/* THEME 1: HIGHER IS BETTER (BAR CHARTS) */}
      {performanceMetrics.length > 0 && (
        <div className="rounded-lg bg-slate-50/50 p-5 ring-1 ring-slate-100">
          <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wide">Performance Scores (Higher is Better)</h3>
          <p className="mt-1 text-xs text-slate-500 mb-2">Metrics where a longer bar means better performance. Evaluates positive hit rates against ground truth.</p>
          <div className="divide-y divide-slate-100">
            {performanceMetrics.map((r) => <MetricBarChart key={r.key} r={r} />)}
          </div>
        </div>
      )}

      {/* THEME 2: LOWER IS BETTER (COMPARATIVE TABLE - NO BARS) */}
      {errorMetrics.length > 0 && (
        <div className="rounded-lg bg-slate-50/50 p-5 ring-1 ring-slate-100">
          <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wide">Error Measurements (Lower is Better)</h3>
          <p className="mt-1 text-xs text-slate-500 mb-4">Metrics measuring pixel deviation or error rates. Presented in a table to prevent graph misinterpretation.</p>
          
          <div className="overflow-hidden rounded-xl ring-1 ring-slate-200 bg-white">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50 text-xs text-slate-500">
                <tr>
                  <th className="w-1/2 px-4 py-3 font-medium">Error Metric</th>
                  <th className="w-1/6 px-4 py-3 text-right font-medium" style={{ color: OURS }}>BettaTool</th>
                  <th className="w-1/6 px-4 py-3 text-right font-medium" style={{ color: MFLD }}>MFLD-Net</th>
                  <th className="w-1/6 px-4 py-3 text-center font-medium">Winner</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {errorMetrics.map((r) => {
                  const bettaToolVal = r.ours_deferral != null ? r.ours_deferral : r.ours;
                  return (
                    <tr key={r.key} className="hover:bg-slate-50/50">
                      <td className="w-1/2 px-4 py-3 font-medium text-slate-700">{r.label}</td>
                      <td className={`w-1/6 px-4 py-3 text-right tabular-nums ${r.better === "ours" ? "font-bold text-slate-900" : "text-slate-500"}`}>
                        {valueText(bettaToolVal, r.unit)}
                      </td>
                      <td className={`w-1/6 px-4 py-3 text-right tabular-nums ${r.better === "mfld" ? "font-bold text-slate-900" : "text-slate-500"}`}>
                        {valueText(r.mfld, r.unit)}
                      </td>
                      <td className="w-1/6 px-4 py-2 text-center">
                        {r.better && (
                          <span className={`inline-block rounded-full px-2 py-1 text-xs font-medium ${r.better === "ours" ? "bg-emerald-100 text-emerald-700" : "bg-slate-200/70 text-slate-600"}`}>
                            {r.better === "ours" ? "BettaTool" : "MFLD-Net"}
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Keypoint Breakdown Table */}
          {perKp.length > 0 && (
            <div className="mt-6">
              <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Mean Pixel Error Breakdown by Keypoint</h4>
              <div className="overflow-hidden rounded-xl ring-1 ring-slate-200 bg-white">
                <table className="w-full text-left text-sm">
                  <thead className="bg-slate-50 text-xs text-slate-500">
                    <tr>
                      <th className="w-1/2 px-4 py-3 font-medium">Keypoint</th>
                      <th className="w-1/6 px-4 py-3 text-right font-medium" style={{ color: OURS }}>BettaTool</th>
                      <th className="w-1/6 px-4 py-3 text-right font-medium" style={{ color: MFLD }}>MFLD-Net</th>
                      <th className="w-1/6 px-4 py-3 text-center font-medium">Winner</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {perKp.map((k) => {
                      const better = k.ours_px <= k.mfld_px ? "ours" : "mfld";
                      return (
                        <tr key={k.name} className="hover:bg-slate-50/50">
                          <td className="w-1/2 px-4 py-3 font-medium text-slate-700">{k.label}</td>
                          <td className={`w-1/6 px-4 py-3 text-right tabular-nums ${better === "ours" ? "font-bold text-slate-900" : "text-slate-500"}`}>
                            {fmt(k.ours_px)} px
                          </td>
                          <td className={`w-1/6 px-4 py-3 text-right tabular-nums ${better === "mfld" ? "font-bold text-slate-900" : "text-slate-500"}`}>
                            {fmt(k.mfld_px)} px
                          </td>
                          <td className="w-1/6 px-4 py-2 text-center">
                            <span className={`inline-block rounded-full px-2 py-1 text-xs font-medium ${better === "ours" ? "bg-emerald-100 text-emerald-700" : "bg-slate-200/70 text-slate-600"}`}>
                              {better === "ours" ? "BettaTool" : "MFLD-Net"}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/** Labelled held-out examples. Lines are only drawn when hovering the table. */
function ExamplesPanel({ data }) {
  const examples = data?.examples ?? [];
  const [sel, setSel] = useState(0);
  const [hovered, setHovered] = useState(null);
  const [imgFailed, setImgFailed] = useState(false);
  
  useEffect(() => {
    setImgFailed(false);
    setHovered(null);
  }, [sel]);

  if (!examples.length) return null;
  const ex = examples[Math.min(sel, examples.length - 1)];
  const kps = ex.keypoints.filter((k) => k.visible);
  const times = ex.mfld_mean_px / Math.max(ex.ours_mean_px, 1e-6);

  return (
    <div>
      <p className="text-sm text-slate-500">{data.examples_note}</p>
      <div className="mt-3 flex flex-wrap gap-2">
        {examples.map((e, i) => (
          <button
            key={e.image_id}
            type="button"
            onClick={() => setSel(i)}
            className={`rounded-full px-3.5 py-1.5 text-xs font-medium ring-1 transition ${
              i === sel ? "bg-betta-600 text-white ring-betta-600" : "bg-white text-slate-600 ring-slate-200 hover:bg-betta-50"
            }`}
          >
            Photo {i + 1}
          </button>
        ))}
      </div>
      <div className="mt-5 grid gap-8 lg:grid-cols-2">
        <div>
          <div className="mb-3 text-right">
            <span className="text-xs italic text-slate-400">Hover table to see displacement arrows</span>
          </div>
          {imgFailed ? (
            <p className="rounded-xl bg-amber-50 px-5 py-4 text-sm text-amber-800">
              This example photo is not available on this machine (the labelled dataset is not installed here).
            </p>
          ) : (
            <ZoomableImage
              key={ex.image_id}
              width={ex.width}
              height={ex.height}
              imageUrl={exampleImageUrl(ex.image_id)}
              ariaLabel="Human labels and both models keypoints on a held-out photo"
              onImageError={() => setImgFailed(true)}
            >
              {(r) => (
                <>
                  <defs>
                    <marker id="arrow-mfld" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                      <path d="M 0 0 L 6 3 L 0 6 z" fill={MFLD} opacity="0.95" />
                    </marker>
                    <marker id="arrow-ours" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                      <path d="M 0 0 L 6 3 L 0 6 z" fill={OURS} opacity="0.95" />
                    </marker>
                  </defs>

                  {/* Interactive Hover Lines */}
                  {kps.map((k) => (
                    hovered === k.name && (
                      <g key={`lines-${k.name}`}>
                        <line x1={k.mfld[0]} y1={k.mfld[1]} x2={k.gt[0]} y2={k.gt[1]} stroke={MFLD} strokeWidth={r * 0.6} strokeDasharray={`${r} ${r}`} opacity="0.95" markerEnd="url(#arrow-mfld)" />
                        <line x1={k.ours[0]} y1={k.ours[1]} x2={k.gt[0]} y2={k.gt[1]} stroke={OURS} strokeWidth={r * 0.6} strokeDasharray={`${r} ${r}`} opacity="0.95" markerEnd="url(#arrow-ours)" />
                      </g>
                    )
                  ))}

                  {/* Static Points */}
                  {kps.map((k) => (
                    <rect 
                      key={`m${k.name}`} 
                      x={k.mfld[0] - r} y={k.mfld[1] - r} width={2 * r} height={2 * r} 
                      transform={`rotate(45 ${k.mfld[0]} ${k.mfld[1]})`} 
                      fill={MFLD} stroke="#fff" strokeWidth={r * 0.3} 
                      opacity={hovered == null || hovered === k.name ? 1 : 0.2}
                      className="transition-opacity duration-200"
                    />
                  ))}
                  {kps.map((k) => (
                    <circle 
                      key={`o${k.name}`} 
                      cx={k.ours[0]} cy={k.ours[1]} r={r} 
                      fill={OURS} stroke="#fff" strokeWidth={r * 0.3} 
                      opacity={hovered == null || hovered === k.name ? 1 : 0.2}
                      className="transition-opacity duration-200"
                    />
                  ))}
                  {kps.map((k) => (
                    <path 
                      key={`g${k.name}`} 
                      d={`M ${k.gt[0] - r*1.5} ${k.gt[1]} L ${k.gt[0] + r*1.5} ${k.gt[1]} M ${k.gt[0]} ${k.gt[1] - r*1.5} L ${k.gt[0]} ${k.gt[1] + r*1.5}`}
                      stroke={GT} strokeWidth={r * 0.6} 
                      opacity={hovered == null || hovered === k.name ? 1 : 0.4}
                      className="transition-opacity duration-200"
                    />
                  ))}
                </>
              )}
            </ZoomableImage>
          )}
          <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-slate-500">
            <span className="flex items-center gap-1.5"><span className="inline-block h-4 w-4 text-center font-bold" style={{ color: GT }}>+</span> Human label</span>
            <span className="flex items-center gap-1.5"><span className="inline-block h-3 w-3 rounded-full" style={{ backgroundColor: OURS }} /> BettaTool</span>
            <span className="flex items-center gap-1.5"><span className="inline-block h-3 w-3 rotate-45" style={{ backgroundColor: MFLD }} /> MFLD-Net</span>
          </div>
        </div>
        <div>
          <div className="grid grid-cols-3 gap-3">
            <div className="rounded-xl bg-slate-50 p-4">
              <p className="text-xs text-slate-400">BettaTool, mean error</p>
              <p className="mt-1 text-xl font-semibold" style={{ color: OURS }}>{fmt(ex.ours_mean_px)} px</p>
              <p className="text-xs text-slate-400">{fmt((ex.ours_mean_px / ex.body_length_px) * 100)}% of body length</p>
            </div>
            <div className="rounded-xl bg-slate-50 p-4">
              <p className="text-xs text-slate-400">MFLD-Net, mean error</p>
              <p className="mt-1 text-xl font-semibold" style={{ color: MFLD }}>{fmt(ex.mfld_mean_px)} px</p>
              <p className="text-xs text-slate-400">{fmt((ex.mfld_mean_px / ex.body_length_px) * 100)}% of body length</p>
            </div>
            <div className="rounded-xl bg-slate-50 p-4">
              <p className="text-xs text-slate-400">On this photo</p>
              <p className="mt-1 text-xl font-semibold text-slate-700">{times >= 1 ? `${fmt(times)}× lower` : `${fmt(1 / times)}× higher`}</p>
              <p className="text-xs text-slate-400">error for BettaTool</p>
            </div>
          </div>
          <div className="mt-5 max-h-[22rem] overflow-y-auto rounded-xl ring-1 ring-slate-100">
            <table className="w-full text-left text-sm">
              <thead className="sticky top-0 bg-white text-xs text-slate-400">
                <tr>
                  <th className="px-3 py-2 font-medium">Hover Keypoint to view error</th>
                  <th className="px-3 py-2 text-right font-medium" style={{ color: OURS }}>BettaTool px</th>
                  <th className="px-3 py-2 text-right font-medium" style={{ color: MFLD }}>MFLD px</th>
                </tr>
              </thead>
              <tbody>
                {kps.map((k) => {
                  const eo = dist(k.gt, k.ours);
                  const em = dist(k.gt, k.mfld);
                  return (
                    <tr 
                      key={k.name} 
                      className={`border-t border-slate-50 cursor-pointer transition-colors ${hovered === k.name ? "bg-betta-50" : ""}`}
                      onMouseEnter={() => setHovered(k.name)} 
                      onMouseLeave={() => setHovered(null)}
                    >
                      <td className="px-3 py-1.5 font-medium text-slate-700">{k.label}</td>
                      <td className={`px-3 py-1.5 text-right tabular-nums ${eo <= em ? "font-bold text-slate-900" : "text-slate-500"}`}>{fmt(eo)}</td>
                      <td className={`px-3 py-1.5 text-right tabular-nums ${em < eo ? "font-bold text-slate-900" : "text-slate-500"}`}>{fmt(em)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function CompareTab({ file, imageUrl, active }) {
  const live = useAbortableLoad((signal) => compareWithMfld(file, { signal }), [file], active && !!file);
  const bench = useAbortableLoad((signal) => getMfldBenchmark({ signal }), [], active);
  return (
    <div className="space-y-8 pb-10">
      <section className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
        <h2 className="text-base font-semibold text-slate-700">Detected keypoints: BettaTool vs MFLD-Net on this photo</h2>
        <p className="mb-5 mt-1 text-sm text-slate-400">Both models see the same uploaded photo. Where they disagree, hover a row to find the keypoint on the image.</p>
        <LivePanel imageUrl={imageUrl} state={live} />
      </section>
      {bench.data?.examples?.length > 0 && (
        <section className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
          <h2 className="text-base font-semibold text-slate-700">Labelled test photos: who is closer to the human label?</h2>
          <p className="mb-5 mt-1 text-sm text-slate-400">Unlike the live photo above, these held-out photos have human labels, so the better model can be seen directly.</p>
          <ExamplesPanel data={bench.data} />
        </section>
      )}
      <section className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
        <h2 className="mb-5 text-base font-semibold text-slate-700">System Accuracy Comparisons</h2>
        <BenchmarkPanel state={bench} />
      </section>
    </div>
  );
}

/** Compare for an already-saved photo: fetches the stored image back into a File, then runs the live comparison on it. */
export function RemoteCompareTab({ imageUrl, name = "photo", active = true }) {
  const [state, setState] = useState({ file: null, error: null });
  useEffect(() => {
    if (!imageUrl || !active) return undefined;
    const ctrl = new AbortController();
    setState({ file: null, error: null });
    fetch(imageUrl, { signal: ctrl.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`Could not load the saved image (${res.status}).`);
        return res.blob();
      })
      .then((blob) => {
        const type = blob.type || "image/jpeg";
        const ext = type.split("/")[1] || "jpg";
        const base = String(name).replace(/\.[^.]+$/, "");
        setState({ file: new File([blob], `${base}.${ext}`, { type }), error: null });
      })
      .catch((err) => {
        if (err.name !== "AbortError") setState({ file: null, error: err.message });
      });
    return () => ctrl.abort();
  }, [imageUrl, name, active]);

  if (state.error) return <p className="rounded-xl bg-red-50 px-5 py-4 text-sm text-red-700">{state.error}</p>;
  if (!state.file) return <p className="text-base text-slate-400">Loading image…</p>;
  return <CompareTab file={state.file} imageUrl={imageUrl} active={active} />;
}

export default CompareTab;