import { useEffect, useMemo, useState } from "react";
import { compareWithMfld, exampleImageUrl, getMfldBenchmark } from "../api/client.js";

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

function Overlay({ imageUrl, data, showOurs, showMfld, showLines, hovered, setHovered }) {
  const { image_width: w, image_height: h, keypoints } = data;
  const r = Math.max(w, h) / 170;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full rounded-xl bg-slate-100" role="img" aria-label="Keypoints of both models on the uploaded photo">
      <image href={imageUrl} x="0" y="0" width={w} height={h} />
      {showLines &&
        keypoints.map((k) => (
          <line
            key={`l${k.index}`}
            x1={k.ours_x} y1={k.ours_y} x2={k.mfld_x} y2={k.mfld_y}
            stroke={k.distance_pct_body > DISAGREE_PCT ? "#dc2626" : "#ffffff"}
            strokeWidth={r * (hovered === k.index ? 0.9 : 0.5)}
            opacity={k.distance_pct_body > DISAGREE_PCT ? 0.95 : 0.6}
          />
        ))}
      {showMfld &&
        keypoints.map((k) => (
          <rect
            key={`m${k.index}`}
            x={k.mfld_x - r} y={k.mfld_y - r} width={2 * r} height={2 * r}
            transform={`rotate(45 ${k.mfld_x} ${k.mfld_y})`}
            fill={MFLD} stroke="#ffffff" strokeWidth={r * 0.3}
            opacity={hovered == null || hovered === k.index ? 1 : 0.35}
            onMouseEnter={() => setHovered(k.index)} onMouseLeave={() => setHovered(null)}
          />
        ))}
      {showOurs &&
        keypoints.map((k) => (
          <circle
            key={`o${k.index}`}
            cx={k.ours_x} cy={k.ours_y} r={r * (hovered === k.index ? 1.5 : 1.1)}
            fill={OURS} stroke="#ffffff" strokeWidth={r * 0.3}
            opacity={hovered == null || hovered === k.index ? 1 : 0.35}
            onMouseEnter={() => setHovered(k.index)} onMouseLeave={() => setHovered(null)}
          />
        ))}
    </svg>
  );
}

function Toggle({ checked, onChange, children }) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-sm text-slate-600">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="h-4 w-4 accent-betta-600" />
      {children}
    </label>
  );
}

function LivePanel({ imageUrl, state }) {
  const [showOurs, setShowOurs] = useState(true);
  const [showMfld, setShowMfld] = useState(true);
  const [showLines, setShowLines] = useState(true);
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
  const sorted = [...data.keypoints].sort((a, b) => b.distance_pct_body - a.distance_pct_body);
  const maxPct = Math.max(...data.keypoints.map((k) => k.distance_pct_body), DISAGREE_PCT * 2);

  return (
    <div className="grid gap-8 lg:grid-cols-2">
      <div>
        <div className="mb-3 flex flex-wrap items-center gap-x-5 gap-y-2">
          <Toggle checked={showOurs} onChange={setShowOurs}>
            <span className="inline-block h-3 w-3 rounded-full" style={{ backgroundColor: OURS }} /> Ours ({fmt(data.ours.params_millions)} M params)
          </Toggle>
          <Toggle checked={showMfld} onChange={setShowMfld}>
            <span className="inline-block h-3 w-3 rotate-45" style={{ backgroundColor: MFLD }} /> MFLD-Net ({fmt(data.mfld.params_millions, 2)} M params)
          </Toggle>
          <Toggle checked={showLines} onChange={setShowLines}>Difference lines</Toggle>
        </div>
        <Overlay imageUrl={imageUrl} data={data} showOurs={showOurs} showMfld={showMfld} showLines={showLines} hovered={hovered} setHovered={setHovered} />
        <p className="mt-3 text-xs text-slate-400">
          Red lines: the two models place that keypoint more than {DISAGREE_PCT}% of the body length apart. MFLD-Net: {data.mfld.note}.
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
              <tr><th className="px-3 py-2 font-medium">Keypoint</th><th className="px-3 py-2 font-medium">Distance between models</th><th className="px-3 py-2 text-right font-medium">px</th></tr>
            </thead>
            <tbody>
              {sorted.map((k) => (
                <tr key={k.index} onMouseEnter={() => setHovered(k.index)} onMouseLeave={() => setHovered(null)} className={hovered === k.index ? "bg-betta-50" : ""}>
                  <td className="px-3 py-1.5 text-slate-600">{k.label}</td>
                  <td className="px-3 py-1.5">
                    <div className="flex items-center gap-2">
                      <div className="h-2 flex-1 rounded-full bg-slate-100">
                        <div className="h-2 rounded-full" style={{ width: `${Math.min(100, (k.distance_pct_body / maxPct) * 100)}%`, backgroundColor: k.distance_pct_body > DISAGREE_PCT ? "#dc2626" : "#94a3b8" }} />
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
        {data.notes.map((n, i) => (
          <p key={i} className="mt-3 text-xs text-slate-400">{n}</p>
        ))}
      </div>
    </div>
  );
}

function Bar({ value, max, color }) {
  return <div className="h-2.5 rounded-full" style={{ width: `${Math.max(2, (value / max) * 100)}%`, backgroundColor: color }} />;
}

function BenchmarkPanel({ state }) {
  const rows = state.data?.rows ?? [];
  const perKp = state.data?.per_keypoint ?? [];
  const maxKp = useMemo(() => Math.max(1, ...perKp.flatMap((k) => [k.ours_px, k.mfld_px])), [perKp]);
  if (state.status === "loading") return <p className="text-sm text-slate-500">Loading the benchmark…</p>;
  if (state.status === "error") return <p className="rounded-xl bg-amber-50 px-5 py-4 text-sm text-amber-800">{state.error}</p>;
  if (!state.data) return null;
  const d = state.data;
  return (
    <div>
      <p className="text-sm text-slate-500">{d.protocol}</p>
      <div className="mt-4 overflow-x-auto rounded-xl ring-1 ring-slate-100">
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-50 text-xs text-slate-400">
            <tr>
              <th className="px-4 py-2 font-medium">Measured on {d.n_images} labelled test images</th>
              <th className="px-4 py-2 text-right font-medium" style={{ color: OURS }}>Ours</th>
              <th className="px-4 py-2 text-right font-medium" style={{ color: MFLD }}>MFLD-Net</th>
              <th className="px-4 py-2 text-right font-medium">Better</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key} className="border-t border-slate-100">
                <td className="px-4 py-2 text-slate-600">
                  {r.label} <span className="text-xs text-slate-400">({r.unit}{r.higher_is_better ? ", higher is better" : ", lower is better"})</span>
                  {r.note && <p className="text-xs text-slate-400">{r.note}</p>}
                </td>
                <td className="px-4 py-2 text-right tabular-nums text-slate-700">{fmt(r.ours, r.ours < 10 ? 2 : 1)}</td>
                <td className="px-4 py-2 text-right tabular-nums text-slate-700">{fmt(r.mfld, r.mfld < 10 ? 2 : 1)}</td>
                <td className="px-4 py-2 text-right">
                  <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${r.better === "ours" ? "bg-emerald-100 text-emerald-700" : "bg-slate-200/70 text-slate-600"}`}>
                    {r.better === "ours" ? "Ours" : "MFLD-Net"}{r.times_better ? ` ${fmt(r.times_better)}×` : ""}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-6 text-sm font-medium text-slate-600">Mean error per keypoint (px, lower is better)</p>
      <div className="mt-3 grid gap-x-8 gap-y-1.5 sm:grid-cols-2">
        {perKp.map((k) => (
          <div key={k.name} className="text-xs text-slate-500">
            <div className="flex justify-between"><span>{k.label}</span><span className="tabular-nums">{fmt(k.ours_px)} vs {fmt(k.mfld_px)}</span></div>
            <Bar value={k.ours_px} max={maxKp} color={OURS} />
            <div className="mt-0.5"><Bar value={k.mfld_px} max={maxKp} color={MFLD} /></div>
          </div>
        ))}
      </div>

      <ul className="mt-6 list-disc space-y-1 pl-5 text-xs text-slate-400">
        {d.caveats.map((c, i) => <li key={i}>{c}</li>)}
        <li>Ours: {d.ours.training}. MFLD-Net: {d.mfld.training}. Generated {d.generated}.</li>
      </ul>
    </div>
  );
}


const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/** Labelled held-out examples: the human label (green) next to both models, with each model's error line. */
function ExamplesPanel({ data }) {
  const examples = data?.examples ?? [];
  const [sel, setSel] = useState(0);
  const [imgFailed, setImgFailed] = useState(false);
  useEffect(() => setImgFailed(false), [sel]);
  if (!examples.length) return null;
  const ex = examples[Math.min(sel, examples.length - 1)];
  const kps = ex.keypoints.filter((k) => k.visible);
  const r = Math.max(ex.width, ex.height) / 170;
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
          {imgFailed ? (
            <p className="rounded-xl bg-amber-50 px-5 py-4 text-sm text-amber-800">
              This example photo is not available on this machine (the labelled dataset is not installed here).
            </p>
          ) : (
            <svg viewBox={`0 0 ${ex.width} ${ex.height}`} className="w-full rounded-xl bg-slate-100" role="img" aria-label="Human labels and both models keypoints on a held-out photo">
              <image href={exampleImageUrl(ex.image_id)} x="0" y="0" width={ex.width} height={ex.height} onError={() => setImgFailed(true)} />
              {kps.map((k) => (
                <g key={k.name}>
                  <line x1={k.gt[0]} y1={k.gt[1]} x2={k.mfld[0]} y2={k.mfld[1]} stroke={MFLD} strokeWidth={r * 0.4} opacity="0.85" />
                  <line x1={k.gt[0]} y1={k.gt[1]} x2={k.ours[0]} y2={k.ours[1]} stroke={OURS} strokeWidth={r * 0.4} opacity="0.95" />
                </g>
              ))}
              {kps.map((k) => (
                <rect key={`m${k.name}`} x={k.mfld[0] - r} y={k.mfld[1] - r} width={2 * r} height={2 * r} transform={`rotate(45 ${k.mfld[0]} ${k.mfld[1]})`} fill={MFLD} stroke="#fff" strokeWidth={r * 0.3} />
              ))}
              {kps.map((k) => (
                <circle key={`o${k.name}`} cx={k.ours[0]} cy={k.ours[1]} r={r} fill={OURS} stroke="#fff" strokeWidth={r * 0.3} />
              ))}
              {kps.map((k) => (
                <circle key={`g${k.name}`} cx={k.gt[0]} cy={k.gt[1]} r={r * 1.5} fill="none" stroke={GT} strokeWidth={r * 0.5} />
              ))}
            </svg>
          )}
          <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1 text-xs text-slate-500">
            <span className="flex items-center gap-1.5"><span className="inline-block h-3 w-3 rounded-full border-2" style={{ borderColor: GT }} /> Human label</span>
            <span className="flex items-center gap-1.5"><span className="inline-block h-3 w-3 rounded-full" style={{ backgroundColor: OURS }} /> Ours</span>
            <span className="flex items-center gap-1.5"><span className="inline-block h-3 w-3 rotate-45" style={{ backgroundColor: MFLD }} /> MFLD-Net</span>
            <span>Lines run from the label to each model point: a shorter line is a better answer.</span>
          </div>
        </div>
        <div>
          <div className="grid grid-cols-3 gap-3">
            <div className="rounded-xl bg-slate-50 p-4">
              <p className="text-xs text-slate-400">Ours, mean error</p>
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
              <p className="text-xs text-slate-400">error for ours</p>
            </div>
          </div>
          <div className="mt-5 max-h-[22rem] overflow-y-auto rounded-xl ring-1 ring-slate-100">
            <table className="w-full text-left text-sm">
              <thead className="sticky top-0 bg-white text-xs text-slate-400">
                <tr>
                  <th className="px-3 py-2 font-medium">Keypoint</th>
                  <th className="px-3 py-2 text-right font-medium" style={{ color: OURS }}>Ours px</th>
                  <th className="px-3 py-2 text-right font-medium" style={{ color: MFLD }}>MFLD px</th>
                </tr>
              </thead>
              <tbody>
                {kps.map((k) => {
                  const eo = dist(k.gt, k.ours);
                  const em = dist(k.gt, k.mfld);
                  return (
                    <tr key={k.name} className="border-t border-slate-50">
                      <td className="px-3 py-1.5 text-slate-600">{k.label}</td>
                      <td className={`px-3 py-1.5 text-right tabular-nums ${eo <= em ? "font-semibold text-slate-800" : "text-slate-500"}`}>{fmt(eo)}</td>
                      <td className={`px-3 py-1.5 text-right tabular-nums ${em < eo ? "font-semibold text-slate-800" : "text-slate-500"}`}>{fmt(em)}</td>
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
    <div className="space-y-8">
      <section className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
        <h2 className="text-base font-semibold text-slate-700">Detected keypoints: ours vs MFLD-Net on this photo</h2>
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
        <h2 className="text-base font-semibold text-slate-700">Which one is more accurate? Measured against labels</h2>
        <p className="mb-5 mt-1 text-sm text-slate-400">The evidence for accuracy comes from held-out labelled images, not from a single photo.</p>
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
