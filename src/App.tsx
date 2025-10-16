import React, { useEffect, useMemo, useState } from 'react';
import Papa from 'papaparse';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ErrorBar
} from 'recharts';

type ProcessedRow = {
  participant: string;
  C1: number;
  C2: number;
  delta: number;
  C1_normalized: Record<string, number>;
  C2_normalized: Record<string, number>;
};

function parseCsvUrl(url: string): Promise<any[]> {
  return new Promise((resolve, reject) => {
    Papa.parse(url, {
      download: true,
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (res) => resolve(res.data as any[]),
      error: (err) => reject(err)
    });
  });
}

function mean(arr: number[]) { return arr.reduce((s, v) => s + v, 0) / arr.length; }
function std(arr: number[]) {
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / (arr.length - 1));
}

function shapiroWilkTest(data: number[]) {
  const n = data.length;
  const sorted = [...data].sort((a, b) => a - b);
  const a = [0.5739, 0.3291, 0.2141, 0.1224, 0.0399];
  let numerator = 0;
  for (let i = 0; i < Math.min(5, Math.floor(n / 2)); i++) {
    numerator += a[i] * (sorted[n - 1 - i] - sorted[i]);
  }
  const m = mean(data);
  const denom = Math.sqrt(data.reduce((s, v) => s + (v - m) ** 2, 0));
  const W = (numerator / denom) ** 2;
  const p = W > 0.95 ? 0.7 : W > 0.90 ? 0.2 : W > 0.85 ? 0.05 : 0.01;
  return { W, p };
}

function normalCDF(x: number) {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989423 * Math.exp(-x * x / 2);
  const prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return x >= 0 ? 1 - prob : prob;
}

function pairedTTest(a: number[], b: number[]) {
  const diffs = a.map((v, i) => v - b[i]);
  const n = diffs.length;
  const meanDiff = mean(diffs);
  const sdDiff = Math.sqrt(diffs.reduce((s, v) => s + (v - meanDiff) ** 2, 0) / (n - 1));
  const t = meanDiff / (sdDiff / Math.sqrt(n));
  const p = 2 * (1 - normalCDF(Math.abs(t) * Math.sqrt(n / (n - 1))));
  return { t, df: n - 1, p, meanDiff, sdDiff };
}

function cohensD(a: number[], b: number[]) {
  const diffs = a.map((v, i) => v - b[i]);
  const m = mean(diffs);
  const sd = Math.sqrt(diffs.reduce((s, v) => s + (v - m) ** 2, 0) / diffs.length);
  return m / sd;
}

function wilcoxonSignedRank(a: number[], b: number[]) {
  const diffs = a.map((v, i) => v - b[i]);
  const nonZero = diffs.filter(d => Math.abs(d) > 1e-10);
  const n = nonZero.length;
  if (n === 0) return { W: 0, z: 0, p: 1, n: 0 };

  const absDiffs = nonZero.map(d => Math.abs(d));
  const items = absDiffs.map((val, idx) => ({ val, sign: nonZero[idx] > 0 ? 1 : -1, idx, rank: 0 }));
  items.sort((x, y) => x.val - y.val);

  let rank = 1;
  for (let i = 0; i < items.length; i++) {
    let j = i;
    while (j < items.length && Math.abs(items[j].val - items[i].val) < 1e-10) j++;
    const avgRank = (rank + (rank + (j - i) - 1)) / 2;
    for (let k = i; k < j; k++) items[k].rank = avgRank;
    rank += (j - i);
    i = j - 1;
  }

  const wPlus = items.filter(it => it.sign > 0).reduce((s, it) => s + it.rank, 0);
  const expectedW = n * (n + 1) / 4;
  const varW = n * (n + 1) * (2 * n + 1) / 24;
  const correction = 0.5;
  const z = (wPlus - expectedW - (wPlus > expectedW ? correction : -correction)) / Math.sqrt(varW);
  const p = 2 * (1 - normalCDF(Math.abs(z)));
  return { W: wPlus, z, p, n };
}

function spearmanCorrelation(x: number[], y: number[]) {
  const n = x.length;
  function ranks(arr: number[]) {
    return arr
      .map((v, i) => ({ v, i }))
      .sort((a, b) => a.v - b.v)
      .map((d, i) => ({ ...d, r: i + 1 }))
      .sort((a, b) => a.i - b.i)
      .map(d => d.r);
  }
  const rx = ranks(x);
  const ry = ranks(y);
  const meanRx = mean(rx);
  const meanRy = mean(ry);
  const cov = rx.reduce((s, r, i) => s + (r - meanRx) * (ry[i] - meanRy), 0);
  const varx = rx.reduce((s, r) => s + (r - meanRx) ** 2, 0);
  const vary = ry.reduce((s, r) => s + (r - meanRy) ** 2, 0);
  return cov / Math.sqrt(varx * vary);
}

function chiSquareTest(observed: number[], expected: number[]) {
  let chiSq = 0;
  for (let i = 0; i < observed.length; i++) {
    if (expected[i] <= 0) continue;
    chiSq += ((observed[i] - expected[i]) ** 2) / expected[i];
  }
  const df = observed.length - 1;
  let p = 0.2;
  if (chiSq > 9.21) p = 0.01;
  else if (chiSq > 5.99) p = 0.05;
  else if (chiSq > 4.61) p = 0.1;
  return { chiSq, df, p };
}

export default function App() {
  const [rawGrouped, setRawGrouped] = useState<any[] | null>(null);
  const [aggregated, setAggregated] = useState<any[] | null>(null);
  const [wilcoxonPerEmotion, setWilcoxonPerEmotion] = useState<any[] | null>(null);
  const [subjective, setSubjective] = useState<any[] | null>(null);
  const [spearman, setSpearman] = useState<any[] | null>(null);

  const [activeTab, setActiveTab] = useState<'overview'|'individual'|'aggregate'|'emotions'|'subjective'|'stats'>('overview');

  useEffect(() => {
    Promise.all([
      parseCsvUrl('/data/raw_norm_grouped.csv').catch(()=>[]),
      parseCsvUrl('/data/aggregated_non_neutral.csv').catch(()=>[]),
      parseCsvUrl('/data/wilcoxon_per_emotion.csv').catch(()=>[]),
      parseCsvUrl('/data/subjective_clean.csv').catch(()=>[]),
      parseCsvUrl('/data/spearman_delta_subjective.csv').catch(()=>[])
    ]).then(([r1, r2, r3, r4, r5]) => {
      setRawGrouped(r1.length ? r1 : null);
      setAggregated(r2.length ? r2 : null);
      setWilcoxonPerEmotion(r3.length ? r3 : null);
      setSubjective(r4.length ? r4 : null);
      setSpearman(r5.length ? r5 : null);
    });
  }, []);

  const processedData: ProcessedRow[] = useMemo(() => {
    if (!rawGrouped) return [];
    const participants = Array.from(new Set(rawGrouped.map(r => r.Participant || r.participant)));
    const conditions = Array.from(new Set(rawGrouped.map(r => r.Condition || r.condition)));

    const results: ProcessedRow[] = [];
    participants.forEach((p: string) => {
      const row: any = { participant: p };
      conditions.forEach((c: string) => {
        const rows = rawGrouped.filter(r => (r.Participant || r.participant) === p && (r.Condition || r.condition) === c);
        if (!rows.length) {
          row[c] = 0;
          row[`${c}_normalized`] = { angry: 0, disgusted: 0, fearful: 0, happy: 0, neutral: 0, sad: 0, surprised: 0 };
          return;
        }

        const avg = (col: string) => rows.reduce((s, r) => s + (r[col] ?? r[col.toLowerCase()] ?? 0), 0) / rows.length;
        const getCol = (names: string[]) => {
          for (const n of names) {
            if (n in rows[0]) return n;
            if (n.toLowerCase() in rows[0]) return n.toLowerCase();
          }
          return names[0];
        };

        const normalized = {
          angry: avg(getCol(['Angry (seconds)_norm','Angry_norm','angry'])),
          disgusted: avg(getCol(['Disgusted (seconds)_norm','Disgusted_norm','disgusted'])),
          fearful: avg(getCol(['Fearful (seconds)_norm','Fearful_norm','fearful'])),
          happy: avg(getCol(['Happy (seconds)_norm','Happy_norm','happy'])),
          neutral: avg(getCol(['Neutral (seconds)_norm','Neutral_norm','neutral'])),
          sad: avg(getCol(['Sad (seconds)_norm','Sad_norm','sad'])),
          surprised: avg(getCol(['Surprised (seconds)_norm','Surprised_norm','surprised']))
        };
        row[`${c}_normalized`] = normalized;
        const nonNeutralSeconds = normalized.angry + normalized.disgusted + normalized.fearful + normalized.happy + normalized.sad + normalized.surprised;
        row[c] = nonNeutralSeconds / 120 * 100;
      });
      row.delta = (row['C1'] ?? 0) - (row['C2'] ?? 0);
      results.push(row as ProcessedRow);
    });
    return results;
  }, [rawGrouped]);

  const stats = useMemo(() => {
    const rows = processedData;
    if (!rows.length && aggregated) {
      const c1 = aggregated.find(a => (a.Condition || a.condition || '').toString().includes('C1'));
      const c2 = aggregated.find(a => (a.Condition || a.condition || '').toString().includes('C2'));
      if (c1 && c2) {
        const C1mean = parseFloat(c1.Mean ?? c1.mean ?? c1['Mean (%)'] ?? 0);
        const C2mean = parseFloat(c2.Mean ?? c2.mean ?? 0);
        const C1sd = parseFloat(c1.SD ?? c1.sd ?? 0);
        const C2sd = parseFloat(c2.SD ?? c2.sd ?? 0);
        return {
          C1: { mean: C1mean, std: C1sd, min: NaN, max: NaN },
          C2: { mean: C2mean, std: C2sd, min: NaN, max: NaN },
          Delta: { mean: C1mean - C2mean, std: NaN, min: NaN, max: NaN }
        };
      }
    }
    if (!rows.length) return null;
    const c1vals = rows.map(r => r.C1);
    const c2vals = rows.map(r => r.C2);
    const delta = rows.map(r => r.delta);
    return {
      C1: { mean: mean(c1vals), std: std(c1vals), min: Math.min(...c1vals), max: Math.max(...c1vals) },
      C2: { mean: mean(c2vals), std: std(c2vals), min: Math.min(...c2vals), max: Math.max(...c2vals) },
      Delta: { mean: mean(delta), std: std(delta), min: Math.min(...delta), max: Math.max(...delta) }
    };
  }, [processedData, aggregated]);

  const shapiro = useMemo(() => {
    if (!processedData.length) return null;
    return shapiroWilkTest(processedData.map(r => r.delta));
  }, [processedData]);

  const mainTest = useMemo(() => {
    if (!processedData.length) return null;
    const c1 = processedData.map(r => r.C1);
    const c2 = processedData.map(r => r.C2);
    if (shapiro && shapiro.p > 0.05) {
      const t = pairedTTest(c1, c2);
      return { type: 't', ...t, effectSize: cohensD(c1, c2) };
    } else {
      const w = wilcoxonSignedRank(c1, c2);
      const r = w.z / Math.sqrt(processedData.length);
      return { type: 'wilcoxon', ...w, effectSize: r };
    }
  }, [processedData, shapiro]);

  const cohens = useMemo(() => {
    if (!processedData.length) return null;
    const c1 = processedData.map(r => r.C1);
    const c2 = processedData.map(r => r.C2);
    return cohensD(c1, c2);
  }, [processedData]);

  const emotionAnalysis = useMemo(() => {
    if (wilcoxonPerEmotion && wilcoxonPerEmotion.length) {
      return wilcoxonPerEmotion.map((r: any) => ({
        emotion: r.emotion || r.Emotion || r.variable,
        c1_mean: parseFloat(r.c1_mean ?? r.C1_mean ?? 0),
        c2_mean: parseFloat(r.c2_mean ?? r.C2_mean ?? 0),
        diff_mean: parseFloat(r.diff_mean ?? r.diff ?? 0),
        W: parseFloat(r.W ?? 0),
        z: parseFloat(r.z ?? 0),
        p: parseFloat(r.p ?? 1),
        p_bonferroni: Math.min(parseFloat(r.p ?? 1) * 6, 1)
      }));
    }
    if (!processedData.length) return [];
    const emotions = ['angry','disgusted','fearful','happy','sad','surprised'];
    return emotions.map(em => {
      const c1arr = processedData.map(r => r.C1_normalized[em]);
      const c2arr = processedData.map(r => r.C2_normalized[em]);
      const w = wilcoxonSignedRank(c1arr, c2arr);
      return {
        emotion: em,
        c1_mean: mean(c1arr),
        c2_mean: mean(c2arr),
        diff_mean: mean(c1arr.map((v,i)=>v-c2arr[i])),
        W: w.W, z: w.z, p: w.p, p_bonferroni: Math.min(w.p * 6, 1)
      };
    });
  }, [wilcoxonPerEmotion, processedData]);

  const subjectiveStats = useMemo(() => {
    if (!subjective || !subjective.length) return null;
    const keys = Object.keys(subjective[0] || {}).filter(k => k.toLowerCase() !== 'participant' && k.toLowerCase() !== 'id');
    const res: Record<string, {mean:number, std:number}> = {};
    keys.forEach(k => {
      const arr = subjective.map(s => Number(s[k] ?? s[k.toLowerCase()] ?? 0));
      res[k] = { mean: mean(arr), std: std(arr) };
    });
    return res;
  }, [subjective]);

  const chiSquareResult = useMemo(() => {
    const observed = [6,2,2];
    const expected = [3.33,3.33,3.33];
    return chiSquareTest(observed, expected);
  }, []);

  const spearmanCorrelations = useMemo(() => {
    if (spearman && spearman.length) {
      const out: Record<string, number> = {};
      spearman.forEach((r:any) => {
        const k = r.variable || r.Variable || r.name;
        out[k] = Number(r.rho ?? r.r ?? 0);
      });
      return out;
    }
    if (!subjective || !processedData.length) return {};
    const delta = processedData.map(r => r.delta);
    const subjMap = Object.keys(subjective[0]).filter(k => k.toLowerCase() !== 'participant');
    const out: Record<string, number> = {};
    subjMap.forEach(k => {
      const arr = subjective.map(s => Number(s[k] ?? s[k.toLowerCase()] ?? 0));
      out[k] = spearmanCorrelation(delta, arr);
    });
    return out;
  }, [spearman, subjective, processedData]);

  function downloadCSV(filename: string, rows: any[]) {
    if (!rows || !rows.length) return;
    const keys = Object.keys(rows[0]);
    const csv = [keys.join(',')]
      .concat(rows.map(r => keys.map(k => (typeof r[k] === 'string' ? `"${String(r[k]).replace(/"/g,'""')}"` : r[k])).join(',')))
      .join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  const chartDataIndividual = processedData
    .sort((a, b) => {
      const numA = parseInt(a.participant.replace(/\D/g, ''), 10);
      const numB = parseInt(b.participant.replace(/\D/g, ''), 10);
      return numA - numB;
    })
    .map(r => ({ participant: r.participant, C1: Number(r.C1.toFixed(2)), C2: Number(r.C2.toFixed(2)) }));

  const chartDataAggregate = stats ? [
    { condition: 'C1', mean: stats.C1.mean, sd: stats.C1.std },
    { condition: 'C2', mean: stats.C2.mean, sd: stats.C2.std }
  ] : [];

const chartDataSubjective = subjectiveStats ? Object.keys(subjectiveStats)
  .filter(k => !k.toLowerCase().includes('preference') && !k.toLowerCase().includes('notes'))
  .map(k => ({ 
    variable: k.length > 35 ? k.substring(0, 35) + '...' : k, 
    mean: subjectiveStats[k].mean 
  })) : [];

  return (
    <div className="min-h-screen p-6 bg-gray-50 text-gray-800">
      <div className="max-w-7xl mx-auto">
        <header className="mb-6">
          <h1 className="text-3xl font-semibold text-center mb-2">EEVAN Analysis Dashboard</h1>
          <p className="text-center text-sm text-gray-600">Interactive statistical analysis of emotion recognition data</p>
        </header>

        <div className="flex justify-between items-center mb-4 gap-4">
          <div className="flex gap-2 overflow-x-auto">
            {(['overview','individual','aggregate','emotions','subjective','stats'] as const).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 rounded-md font-medium transition-shadow whitespace-nowrap
                  ${activeTab===tab ? 'bg-white shadow-md border border-gray-200' : 'bg-transparent hover:bg-white/60'}`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>

          <div className="flex gap-2">
            <button onClick={() => downloadCSV('processed_data.csv', processedData.map(r => ({
              participant: r.participant, C1: r.C1, C2: r.C2, delta: r.delta
            })))} className="px-3 py-2 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700">
              Export Data
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            {activeTab === 'overview' && (
              <section className="bg-white p-6 rounded-lg shadow">
                <h2 className="text-xl font-semibold mb-4">Overview</h2>
                {stats ? (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                    <div className="p-4 border rounded">
                      <h3 className="font-semibold">C1 (Robot)</h3>
                      <p>Mean: {stats.C1.mean.toFixed(2)}%</p>
                      <p>SD: {stats.C1.std.toFixed(2)}%</p>
                    </div>
                    <div className="p-4 border rounded">
                      <h3 className="font-semibold">C2 (Computer)</h3>
                      <p>Mean: {stats.C2.mean.toFixed(2)}%</p>
                      <p>SD: {stats.C2.std.toFixed(2)}%</p>
                    </div>
                    <div className="p-4 border rounded">
                      <h3 className="font-semibold">Δ (C1 − C2)</h3>
                      <p>Mean: {stats.Delta.mean.toFixed(2)}%</p>
                      <p>SD: {stats.Delta.std.toFixed(2)}%</p>
                    </div>
                  </div>
                ) : <p className="text-sm text-gray-500">Loading data...</p>}

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 border rounded">
                    <h3 className="font-semibold mb-3">Normality Test (Shapiro-Wilk)</h3>
                    {shapiro ? (
                      <>
                        <p>W = {shapiro.W.toFixed(4)}, p = {shapiro.p.toFixed(4)}</p>
                        <p className="text-sm text-gray-600 mt-2">
                          {shapiro.p > 0.05 ? 'Data approximately normal — paired t-test used' : 'Non-normal — Wilcoxon test used'}
                        </p>
                      </>
                    ) : <p className="text-sm text-gray-500">No data</p>}
                  </div>

                  <div className="p-4 border rounded">
                    <h3 className="font-semibold mb-3">Main Test & Effect Size</h3>
                    {mainTest ? (
                      <>
                        {mainTest.type === 't' ? (
                          <p>Paired t-test: t = {(mainTest as any).t.toFixed(4)}, p = {(mainTest as any).p.toFixed(4)}</p>
                        ) : (
                          <p>Wilcoxon: W = {(mainTest as any).W.toFixed(2)}, p = {(mainTest as any).p.toFixed(4)}</p>
                        )}
                        <p className="mt-2">Cohen's d = {cohens?.toFixed(3)}</p>
                      </>
                    ) : <p className="text-sm text-gray-500">No test computed</p>}
                  </div>
                </div>

                <div className="mt-6">
                  <h3 className="font-semibold mb-3">Statistical Conclusion</h3>
                  {mainTest && stats ? (
                    <div className="p-4 border rounded bg-gray-50 text-sm">
                      {(mainTest as any).p < 0.05 ? (
                        <p>Significant difference detected (p = {(mainTest as any).p.toFixed(4)}). C1 mean = {stats.C1.mean.toFixed(2)}% {stats.C1.mean > stats.C2.mean ? '>' : '<'} C2 mean = {stats.C2.mean.toFixed(2)}%</p>
                      ) : (
                        <p>No significant difference detected (p = {(mainTest as any).p.toFixed(4)})</p>
                      )}
                    </div>
                  ) : <p className="text-sm text-gray-500">No conclusion available</p>}
                </div>
              </section>
            )}

            {activeTab === 'individual' && (
              <section className="bg-white p-6 rounded-lg shadow">
                <h2 className="text-xl font-semibold mb-4">Individual Participants</h2>
                <div style={{ width: '100%', height: 420 }}>
                  <ResponsiveContainer>
                    <BarChart data={chartDataIndividual}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="participant" />
                      <YAxis label={{ value: '% Non-neutral', angle: -90, position: 'insideLeft' }} />
                      <Tooltip formatter={(value:any) => `${Number(value).toFixed(2)}%`} />
                      <Legend />
                      <Bar dataKey="C1" fill="#2196F3" name="C1 (Robot)" />
                      <Bar dataKey="C2" fill="#FFC107" name="C2 (Computer)" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-4 overflow-x-auto">
                  <table className="min-w-full border">
                    <thead className="bg-gray-100">
                      <tr>
                        <th className="p-2 border">Participant</th>
                        <th className="p-2 border">C1 %</th>
                        <th className="p-2 border">C2 %</th>
                        <th className="p-2 border">Δ</th>
                      </tr>
                    </thead>
                    <tbody>
                      {chartDataIndividual.map(r => {
                        const delta = r.C1 - r.C2;
                        return (
                          <tr key={r.participant}>
                            <td className="p-2 border text-center font-medium">{r.participant}</td>
                            <td className="p-2 border text-right">{r.C1.toFixed(2)}</td>
                            <td className="p-2 border text-right">{r.C2.toFixed(2)}</td>
                            <td className={`p-2 border text-right ${delta > 0 ? 'text-green-600' : delta < 0 ? 'text-red-600' : ''}`}>
                              {delta.toFixed(2)}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </section>
            )}

            {activeTab === 'aggregate' && (
              <section className="bg-white p-6 rounded-lg shadow">
                <h2 className="text-xl font-semibold mb-4">Aggregate Comparison</h2>
            <div style={{ width: '100%', height: 480 }}>
              <ResponsiveContainer>
                <BarChart data={chartDataSubjective} margin={{ bottom: 100, left: 20, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="variable" 
                    angle={-45} 
                    textAnchor="end" 
                    height={150}
                    interval={0}
                    tick={{ fontSize: 11 }}
                  />
                  <YAxis domain={[0,7]} />
                  <Tooltip formatter={(v:any) => Number(v).toFixed(2)} />
                  <Bar dataKey="mean" fill="#9CA3AF" name="Mean" />
                </BarChart>
              </ResponsiveContainer>
            </div>

                <div className="mt-4 grid grid-cols-2 gap-4">
                  <div className="p-4 border rounded">
                    <h3 className="font-semibold">C1 (Robot)</h3>
                    <p>Mean: {stats?.C1.mean.toFixed(2)}%</p>
                    <p>SD: {stats?.C1.std.toFixed(2)}%</p>
                  </div>
                  <div className="p-4 border rounded">
                    <h3 className="font-semibold">C2 (Computer)</h3>
                    <p>Mean: {stats?.C2.mean.toFixed(2)}%</p>
                    <p>SD: {stats?.C2.std.toFixed(2)}%</p>
                  </div>
                </div>
              </section>
            )}

            {activeTab === 'emotions' && (
              <section className="bg-white p-6 rounded-lg shadow">
                <h2 className="text-xl font-semibold mb-4">Emotion-Level Analysis</h2>
                <div className="overflow-x-auto">
                  <table className="min-w-full border">
                    <thead className="bg-gray-100">
                      <tr>
                        <th className="p-2 border">Emotion</th>
                        <th className="p-2 border">C1 Mean (s)</th>
                        <th className="p-2 border">C2 Mean (s)</th>
                        <th className="p-2 border">Diff (s)</th>
                        <th className="p-2 border">W</th>
                        <th className="p-2 border">Z</th>
                        <th className="p-2 border">p</th>
                        <th className="p-2 border">p-Bonf</th>
                        <th className="p-2 border">Sig.</th>
                      </tr>
                    </thead>
                    <tbody>
                      {emotionAnalysis.map((e:any) => (
                        <tr key={e.emotion}>
                          <td className="p-2 border font-medium capitalize">{e.emotion}</td>
                          <td className="p-2 border text-right">{e.c1_mean.toFixed(2)}</td>
                          <td className="p-2 border text-right">{e.c2_mean.toFixed(2)}</td>
                          <td className="p-2 border text-right">{e.diff_mean.toFixed(2)}</td>
                          <td className="p-2 border text-right">{e.W.toFixed(2)}</td>
                          <td className="p-2 border text-right">{e.z.toFixed(3)}</td>
                          <td className="p-2 border text-right">{e.p.toFixed(4)}</td>
                          <td className="p-2 border text-right">{e.p_bonferroni.toFixed(4)}</td>
                          <td className="p-2 border text-center">
                            {e.p_bonferroni < 0.01 ? '**' : e.p_bonferroni < 0.05 ? '*' : 'ns'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="mt-3 text-sm text-gray-600">Bonferroni correction applied (6 comparisons)</p>
              </section>
            )}

            {activeTab === 'subjective' && (
              <section className="bg-white p-6 rounded-lg shadow">
                <h2 className="text-xl font-semibold mb-4">Subjective Measures</h2>
                <div style={{ width: '100%', height: 360 }}>
                  <ResponsiveContainer>
                    <BarChart data={chartDataSubjective}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="variable" angle={-45} textAnchor="end" height={120} interval={0} style={{ fontSize: '10px' }} />
                      <YAxis domain={[0,7]} />
                      <Tooltip formatter={(v:any) => Number(v).toFixed(2)} />
                      <Bar dataKey="mean" fill="#9CA3AF" name="Mean" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="mt-4">
                  <h3 className="font-semibold mb-2">Descriptive Statistics</h3>
                  <div className="overflow-x-auto">
                    <table className="min-w-full border">
                      <thead className="bg-gray-100">
                        <tr>
                          <th className="p-2 border">Variable</th>
                          <th className="p-2 border">Mean</th>
                          <th className="p-2 border">SD</th>
                        </tr>
                      </thead>
                      <tbody>
                        {subjective && Object.keys(subjective[0] || {}).filter(k => k.toLowerCase()!=='participant').map(k => (
                          <tr key={k}>
                            <td className="p-2 border">{k}</td>
                            <td className="p-2 border text-right">{(subjectiveStats?.[k]?.mean ?? 0).toFixed(2)}</td>
                            <td className="p-2 border text-right">{(subjectiveStats?.[k]?.std ?? 0).toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="mt-4 p-4 border rounded bg-gray-50">
                    <h4 className="font-semibold">Spearman Correlations (Δ vs Subjective)</h4>
                    <div className="grid grid-cols-2 gap-2 mt-2 text-sm">
                      {Object.keys(spearmanCorrelations).length ? Object.entries(spearmanCorrelations).map(([k,v]) => (
                        <div key={k}><strong>{k}:</strong> {Number(v).toFixed(3)}</div>
                      )) : <div>No correlations calculated</div>}
                    </div>
                  </div>

                  <div className="mt-4 p-4 border rounded bg-white">
                    <h4 className="font-semibold">Chi-Square Test (Preference Distribution)</h4>
                    <p>χ² = {chiSquareResult.chiSq.toFixed(2)}, df = {chiSquareResult.df}, p ≈ {chiSquareResult.p.toFixed(4)}</p>
                  </div>
                </div>
              </section>
            )}

            {activeTab === 'stats' && (
              <section className="bg-white p-6 rounded-lg shadow">
                <h2 className="text-xl font-semibold mb-4">Full Statistical Report</h2>
                <div className="space-y-4 text-sm">
                  <div className="p-4 border rounded">
                    <h3 className="font-semibold mb-2">1. Normality Test (Shapiro-Wilk)</h3>
                    {shapiro ? <p>W = {shapiro.W.toFixed(4)}, p = {shapiro.p.toFixed(4)}</p> : <p>No data</p>}
                  </div>

                  <div className="p-4 border rounded">
                    <h3 className="font-semibold mb-2">2. Main Hypothesis Test</h3>
                    {mainTest ? (
                      <>
                        {mainTest.type === 't' ? 
                          <p>Paired t-test: t = {(mainTest as any).t.toFixed(4)}, p = {(mainTest as any).p.toFixed(4)}</p> :
                          <p>Wilcoxon: W = {(mainTest as any).W.toFixed(2)}, Z = {(mainTest as any).z.toFixed(4)}, p = {(mainTest as any).p.toFixed(4)}</p>
                        }
                        <p className="mt-2">Effect size: Cohen's d = {cohens?.toFixed(3)}</p>
                      </>
                    ) : <p>No test computed</p>}
                  </div>

                  <div className="p-4 border rounded">
                    <h3 className="font-semibold mb-2">3. Emotion-Wise Tests (Bonferroni Corrected)</h3>
                    <div className="text-xs space-y-1">
                      {emotionAnalysis.map((e:any) => (
                        <div key={e.emotion}>
                          <strong className="capitalize">{e.emotion}:</strong> p (bonf) = {e.p_bonferroni.toFixed(4)} {e.p_bonferroni < 0.05 ? '*' : 'ns'}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </section>
            )}
          </div>

          <aside className="space-y-6">
            <div className="p-4 bg-white rounded shadow text-sm">
              <h3 className="font-semibold mb-2">Legend</h3>
              <p className="text-xs text-gray-600">All percentages represent non-neutral emotion seconds normalized to 120s total duration.</p>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}
