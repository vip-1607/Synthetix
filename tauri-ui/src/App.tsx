import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";

type SummaryOutcome = { output_path: string; message: string; elapsed_ms: number; peak_memory_bytes: number };
type SummaryMetrics = { elapsed_ms: number; peak_memory_bytes: number };
const documentFilters = [{ name: "문서", extensions: ["txt", "md", "markdown", "pdf"] }];
const suggestedOutput = (path: string) => path.replace(/\.[^/.\\]+$/, "_요약.txt");

export default function App() {
  const [inputPaths, setInputPaths] = useState<string[]>([]);
  const [outputPath, setOutputPath] = useState("");
  const [status, setStatus] = useState("문서를 선택하면 요약 준비가 완료됩니다.");
  const [metrics, setMetrics] = useState<SummaryMetrics | null>(null);
  const [running, setRunning] = useState(false);
  async function selectInput() {
    const selected = await open({ multiple: true, directory: false, filters: documentFilters });
    const paths = typeof selected === "string" ? [selected] : selected ?? [];
    if (paths.length === 0) return;
    setInputPaths(paths); setOutputPath((current) => current || suggestedOutput(paths[0])); setMetrics(null);
    setStatus(`${paths.length}개 문서를 하나로 요약할 준비가 됐습니다.`);
  }
  async function selectOutput() {
    const selected = await save({ defaultPath: outputPath || "summary_result.txt", filters: [{ name: "텍스트", extensions: ["txt"] }] });
    if (selected) setOutputPath(selected);
  }
  async function summarize() {
    if (inputPaths.length === 0 || !outputPath) { setStatus("입력 문서와 저장 위치를 지정하세요."); return; }
    setRunning(true); setMetrics(null); setStatus("요약 중...");
    try {
      const outcome = await invoke<SummaryOutcome>("summarize", { inputPaths, outputPath });
      setStatus(`${outcome.message}\n${outcome.output_path}`);
      setMetrics({ elapsed_ms: outcome.elapsed_ms, peak_memory_bytes: outcome.peak_memory_bytes });
    } catch (error) { setStatus(`요약에 실패했습니다. ${String(error)}`); } finally { setRunning(false); }
  }
  return <main className="app-shell">
    <section className="hero"><h1>Synthetix</h1><p>Text, Markdown, PDF를 요약합니다.</p></section>
    <section className="card form-card" aria-label="요약 설정">
      <div className="step"><span>1</span><div><h2>원본 문서</h2><p>여러 Text · Markdown · PDF를 하나로 요약합니다.</p></div></div>
      <div className="path-row"><input value={inputPaths.length ? `${inputPaths.length}개 파일 선택됨 — ${inputPaths.map((path) => path.split(/[\\/]/).slice(-1)[0]).join(", ")}` : ""} readOnly placeholder="파일을 선택하세요." /><button className="secondary" onClick={selectInput}>파일 선택</button></div>
      <div className="step"><span>2</span><div><h2>저장 위치</h2><p>텍스트 파일로 저장됩니다.</p></div></div>
      <div className="path-row"><input value={outputPath} readOnly placeholder="저장할 위치를 지정하세요." /><button className="secondary" onClick={selectOutput}>저장 위치</button></div>
      <button className="primary" onClick={summarize} disabled={running}>{running ? "요약하는 중…" : "요약 시작"}</button>
    </section>
    <section className={`status ${running ? "running" : ""}`} aria-live="polite"><strong>{running ? "처리 중" : "상태"}</strong><p>{status}</p></section>
    {metrics && <section className="metrics" aria-label="요약 성능"><span>소요 시간 <strong>{(metrics.elapsed_ms / 1000).toFixed(2)}초</strong></span><span>소요 메모리 <strong>{metrics.peak_memory_bytes > 0 ? `${(metrics.peak_memory_bytes / 1024 / 1024).toFixed(1)} MB` : "측정 불가"}</strong></span></section>}
  </main>;
}
