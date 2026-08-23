#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use serde::Serialize;
use tauri::AppHandle;

#[derive(Serialize)] struct SummaryOutcome {
    output_path: String,
    message: String,
    elapsed_ms: u128,
    peak_memory_bytes: u64,
}

fn parse_engine_metrics(stdout: &[u8]) -> Option<(u128, u64)> {
    let stdout = String::from_utf8_lossy(stdout);
    let line = stdout
        .lines()
        .find(|line| line.starts_with("SYNTETHIX_METRICS "))?;
    let mut elapsed_ms = None;
    let mut peak_memory_bytes = None;
    for field in line.split_whitespace().skip(1) {
        let (name, value) = field.split_once('=')?;
        match name {
            "elapsed_ms" => elapsed_ms = value.parse().ok(),
            "peak_memory_bytes" => peak_memory_bytes = value.parse().ok(),
            _ => {}
        }
    }
    Some((elapsed_ms?, peak_memory_bytes?))
}
// 배포본에는 UI EXE 하나만 둔다. 엔진은 컴파일 시 이 실행 파일 안에 포함된다.
const EMBEDDED_ENGINE: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Synthetix-engine.exe"));

fn extract_embedded_engine() -> Result<PathBuf, String> {
    let directory = std::env::temp_dir().join("Synthetix");
    fs::create_dir_all(&directory).map_err(|error| format!("임시 엔진 폴더를 만들지 못했습니다: {error}"))?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("임시 파일 이름을 만들지 못했습니다: {error}"))?
        .as_nanos();
    let path = directory.join(format!("synthetix-engine-{}-{nonce}.exe", std::process::id()));
    fs::write(&path, EMBEDDED_ENGINE).map_err(|error| format!("내장 요약 엔진을 준비하지 못했습니다: {error}"))?;
    Ok(path)
}
#[tauri::command]
fn summarize(_app: AppHandle, input_paths: Vec<String>, output_path: String) -> Result<SummaryOutcome, String> {
    if input_paths.is_empty() { return Err("통합할 입력 파일을 하나 이상 선택하세요.".to_string()); }
    let engine_path = extract_embedded_engine()?;
    let started_at = Instant::now();
    let output = (|| {
        let mut command = Command::new(&engine_path);
        #[cfg(windows)]
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW: GUI 뒤에 콘솔 창을 띄우지 않는다.
        command.args(&input_paths).args(["-o", &output_path]);
        command.output().map_err(|error| format!("요약 엔진을 실행하지 못했습니다: {error}"))
    })();
    let _ = fs::remove_file(&engine_path);
    let output = output?;
    if !output.status.success() { let detail = String::from_utf8_lossy(&output.stderr).trim().to_string(); return Err(if detail.is_empty() { "요약 엔진이 정상적으로 종료되지 않았습니다.".to_string() } else { format!("요약 엔진 오류: {detail}") }); }
    let (elapsed_ms, peak_memory_bytes) = parse_engine_metrics(&output.stdout)
        .unwrap_or((started_at.elapsed().as_millis(), 0));
    Ok(SummaryOutcome {
        output_path,
        message: format!("{}개 문서를 통합해 요약 파일을 만들었습니다.", input_paths.len()),
        elapsed_ms,
        peak_memory_bytes,
    })
}
fn main() { tauri::Builder::default().plugin(tauri_plugin_dialog::init()).invoke_handler(tauri::generate_handler![summarize]).run(tauri::generate_context!()).expect("Tauri 앱을 시작하지 못했습니다"); }
