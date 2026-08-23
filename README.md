# Synthetix

GPU나 외부 LLM API 없이, 입력 문서에서 중요한 문장을 추출한다.

## 파이프라인

1. 형식별 전처리와 한국어 문장 분할·토큰화
2. 문서 벡터와 문맥 단어를 함께 학습하는 PV-DM 방식의 경량 Doc2Vec 학습
3. HNSW 기반 근사 최근접 문장 그래프 생성
4. 노드당 100회 random walk를 사용하는 Node2Vec 2차원 매핑과 HNSW 간선 기반 sparse HDBSCAN 분할
5. HNSW 희소 간선만 사용하는 클러스터별 가중 PageRank 및 양방향 Kneedle elbow 지점까지 문장 보존
6. 각 HDBSCAN 클러스터에서 1~2문장을 보존하고, 원문 순서로 2~3개 사람용 문단 재배열
7. 큰 책은 문장 청크별로 병렬 요약하고, 여러 파일은 파일별 요약본을 다시 통합 요약

문장 임베딩과 Node2Vec은 모두 이 저장소 안에서 학습한다. 따라서 트랜스포머 모델 다운로드나 GPU가 필요 없다. HDBSCAN과 PageRank도 완전 거리 행렬을 만들지 않으며, 기본 1,200문장 청크와 제한된 워커 풀로 메모리 사용량을 제한한다. 핵심 그래프 단계는 실질적으로 O(N log N)에 가깝다.

## 실행

Rust 1.75 이상이 필요하다.

### React + Tauri Windows UI

`Synthetix-Portable.exe` 하나만 복사해 실행하면 된다. 요약 엔진은 실행 파일 안에 포함되어 있으며, 설치 마법사·레지스트리·별도 엔진 파일이 필요 없다. React 기반 화면에서 여러 문서를 선택하고 저장 위치를 한 번 지정하면, 파일·책 단위 요약을 병렬로 만든 뒤 요약본을 재요약해 **하나의 요약 파일**을 만든다.

UI를 개발 모드로 실행하려면 Node.js와 Rust를 설치한 뒤, 먼저 Rust 요약 엔진을 만든다. `Synthetix-engine.exe`는 포터블 앱에 내장되는 빌드 중간 파일이므로 GitHub에는 올리지 않는다.

```powershell
cd C:\경로\Synthetix
cargo build --release --bin synthetix
Copy-Item .\target\release\synthetix.exe .\Synthetix-engine.exe -Force
cd tauri-ui
npm.cmd install
npm.cmd run tauri -- dev
```

단일 포터블 EXE를 다시 만들려면 `tauri-ui` 폴더에서 다음을 실행한다.

```powershell
$env:CARGO_TARGET_DIR = "$PWD\src-tauri\target-portable"
npm.cmd run tauri -- build --no-bundle
Copy-Item .\src-tauri\target-portable\release\synthetix.exe ..\Synthetix-Portable.exe -Force
```

주요 조절값:

```text
--dim 64          Doc2Vec 차원 (기본 64)
--epochs 20       Doc2Vec 학습 epoch (기본 20)
--k 8             HNSW 이웃 수 (기본 8)
--walks 100       노드별 Node2Vec walk 수 (기본 100)
--walk-length 20  walk 길이 (기본 20)
--min-count 2     Doc2Vec 어휘 최소 빈도
--chunk-sentences 1200  계층 요약 청크 크기
--min-cluster-size 5    sparse HDBSCAN 최소 군집 크기
--workers 0       병렬 워커 수 (0이면 CPU에 맞춰 최대 4개)
```

품질을 우선할 때는 `--epochs 100`을 줄 수 있지만, 처리 시간은 입력 토큰 수와 함께 증가한다.

여러 문서를 하나로 통합하려면 입력 파일을 나란히 지정한다. 각 파일은 먼저 독립적으로 요약되며, 마지막에 파일별 요약본만 다시 분석하므로 전체 원문에 대한 완전 거리 행렬은 만들지 않는다.

```powershell
cargo run --release -- report-a.pdf report-b.md notes.txt -o 통합_요약.txt
```

## 입력 형식

`.txt`, `.md`, 텍스트 레이어가 있는 `.pdf`를 지원한다. 스캔본 PDF는 OCR을 거친 뒤 입력해야 한다.

입력은 요약 전에 다음처럼 정제한다.

- 공통: UTF-8·공백 정규화, 줄바꿈으로 분리된 문단 복원, 지나치게 짧은 문장 조각 제외
- Markdown: YAML front matter, 코드 블록, 이미지·링크 문법, 목록·인용 표식을 본문에서 분리
- PDF: 줄바꿈으로 나뉜 본문 결합, 단어 끝 하이픈 복원, 쪽수·각주 표식·반복 머리말/꼬리말 제거

PDF 원문이 스캔 이미지이거나 글꼴 인코딩이 손상된 경우에는 텍스트 추출 품질 자체를 보장할 수 없다.
