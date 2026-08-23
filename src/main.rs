use anyhow::{bail, Context, Result};
use clap::Parser;
use hnsw_rs::prelude::*;
use kneed::knee_locator::{InterpMethod, KneeLocator, KneeLocatorParams, ValidCurve, ValidDirection};
use rand::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[cfg(windows)]
fn peak_memory_bytes() -> Option<usize> {
    use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    unsafe {
        let mut counters: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
        counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
        (GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut counters as *mut PROCESS_MEMORY_COUNTERS_EX as *mut _,
            counters.cb,
        ) != 0)
            .then_some(counters.PeakWorkingSetSize)
    }
}

#[cfg(not(windows))]
fn peak_memory_bytes() -> Option<usize> {
    None
}

#[derive(Parser, Debug)]
#[command(name = "synthetix", about = "Synthetix — CPU 기반 Doc2Vec/HNSW 추출 요약기")]
struct Args {
    /// 입력 파일 하나 이상 (.txt, .md, .pdf). 여러 파일은 하나의 요약으로 통합한다.
    #[arg(required = true, num_args = 1..)]
    inputs: Vec<PathBuf>,

    /// 요약 결과 파일
    #[arg(short, long, default_value = "summary_result.txt")]
    output: PathBuf,

    /// Doc2Vec 차원
    #[arg(long, default_value_t = 64)]
    dim: usize,

    /// Doc2Vec 학습 횟수
    #[arg(long, default_value_t = 20)]
    epochs: usize,

    /// HNSW에서 문장마다 연결할 이웃 수
    #[arg(long, default_value_t = 8)]
    k: usize,

    /// 노드마다 생성할 Node2Vec random walk 수
    #[arg(long, default_value_t = 100)]
    walks: usize,

    /// Node2Vec random walk 길이
    #[arg(long, default_value_t = 20)]
    walk_length: usize,

    /// Doc2Vec 어휘에 포함할 최소 출현 횟수
    #[arg(long, default_value_t = 2)]
    min_count: usize,

    /// 책을 계층 요약할 때 한 번에 처리할 최대 문장 수
    #[arg(long, default_value_t = 1_200)]
    chunk_sentences: usize,

    /// sparse HDBSCAN의 최소 클러스터 크기
    #[arg(long, default_value_t = 5)]
    min_cluster_size: usize,

    /// 파일 단위 병렬 워커 수(0이면 CPU에 맞춰 최대 4개)
    #[arg(long, default_value_t = 0)]
    workers: usize,
}

#[derive(Clone, Copy)]
struct PipelineConfig {
    dim: usize,
    epochs: usize,
    k: usize,
    walks: usize,
    walk_length: usize,
    min_count: usize,
    chunk_sentences: usize,
    min_cluster_size: usize,
}

impl From<&Args> for PipelineConfig {
    fn from(args: &Args) -> Self {
        Self {
            dim: args.dim,
            epochs: args.epochs,
            k: args.k,
            walks: args.walks,
            walk_length: args.walk_length,
            min_count: args.min_count,
            chunk_sentences: args.chunk_sentences.max(20),
            min_cluster_size: args.min_cluster_size.max(2),
        }
    }
}

#[derive(Clone, Copy)]
struct Edge {
    to: usize,
    weight: f32,
}

struct Graph {
    adj: Vec<Vec<Edge>>,
}

impl Graph {
    fn new(nodes: usize) -> Self {
        Self { adj: vec![Vec::new(); nodes] }
    }

    fn add_edge(&mut self, a: usize, b: usize, weight: f32) {
        self.adj[a].push(Edge { to: b, weight });
        self.adj[b].push(Edge { to: a, weight });
    }

    fn are_neighbors(&self, a: usize, b: usize) -> bool {
        self.adj[a].iter().any(|edge| edge.to == b)
    }
}

struct Doc2Vec {
    document_vectors: Vec<Vec<f32>>,
}

#[derive(Clone, Copy)]
enum DocumentFormat {
    Text,
    Markdown,
    Pdf,
}

struct Document {
    text: String,
    format: DocumentFormat,
}

struct FileSummary {
    summary: String,
    sentence_count: usize,
    levels: usize,
}

fn main() -> Result<()> {
    let started_at = Instant::now();
    let args = Args::parse();
    let config = PipelineConfig::from(&args);
    let available = std::thread::available_parallelism().map(|count| count.get()).unwrap_or(1);
    let workers = if args.workers == 0 { available.min(4) } else { args.workers.min(available).max(1) };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build().context("병렬 요약 워커를 만들 수 없습니다")?;

    println!("[0/4] 파일 단위 병렬 전처리 및 계층 요약 (파일={}, 워커={workers})", args.inputs.len());
    let results = pool.install(|| {
        args.inputs
            .par_iter()
            .map(|path| summarize_file(path, &config).with_context(|| format!("문서를 요약할 수 없습니다: {}", path.display())))
            .collect::<Vec<_>>()
    });
    let mut file_summaries = Vec::with_capacity(results.len());
    for result in results {
        file_summaries.push(result?);
    }
    for (path, summary) in args.inputs.iter().zip(&file_summaries) {
        println!("[1/4] {}: {}문장, 계층 {}단계", path.display(), summary.sentence_count, summary.levels);
    }

    println!("[2/4] 파일별 요약본 통합 재요약");
    let summary = if file_summaries.len() == 1 {
        file_summaries.pop().unwrap().summary
    } else {
        let combined = file_summaries.into_iter().map(|item| item.summary).collect::<Vec<_>>().join("\n\n");
        let sentences = split_sentences(&combined);
        hierarchical_summarize(sentences, &config)?.0
    };
    if summary.trim().is_empty() {
        bail!("요약 결과가 비어 있습니다.");
    }

    println!("[3/4] 통합 요약 저장");
    fs::write(&args.output, &summary)
        .with_context(|| format!("출력 파일을 쓸 수 없습니다: {}", args.output.display()))?;
    println!("완료: {}", args.output.display());
    let elapsed_ms = started_at.elapsed().as_millis();
    let peak_memory_bytes = peak_memory_bytes().unwrap_or(0);
    println!("SYNTETHIX_METRICS elapsed_ms={elapsed_ms} peak_memory_bytes={peak_memory_bytes}");
    Ok(())
}

fn summarize_file(path: &Path, config: &PipelineConfig) -> Result<FileSummary> {
    let preprocessed = preprocess_document(read_document(path)?);
    let sentences = split_sentences(&preprocessed);
    if sentences.is_empty() {
        bail!("요약할 문장을 찾지 못했습니다.");
    }
    let sentence_count = sentences.len();
    let (summary, levels) = hierarchical_summarize(sentences, config)?;
    Ok(FileSummary { summary, sentence_count, levels })
}

fn hierarchical_summarize(mut sentences: Vec<String>, config: &PipelineConfig) -> Result<(String, usize)> {
    if sentences.is_empty() {
        bail!("계층 요약에 사용할 문장이 없습니다.");
    }
    let mut levels = 0usize;
    loop {
        levels += 1;
        if sentences.len() <= config.chunk_sentences {
            return Ok((summarize_chunk(&sentences, config)?, levels));
        }

        let previous_len = sentences.len();
        let chunk_summaries = sentences
            .par_chunks(config.chunk_sentences)
            .map(|chunk| summarize_chunk(chunk, config))
            .collect::<Vec<_>>();
        let mut reduced = Vec::new();
        for chunk_summary in chunk_summaries {
            reduced.extend(split_sentences(&chunk_summary?));
        }
        if reduced.is_empty() {
            bail!("계층 요약 중 모든 문장이 제거되었습니다.");
        }
        if reduced.len() >= previous_len {
            reduced = uniform_downsample(reduced, (previous_len / 2).max(1));
        }
        sentences = reduced;
    }
}

fn uniform_downsample(values: Vec<String>, target: usize) -> Vec<String> {
    if values.len() <= target {
        return values;
    }
    let len = values.len();
    let mut selected = Vec::with_capacity(target);
    for index in 0..target {
        let source = index * len / target;
        selected.push(values[source].clone());
    }
    selected
}

fn summarize_chunk(sentences: &[String], config: &PipelineConfig) -> Result<String> {
    if sentences.len() <= 2 {
        return Ok(sentences.join(" "));
    }
    let tokenized = sentences.iter().map(|sentence| tokenize(sentence)).collect::<Vec<_>>();
    let doc2vec = train_doc2vec(&tokenized, config.dim, config.epochs, config.min_count)
        .or_else(|error| {
            if config.min_count > 1 {
                train_doc2vec(&tokenized, config.dim, config.epochs, 1)
            } else {
                Err(error)
            }
        })?;
    let embeddings = normalize_matrix(doc2vec.document_vectors);
    let graph = build_hnsw_graph(&embeddings, config.k)?;
    let mapped = node2vec_2d(&graph, config.walks, config.walk_length);
    let labels = sparse_hdbscan(&graph, &mapped, config.min_cluster_size);
    let clusters = clusters_from_labels(&labels, sentences.len());
    let mut pruned = BTreeMap::new();
    for (cluster_id, nodes) in &clusters {
        pruned.insert(*cluster_id, prune_cluster(nodes, &graph));
    }
    Ok(reassemble(&pruned, sentences))
}

fn read_document(path: &Path) -> Result<Document> {
    let extension = path.extension().and_then(|value| value.to_str()).unwrap_or("").to_ascii_lowercase();
    match extension.as_str() {
        "pdf" => pdf_extract::extract_text(path)
            .map(|text| Document { text, format: DocumentFormat::Pdf })
            .with_context(|| format!("PDF 텍스트를 읽을 수 없습니다: {}", path.display())),
        "md" | "markdown" => fs::read_to_string(path)
            .map(|text| Document { text, format: DocumentFormat::Markdown })
            .with_context(|| format!("파일을 읽을 수 없습니다: {}", path.display())),
        _ => fs::read_to_string(path)
            .map(|text| Document { text, format: DocumentFormat::Text })
            .with_context(|| format!("UTF-8 텍스트를 읽을 수 없습니다: {}", path.display())),
    }
}

fn preprocess_document(document: Document) -> String {
    match document.format {
        DocumentFormat::Pdf => preprocess_pdf(&document.text),
        DocumentFormat::Markdown => rebuild_paragraphs(&strip_markdown(&document.text)),
        DocumentFormat::Text => rebuild_paragraphs(&document.text),
    }
}

fn strip_markdown(text: &str) -> String {
    let front_matter = Regex::new(r"(?s)\A---\s*\n.*?\n---\s*").unwrap();
    let code_blocks = Regex::new(r"(?s)```.*?```").unwrap();
    let images = Regex::new(r"!\[([^\]]*)\]\([^\)]*\)").unwrap();
    let links = Regex::new(r"\[([^\]]+)\]\([^\)]*\)").unwrap();
    let html = Regex::new(r"<[^>]+>").unwrap();
    let prefixes = Regex::new(r"(?m)^\s{0,3}(?:#{1,6}\s+|>\s?|[-*+]\s+|\d+[.)]\s+)").unwrap();
    let without_front_matter = front_matter.replace(text, "");
    let without_code = code_blocks.replace_all(&without_front_matter, " ");
    let with_alt_text = images.replace_all(&without_code, "$1");
    let with_link_text = links.replace_all(&with_alt_text, "$1");
    let without_html = html.replace_all(&with_link_text, " ");
    prefixes.replace_all(&without_html, "").replace(['*', '`', '~'], " ").to_string()
}

fn rebuild_paragraphs(text: &str) -> String {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n").replace('\u{00a0}', " ").replace('\u{200b}', "");
    let mut paragraphs = Vec::new();
    let mut lines = Vec::new();
    for raw_line in normalized.lines() {
        let line = raw_line.split_whitespace().collect::<Vec<_>>().join(" ");
        if line.is_empty() {
            if !lines.is_empty() {
                paragraphs.push(lines.join(" "));
                lines.clear();
            }
        } else {
            lines.push(line);
        }
    }
    if !lines.is_empty() {
        paragraphs.push(lines.join(" "));
    }
    paragraphs.join("\n\n")
}

fn preprocess_pdf(text: &str) -> String {
    let page_number = Regex::new(r"^\s*(?:-?\d{1,4}-?|제\s*\d+\s*쪽)\s*$").unwrap();
    let footnote = Regex::new(r"^\s*\*?\s*\d{1,2}\)\s*").unwrap();
    let normalized_lines = text.replace("\r\n", "\n").replace('\r', "\n").replace('\u{00ad}', "").replace('\u{00a0}', " ")
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    let mut occurrences = HashMap::new();
    for line in &normalized_lines {
        if line.chars().count() <= 100 {
            *occurrences.entry(line.clone()).or_insert(0usize) += 1;
        }
    }
    let mut restored = String::new();
    for line in normalized_lines {
        let alphanumeric = line.chars().filter(|character| character.is_alphanumeric()).count();
        let repeated_header = occurrences.get(&line).copied().unwrap_or(0) >= 3;
        if alphanumeric < 3 || page_number.is_match(&line) || footnote.is_match(&line) || repeated_header {
            continue;
        }
        if restored.is_empty() {
            restored.push_str(&line);
        } else if restored.ends_with('-') {
            restored.pop();
            restored.push_str(line.trim_start());
        } else {
            restored.push(' ');
            restored.push_str(&line);
        }
    }
    rebuild_paragraphs(&restored)
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    for paragraph in text.split("\n\n") {
        let mut current = String::new();
        let chars = paragraph.chars().collect::<Vec<_>>();
        for (index, ch) in chars.iter().copied().enumerate() {
            current.push(ch);
            if is_sentence_boundary(&chars, index) {
                push_sentence(&mut sentences, &current);
                current.clear();
            }
        }
        push_sentence(&mut sentences, &current);
    }
    sentences
}

fn is_sentence_boundary(chars: &[char], index: usize) -> bool {
    let ch = chars[index];
    if !matches!(ch, '.' | '!' | '?' | '。' | '！' | '？') {
        return false;
    }

    // 합병비율(1:0.3500885), 날짜(2015. 7. 17.)처럼 숫자 안의 마침표는
    // 문장 경계가 아니다. PDF 추출문에서 이 지점이 끊기면 사람용 문단이
    // 부자연스러워지므로, 다음 의미 있는 문자가 숫자인 경우에만 이어 붙인다.
    if ch == '.' && index > 0 && chars[index - 1].is_ascii_digit() {
        if let Some(next) = chars[index + 1..].iter().find(|next| !next.is_whitespace()) {
            if next.is_ascii_digit() {
                return false;
            }
        }
    }

    true
}

fn push_sentence(sentences: &mut Vec<String>, raw: &str) {
    let sentence = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let alphanumeric = sentence.chars().filter(|character| character.is_alphanumeric()).count();
    let starts_as_fragment = sentence.starts_with(')') || sentence.starts_with(',') || sentence.starts_with(';');
    if sentence.chars().count() >= 15 && alphanumeric >= 8 && !starts_as_fragment {
        sentences.push(sentence);
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_alphanumeric() && ch != '가' && ch != '각')
        .filter_map(|word| {
            let clean = word.trim().to_lowercase();
            (clean.chars().count() > 1).then_some(clean)
        })
        .collect()
}

fn train_doc2vec(documents: &[Vec<String>], dim: usize, epochs: usize, min_count: usize) -> Result<Doc2Vec> {
    if dim == 0 {
        bail!("--dim은 1 이상이어야 합니다.");
    }
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for document in documents {
        for token in document {
            *counts.entry(token).or_default() += 1;
        }
    }
    let mut vocabulary: Vec<(&str, usize)> = counts.into_iter().filter(|(_, count)| *count >= min_count).collect();
    vocabulary.sort_unstable_by(|a, b| a.0.cmp(b.0));
    if vocabulary.is_empty() {
        bail!("최소 빈도 조건을 만족하는 단어가 없습니다. --min-count를 낮춰보세요.");
    }
    let word_to_id: HashMap<&str, usize> = vocabulary.iter().enumerate().map(|(id, (word, _))| (*word, id)).collect();
    let corpus: Vec<Vec<usize>> = documents.iter().map(|document| document.iter().filter_map(|word| word_to_id.get(word.as_str()).copied()).collect()).collect();

    let mut rng = StdRng::seed_from_u64(42);
    let mut document_vectors = (0..documents.len()).map(|_| random_vector(dim, &mut rng)).collect::<Vec<_>>();
    let mut word_vectors = (0..vocabulary.len()).map(|_| random_vector(dim, &mut rng)).collect::<Vec<_>>();
    let mut output_vectors = (0..vocabulary.len()).map(|_| vec![0.0; dim]).collect::<Vec<_>>();
    let negative_table = build_negative_table(&vocabulary);
    let mut order: Vec<usize> = (0..corpus.len()).collect();

    for epoch in 0..epochs.max(1) {
        order.shuffle(&mut rng);
        let learning_rate = 0.04 * (1.0 - epoch as f32 / epochs.max(1) as f32).max(0.1);
        for &document_id in &order {
            for position in 0..corpus[document_id].len() {
                let word_id = corpus[document_id][position];
                let context = context_window(&corpus[document_id], position, 2);
                if context.is_empty() {
                    continue;
                }
                update_pv_dm(
                    &mut document_vectors[document_id],
                    &mut word_vectors,
                    &context,
                    &mut output_vectors[word_id],
                    1.0,
                    learning_rate,
                );
                for _ in 0..5 {
                    let negative = negative_table[rng.gen_range(0..negative_table.len())];
                    if negative != word_id {
                        update_pv_dm(
                            &mut document_vectors[document_id],
                            &mut word_vectors,
                            &context,
                            &mut output_vectors[negative],
                            0.0,
                            learning_rate,
                        );
                    }
                }
            }
        }
    }
    Ok(Doc2Vec { document_vectors })
}

fn random_vector(dim: usize, rng: &mut StdRng) -> Vec<f32> {
    (0..dim).map(|_| rng.gen_range(-0.5..0.5) / dim as f32).collect()
}

fn build_negative_table(vocabulary: &[(&str, usize)]) -> Vec<usize> {
    let table_size = 50_000usize;
    let total: f32 = vocabulary.iter().map(|(_, count)| (*count as f32).powf(0.75)).sum();
    let mut table = Vec::with_capacity(table_size);
    let mut index = 0usize;
    let mut cumulative = (vocabulary[0].1 as f32).powf(0.75) / total;
    for slot in 0..table_size {
        let ratio = slot as f32 / table_size as f32;
        while ratio > cumulative && index + 1 < vocabulary.len() {
            index += 1;
            cumulative += (vocabulary[index].1 as f32).powf(0.75) / total;
        }
        table.push(index);
    }
    table
}

fn context_window(document: &[usize], position: usize, radius: usize) -> Vec<usize> {
    let start = position.saturating_sub(radius);
    let end = (position + radius + 1).min(document.len());
    (start..end).filter(|&index| index != position).map(|index| document[index]).collect()
}

/// PV-DM: 문서 벡터와 주변 단어 벡터의 평균으로 중심 단어를 예측한다.
fn update_pv_dm(document: &mut [f32], words: &mut [Vec<f32>], context: &[usize], output: &mut [f32], label: f32, learning_rate: f32) {
    let divisor = (context.len() + 1) as f32;
    let mut hidden = document.to_vec();
    for &word_id in context {
        for (hidden_value, word_value) in hidden.iter_mut().zip(&words[word_id]) {
            *hidden_value += word_value;
        }
    }
    for value in &mut hidden {
        *value /= divisor;
    }
    let score: f32 = hidden.iter().zip(output.iter()).map(|(left, right)| left * right).sum();
    let probability = 1.0 / (1.0 + (-score.clamp(-12.0, 12.0)).exp());
    let gradient = (label - probability) * learning_rate;
    for index in 0..document.len() {
        let old_output = output[index];
        output[index] += gradient * hidden[index];
        let input_gradient = gradient * old_output / divisor;
        document[index] += input_gradient;
        for &word_id in context {
            words[word_id][index] += input_gradient;
        }
    }
}

fn normalize_matrix(matrix: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    matrix.into_iter().map(normalize).collect()
}

fn normalize(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

fn build_hnsw_graph(embeddings: &[Vec<f32>], requested_k: usize) -> Result<Graph> {
    if embeddings.len() <= 1 {
        return Ok(Graph::new(embeddings.len()));
    }
    let dim = embeddings[0].len();
    if dim == 0 || embeddings.iter().any(|embedding| embedding.len() != dim) {
        bail!("문장 임베딩 차원이 일관되지 않습니다.");
    }
    let hnsw = Hnsw::<f32, DistCosine>::new(16, embeddings.len(), 16, 200, DistCosine {});
    for (id, embedding) in embeddings.iter().enumerate() {
        hnsw.insert((embedding, id));
    }

    let mut graph = Graph::new(embeddings.len());
    let mut seen = HashSet::new();
    let k = requested_k.clamp(1, embeddings.len() - 1);
    for (source, embedding) in embeddings.iter().enumerate() {
        for neighbor in hnsw.search(embedding, k + 1, 64) {
            let target = neighbor.d_id;
            if source == target {
                continue;
            }
            let pair = (source.min(target), source.max(target));
            if seen.insert(pair) {
                let weight = (-neighbor.distance.powi(2)).exp();
                graph.add_edge(source, target, weight);
            }
        }
    }
    Ok(graph)
}

fn node2vec_2d(graph: &Graph, walks_per_node: usize, walk_length: usize) -> Vec<[f32; 2]> {
    let nodes = graph.adj.len();
    if nodes == 0 {
        return Vec::new();
    }
    let mut rng = StdRng::seed_from_u64(7);
    let mut inputs = (0..nodes).map(|_| [rng.gen_range(-0.1..0.1), rng.gen_range(-0.1..0.1)]).collect::<Vec<_>>();
    let mut outputs = vec![[0.0; 2]; nodes];
    let walks = walks_per_node.max(1);
    for start in 0..nodes {
        for _ in 0..walks {
            let walk = sample_walk(graph, start, walk_length.max(2), &mut rng);
            for position in 0..walk.len() {
                let from = walk[position];
                let left = position.saturating_sub(2);
                let right = (position + 3).min(walk.len());
                for context_position in left..right {
                    if context_position == position {
                        continue;
                    }
                    update_pair_2d(&mut inputs[from], &mut outputs[walk[context_position]], 1.0, 0.03);
                    for _ in 0..3 {
                        let negative = rng.gen_range(0..nodes);
                        if negative != walk[context_position] {
                            update_pair_2d(&mut inputs[from], &mut outputs[negative], 0.0, 0.03);
                        }
                    }
                }
            }
        }
    }
    standardize_2d(inputs)
}

fn sample_walk(graph: &Graph, start: usize, max_length: usize, rng: &mut StdRng) -> Vec<usize> {
    let mut walk = vec![start];
    let mut visited = vec![false; graph.adj.len()];
    visited[start] = true;
    while walk.len() < max_length {
        let current = *walk.last().unwrap();
        if graph.adj[current].is_empty() {
            break;
        }
        let previous = if walk.len() > 1 {
            Some(walk[walk.len() - 2])
        } else {
            None
        };
        let next = weighted_neighbor(graph, current, previous, rng);
        if visited[next] {
            break;
        }
        visited[next] = true;
        walk.push(next);
    }
    walk
}

fn weighted_neighbor(graph: &Graph, current: usize, previous: Option<usize>, rng: &mut StdRng) -> usize {
    let candidates = &graph.adj[current];
    let total: f32 = candidates.iter().map(|edge| {
        let bias = match previous {
            Some(previous_node) if edge.to == previous_node => 1.0,
            Some(previous_node) if graph.are_neighbors(previous_node, edge.to) => 1.0,
            Some(_) => 2.0,
            None => 1.0,
        };
        edge.weight * bias
    }).sum();
    let mut pick = rng.gen_range(0.0..total.max(f32::EPSILON));
    for edge in candidates {
        let bias = match previous {
            Some(previous_node) if edge.to == previous_node || graph.are_neighbors(previous_node, edge.to) => 1.0,
            Some(_) => 2.0,
            None => 1.0,
        };
        pick -= edge.weight * bias;
        if pick <= 0.0 {
            return edge.to;
        }
    }
    candidates.last().unwrap().to
}

fn update_pair_2d(input: &mut [f32; 2], output: &mut [f32; 2], label: f32, learning_rate: f32) {
    let score = input[0] * output[0] + input[1] * output[1];
    let probability = 1.0 / (1.0 + (-score.clamp(-12.0, 12.0)).exp());
    let gradient = (label - probability) * learning_rate;
    for index in 0..2 {
        let old = input[index];
        input[index] += gradient * output[index];
        output[index] += gradient * old;
    }
}

fn standardize_2d(mut values: Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    if values.len() <= 1 {
        return values;
    }
    let mean = [values.iter().map(|point| point[0]).sum::<f32>() / values.len() as f32, values.iter().map(|point| point[1]).sum::<f32>() / values.len() as f32];
    let deviation = [
        (values.iter().map(|point| (point[0] - mean[0]).powi(2)).sum::<f32>() / values.len() as f32).sqrt().max(1e-6),
        (values.iter().map(|point| (point[1] - mean[1]).powi(2)).sum::<f32>() / values.len() as f32).sqrt().max(1e-6),
    ];
    for point in &mut values {
        point[0] = (point[0] - mean[0]) / deviation[0];
        point[1] = (point[1] - mean[1]) / deviation[1];
    }
    values
}

#[derive(Clone, Copy)]
struct SparseDistanceEdge {
    left: usize,
    right: usize,
    distance: f32,
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self { parent: (0..size).collect(), rank: vec![0; size] }
    }

    fn find(&mut self, node: usize) -> usize {
        if self.parent[node] != node {
            self.parent[node] = self.find(self.parent[node]);
        }
        self.parent[node]
    }

    fn union(&mut self, left: usize, right: usize) -> usize {
        let mut left_root = self.find(left);
        let mut right_root = self.find(right);
        if left_root == right_root {
            return left_root;
        }
        if self.rank[left_root] < self.rank[right_root] {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.parent[right_root] = left_root;
        if self.rank[left_root] == self.rank[right_root] {
            self.rank[left_root] += 1;
        }
        left_root
    }
}

struct DensityTreeNode {
    children: Option<(usize, usize)>,
    parent: Option<usize>,
    size: usize,
    merge_distance: f32,
}

/// HNSW의 희소 간선에서만 mutual-reachability MST와 밀도 계층을 만든다.
/// 정확한 전점 HDBSCAN 대신 sparse k-NN 근사를 사용하므로 시간은 O(E log E),
/// 메모리는 O(N + E)이며 E는 HNSW 이웃 수에 비례한다.
fn sparse_hdbscan(graph: &Graph, points: &[[f32; 2]], min_cluster_size: usize) -> Vec<i32> {
    let node_count = graph.adj.len();
    if node_count < min_cluster_size.max(2) || points.len() != node_count {
        return vec![0; node_count];
    }

    let mut raw_edges = Vec::new();
    let mut neighbor_distances = vec![Vec::new(); node_count];
    for left in 0..node_count {
        for edge in &graph.adj[left] {
            if left >= edge.to {
                continue;
            }
            let dx = points[left][0] - points[edge.to][0];
            let dy = points[left][1] - points[edge.to][1];
            let distance = (dx * dx + dy * dy).sqrt().max(1e-6);
            raw_edges.push(SparseDistanceEdge { left, right: edge.to, distance });
            neighbor_distances[left].push(distance);
            neighbor_distances[edge.to].push(distance);
        }
    }
    if raw_edges.is_empty() {
        return vec![0; node_count];
    }

    let min_samples = min_cluster_size.max(2);
    let mut core_distances = vec![f32::INFINITY; node_count];
    for (node, distances) in neighbor_distances.iter_mut().enumerate() {
        if distances.is_empty() {
            continue;
        }
        distances.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
        core_distances[node] = distances[(min_samples - 1).min(distances.len() - 1)];
    }
    for edge in &mut raw_edges {
        edge.distance = edge.distance.max(core_distances[edge.left]).max(core_distances[edge.right]);
    }
    raw_edges.sort_by(|left, right| left.distance.partial_cmp(&right.distance).unwrap_or(Ordering::Equal));

    let mut tree = (0..node_count)
        .map(|_| DensityTreeNode { children: None, parent: None, size: 1, merge_distance: 0.0 })
        .collect::<Vec<_>>();
    let mut union_find = UnionFind::new(node_count);
    let mut component_tree_root = (0..node_count).collect::<Vec<_>>();
    for edge in raw_edges {
        let left_root = union_find.find(edge.left);
        let right_root = union_find.find(edge.right);
        if left_root == right_root {
            continue;
        }
        let left_tree = component_tree_root[left_root];
        let right_tree = component_tree_root[right_root];
        let parent_id = tree.len();
        tree.push(DensityTreeNode {
            children: Some((left_tree, right_tree)),
            parent: None,
            size: tree[left_tree].size + tree[right_tree].size,
            merge_distance: edge.distance.max(1e-6),
        });
        tree[left_tree].parent = Some(parent_id);
        tree[right_tree].parent = Some(parent_id);
        let merged_root = union_find.union(left_root, right_root);
        component_tree_root[merged_root] = parent_id;
    }

    let mut forest_roots = Vec::new();
    let mut seen_roots = HashSet::new();
    for node in 0..node_count {
        let root = union_find.find(node);
        if seen_roots.insert(root) {
            forest_roots.push(component_tree_root[root]);
        }
    }

    let mut selected = Vec::new();
    let multiple_components = forest_roots.len() > 1;
    for root in forest_roots {
        let (_, mut root_selection) = select_density_clusters(root, &tree, min_cluster_size.max(2), multiple_components);
        selected.append(&mut root_selection);
    }
    if selected.is_empty() {
        return vec![0; node_count];
    }

    selected.sort_unstable_by_key(|cluster| minimum_leaf(*cluster, &tree));
    let mut labels = vec![-1; node_count];
    for (label, cluster) in selected.into_iter().enumerate() {
        assign_cluster_leaves(cluster, label as i32, &tree, &mut labels);
    }
    labels
}

fn select_density_clusters(node: usize, tree: &[DensityTreeNode], min_cluster_size: usize, allow_root: bool) -> (f64, Vec<usize>) {
    if tree[node].size < min_cluster_size {
        return (0.0, Vec::new());
    }
    let birth_lambda = tree[node]
        .parent
        .map(|parent| 1.0 / tree[parent].merge_distance.max(1e-6) as f64)
        .unwrap_or(0.0);
    let death_lambda = if tree[node].merge_distance > 0.0 { 1.0 / tree[node].merge_distance as f64 } else { birth_lambda };
    let self_stability = tree[node].size as f64 * (death_lambda - birth_lambda).max(0.0);

    let mut child_stability = 0.0;
    let mut child_selection = Vec::new();
    if let Some((left, right)) = tree[node].children {
        for child in [left, right] {
            let (score, mut selection) = select_density_clusters(child, tree, min_cluster_size, true);
            child_stability += score;
            child_selection.append(&mut selection);
        }
    }
    if child_selection.is_empty() {
        return (self_stability, vec![node]);
    }
    if allow_root && self_stability >= child_stability {
        (self_stability, vec![node])
    } else {
        (child_stability, child_selection)
    }
}

fn minimum_leaf(node: usize, tree: &[DensityTreeNode]) -> usize {
    match tree[node].children {
        Some((left, right)) => minimum_leaf(left, tree).min(minimum_leaf(right, tree)),
        None => node,
    }
}

fn assign_cluster_leaves(node: usize, label: i32, tree: &[DensityTreeNode], labels: &mut [i32]) {
    match tree[node].children {
        Some((left, right)) => {
            assign_cluster_leaves(left, label, tree, labels);
            assign_cluster_leaves(right, label, tree, labels);
        }
        None => labels[node] = label,
    }
}

fn clusters_from_labels(labels: &[i32], total_nodes: usize) -> BTreeMap<i32, Vec<usize>> {
    let mut clusters = BTreeMap::new();
    for (node, &label) in labels.iter().enumerate() {
        if label >= 0 {
            clusters.entry(label).or_insert_with(Vec::new).push(node);
        }
    }
    if clusters.is_empty() {
        clusters.insert(0, (0..total_nodes).collect());
    }
    clusters
}

fn prune_cluster(nodes: &[usize], graph: &Graph) -> Vec<usize> {
    if nodes.len() <= 2 {
        return nodes.to_vec();
    }
    let count = nodes.len();
    let local_index = nodes.iter().enumerate().map(|(index, node)| (*node, index)).collect::<HashMap<_, _>>();
    let mut outgoing = vec![0.0f32; count];
    for (local, node) in nodes.iter().copied().enumerate() {
        outgoing[local] = graph.adj[node]
            .iter()
            .filter(|edge| local_index.contains_key(&edge.to))
            .map(|edge| edge.weight.max(0.0))
            .sum();
    }
    let mut ranks = vec![1.0 / count as f32; count];
    for _ in 0..30 {
        let dangling = outgoing
            .iter()
            .zip(&ranks)
            .filter(|(weight, _)| **weight <= f32::EPSILON)
            .map(|(_, rank)| *rank)
            .sum::<f32>();
        let base = (1.0 - 0.85) / count as f32 + 0.85 * dangling / count as f32;
        let mut next = vec![base; count];
        for (source, node) in nodes.iter().copied().enumerate() {
            if outgoing[source] <= f32::EPSILON {
                continue;
            }
            for edge in &graph.adj[node] {
                if let Some(&target) = local_index.get(&edge.to) {
                    next[target] += 0.85 * ranks[source] * edge.weight.max(0.0) / outgoing[source];
                }
            }
        }
        ranks = next;
    }
    let mut ranked: Vec<(usize, f32)> = nodes.iter().copied().zip(ranks).collect();
    ranked.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
    let elbow_keep = elbow_count(&ranked.iter().map(|(_, value)| *value).collect::<Vec<_>>());
    // 작은 군집은 한 문장, 충분히 큰 군집은 두 문장을 보존해 쟁점 손실을 막는다.
    let minimum_keep = if nodes.len() >= 6 { 2 } else { 1 };
    let keep = elbow_keep.max(minimum_keep).min(2).min(nodes.len());
    ranked.into_iter().take(keep).map(|(node, _)| node).collect()
}

fn elbow_count(values: &[f32]) -> usize {
    if values.len() <= 2 {
        return values.len();
    }
    let indices = (0..values.len()).map(|index| index as f64).collect::<Vec<_>>();
    let scores = values.iter().map(|value| *value as f64).collect::<Vec<_>>();
    let interpolation = 0.01;
    let convex = KneeLocatorParams::new(ValidCurve::Convex, ValidDirection::Decreasing, InterpMethod::Interp1d);
    let concave = KneeLocatorParams::new(ValidCurve::Concave, ValidDirection::Decreasing, InterpMethod::Interp1d);
    let elbows = [convex, concave]
        .into_iter()
        .filter_map(|params| KneeLocator::new(indices.clone(), scores.clone(), interpolation, params).ok().and_then(|locator| locator.norm_elbow()))
        .filter(|elbow| elbow.is_finite() && *elbow >= 0.0)
        .map(|elbow| (elbow as usize).min(values.len() - 1))
        .collect::<Vec<_>>();
    let minimal_elbow = elbows.into_iter().min().unwrap_or(values.len() - 1);
    if minimal_elbow == values.len() - 1 {
        ((values.len() as f32 * 0.25).floor() as usize).max(1)
    } else {
        minimal_elbow.max(1)
    }
}

fn reassemble(clusters: &BTreeMap<i32, Vec<usize>>, sentences: &[String]) -> String {
    if clusters.is_empty() {
        return String::new();
    }
    let mut selected_nodes = clusters.values().flatten().copied().collect::<Vec<_>>();
    selected_nodes.sort_unstable();
    selected_nodes.dedup();

    // 추출 요약은 원문 순서가 가장 자연스럽다. 선택 문장이 충분할 때만 2~3문단으로 나눈다.
    let paragraph_count = if selected_nodes.len() >= 4 { clusters.len().clamp(2, 3) } else { 1 };
    let selected_count = selected_nodes.len();
    let mut paragraphs = vec![Vec::new(); paragraph_count];
    for (position, node) in selected_nodes.into_iter().enumerate() {
        let paragraph = position * paragraph_count / selected_count;
        paragraphs[paragraph].push(sentences[node].trim());
    }
    paragraphs
        .into_iter()
        .map(|paragraph| paragraph.into_iter().filter(|sentence| !sentence.is_empty()).collect::<Vec<_>>().join(" "))
        .filter(|paragraph| !paragraph.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_hdbscan_separates_two_density_groups() {
        let mut graph = Graph::new(12);
        for group_start in [0usize, 6usize] {
            for left in group_start..group_start + 6 {
                for right in left + 1..group_start + 6 {
                    graph.add_edge(left, right, 1.0);
                }
            }
        }
        graph.add_edge(5, 6, 0.001);
        let points = (0..6)
            .map(|index| [index as f32 * 0.02, 0.0])
            .chain((0..6).map(|index| [10.0 + index as f32 * 0.02, 0.0]))
            .collect::<Vec<_>>();
        let labels = sparse_hdbscan(&graph, &points, 3);
        let first = labels[0];
        let second = labels[6];
        assert!(first >= 0 && second >= 0);
        assert_ne!(first, second);
        assert!(labels[..6].iter().all(|label| *label == first));
        assert!(labels[6..].iter().all(|label| *label == second));
    }

    #[test]
    fn sparse_pagerank_keeps_star_center() {
        let mut graph = Graph::new(6);
        for leaf in 1..6 {
            graph.add_edge(0, leaf, 1.0);
        }
        let selected = prune_cluster(&(0..6).collect::<Vec<_>>(), &graph);
        assert!(selected.contains(&0));
        assert!((1..=2).contains(&selected.len()));
    }

    #[test]
    fn hierarchy_reduces_large_input() {
        let sentences = (0..60)
            .map(|index| format!("공통 주제와 핵심 쟁점을 설명하는 충분히 긴 시험 문장 {index}입니다."))
            .collect::<Vec<_>>();
        let config = PipelineConfig {
            dim: 8,
            epochs: 1,
            k: 4,
            walks: 2,
            walk_length: 6,
            min_count: 1,
            chunk_sentences: 20,
            min_cluster_size: 3,
        };
        let (summary, levels) = hierarchical_summarize(sentences.clone(), &config).unwrap();
        assert!(levels >= 2);
        assert!(!summary.is_empty());
        assert!(split_sentences(&summary).len() < sentences.len());
    }
}
