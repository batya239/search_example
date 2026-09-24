use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use ignore::overrides::OverrideBuilder;
use ignore::{DirEntry, WalkBuilder, WalkState};

use crate::embeddings::EmbeddingModel;
use crate::extract;
use crate::languages::{self, Grammar};
use crate::store;

/// Entries accumulate in a thread-local buffer before being sorted and
/// spilled to disk; this keeps per-thread memory bounded without spilling
/// on every single file.
const SPILL_THRESHOLD: usize = 200_000;

pub struct RunStats {
    pub files_indexed: usize,
    pub files_skipped: usize,
}

/// One worker thread's contribution: the spill files it wrote, the
/// `(file_id, relative_path, doc_length)` triples for every file it
/// indexed, and -- when embeddings are enabled -- the `(file_id, vector)`
/// pair for each of those files too.
struct ThreadOutput {
    spill_paths: Vec<PathBuf>,
    docs: Vec<(u64, String, u32)>,
    embeddings: Vec<(u64, Vec<f32>)>,
}

/// Per-thread state for one `WalkParallel` visitor. `ignore` builds exactly
/// one visitor per worker thread (its builder closure is called once per
/// thread, not once per file), so this lives for that thread's entire
/// share of the walk. Its `Drop` impl is the only place that touches
/// shared state, and it does so once per thread rather than once per file.
struct ThreadState<'a> {
    root: &'a Path,
    spill_dir: &'a Path,
    spill_seq: &'a AtomicU64,
    file_id_counter: &'a AtomicU64,
    indexed: &'a AtomicUsize,
    skipped: &'a AtomicUsize,
    outputs: &'a Mutex<Vec<ThreadOutput>>,
    embedding_model: Option<&'a EmbeddingModel>,
    buffer: Vec<(String, u64, u32)>,
    docs: Vec<(u64, String, u32)>,
    embeddings: Vec<(u64, Vec<f32>)>,
    spill_paths: Vec<PathBuf>,
}

impl ThreadState<'_> {
    fn record_file(&mut self, terms: BTreeMap<String, u32>, rel_path: String, embedding: Option<Vec<f32>>) {
        let file_id = self.file_id_counter.fetch_add(1, Ordering::Relaxed);
        let doc_length: u32 = terms.values().sum();
        self.docs.push((file_id, rel_path, doc_length));
        if let Some(vector) = embedding {
            self.embeddings.push((file_id, vector));
        }
        self.buffer
            .extend(terms.into_iter().map(|(term, freq)| (term, file_id, freq)));
        if self.buffer.len() >= SPILL_THRESHOLD {
            self.flush_buffer();
        }
    }

    fn flush_buffer(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        self.buffer.sort_unstable();
        let seq = self.spill_seq.fetch_add(1, Ordering::Relaxed);
        let path = self.spill_dir.join(format!("{seq:010}.spill"));
        match store::write_spill_file(&path, &self.buffer) {
            Ok(()) => self.spill_paths.push(path),
            Err(e) => eprintln!("warning: failed to write spill file {}: {e}", path.display()),
        }
        self.buffer.clear();
    }
}

impl Drop for ThreadState<'_> {
    fn drop(&mut self) {
        self.flush_buffer();
        let output = ThreadOutput {
            spill_paths: std::mem::take(&mut self.spill_paths),
            docs: std::mem::take(&mut self.docs),
            embeddings: std::mem::take(&mut self.embeddings),
        };
        self.outputs.lock().unwrap().push(output);
    }
}

/// Indexes every file under `root` into the project index directory at
/// `project_dir`, respecting `excludes` (directory/file name globs,
/// gitignore-style). When `embedding_model` is `Some`, an `embeddings.bin`
/// vector is computed and stored for every indexed file too (see
/// DESIGN.md's "Component 7"); when it's `None`, that whole path is
/// skipped and no `embeddings.bin` is written.
///
/// Phase 1: `ignore::WalkParallel`'s own thread pool both walks the tree
/// and does the CPU-heavy per-file work (read -> binary-sniff ->
/// parse/tokenize -> extract [-> embed]) inline on whichever worker thread
/// visits that file. Each thread buffers its own `(term, file_id, freq)`
/// triples locally and spills them, sorted, to its own temp files --
/// there's no per-file cross-thread hand-off, only a once-per-thread one at
/// teardown. Phase 2, run after every thread has finished and joined,
/// merges all the spill files into the project's `terms.fst` /
/// `postings.bin` / `docs.bin` (and `embeddings.bin`, if enabled) -- see
/// `store::build_index`.
pub fn run(
    root: &Path,
    project_dir: &Path,
    excludes: &[String],
    threads: Option<usize>,
    embedding_model: Option<&EmbeddingModel>,
) -> Result<RunStats, Box<dyn std::error::Error>> {
    let build_dir = project_dir.join("build.tmp");
    if build_dir.exists() {
        // Leftover from a crashed/interrupted previous build. Must be
        // cleared before starting: the fresh build's file_ids start over
        // at 0, so merging stale spill runs in alongside new ones would
        // silently corrupt the postings.
        fs::remove_dir_all(&build_dir)?;
    }
    let spill_dir = build_dir.join("spill");
    fs::create_dir_all(&spill_dir)?;

    let mut override_builder = OverrideBuilder::new(root);
    for pattern in excludes {
        let negated = if pattern.starts_with('!') {
            pattern.clone()
        } else {
            format!("!{pattern}")
        };
        override_builder.add(&negated)?;
    }
    let overrides = override_builder.build()?;

    let mut walk_builder = WalkBuilder::new(root);
    walk_builder.overrides(overrides);
    if let Some(threads) = threads {
        walk_builder.threads(threads);
    }

    let file_id_counter = AtomicU64::new(0);
    let spill_seq = AtomicU64::new(0);
    let indexed = AtomicUsize::new(0);
    let skipped = AtomicUsize::new(0);
    let outputs: Mutex<Vec<ThreadOutput>> = Mutex::new(Vec::new());

    let walker = walk_builder.build_parallel();
    walker.run(|| {
        let mut state = ThreadState {
            root,
            spill_dir: &spill_dir,
            spill_seq: &spill_seq,
            file_id_counter: &file_id_counter,
            indexed: &indexed,
            skipped: &skipped,
            outputs: &outputs,
            embedding_model,
            buffer: Vec::new(),
            docs: Vec::new(),
            embeddings: Vec::new(),
            spill_paths: Vec::new(),
        };
        Box::new(move |entry| {
            if let Ok(entry) = entry {
                process_entry(&entry, &mut state);
            }
            WalkState::Continue
        })
    });

    let outputs = outputs.into_inner().unwrap();
    let total_files = file_id_counter.load(Ordering::Relaxed) as usize;
    let mut doc_entries: Vec<Option<(String, u32)>> = vec![None; total_files];
    let mut embedding_slots: Vec<Option<Vec<f32>>> = if embedding_model.is_some() {
        vec![None; total_files]
    } else {
        Vec::new()
    };
    let mut spill_paths = Vec::new();
    for output in outputs {
        spill_paths.extend(output.spill_paths);
        for (file_id, path, doc_length) in output.docs {
            doc_entries[file_id as usize] = Some((path, doc_length));
        }
        for (file_id, vector) in output.embeddings {
            embedding_slots[file_id as usize] = Some(vector);
        }
    }
    let doc_entries: Vec<(String, u32)> = doc_entries
        .into_iter()
        .map(|d| d.expect("every file_id must have exactly one recorded doc entry"))
        .collect();

    let embedding_vectors: Option<Vec<Vec<f32>>> = embedding_model.map(|_| {
        embedding_slots
            .into_iter()
            .map(|v| v.expect("every file_id must have exactly one recorded embedding"))
            .collect()
    });
    let embeddings_arg = embedding_model
        .zip(embedding_vectors.as_ref())
        .map(|(model, vectors)| (model.dim(), vectors.as_slice()));

    store::build_index(&build_dir, project_dir, &spill_paths, &doc_entries, embeddings_arg)?;
    fs::remove_dir_all(&build_dir)?;

    Ok(RunStats {
        files_indexed: indexed.load(Ordering::Relaxed),
        files_skipped: skipped.load(Ordering::Relaxed),
    })
}

fn process_entry(entry: &DirEntry, state: &mut ThreadState) {
    let is_file = entry.file_type().is_some_and(|ft| ft.is_file());
    if !is_file {
        return;
    }

    let path = entry.path();
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return;
    };
    let ext = ext.to_ascii_lowercase();

    let grammar = Grammar::for_extension(&ext);
    let is_fallback = languages::is_fallback_extension(&ext);
    if grammar.is_none() && !is_fallback {
        return; // not in the extension allowlist
    }

    let Ok(bytes) = std::fs::read(path) else {
        state.skipped.fetch_add(1, Ordering::Relaxed);
        return;
    };

    // Defense-in-depth: the extension allowlist already limits us to source
    // and structured-text formats, but a file can still lie about its
    // contents (e.g. a binary blob named `foo.json`).
    if content_inspector::inspect(&bytes).is_binary() {
        state.skipped.fetch_add(1, Ordering::Relaxed);
        return;
    }

    let Ok(rel_path) = path.strip_prefix(state.root) else {
        state.skipped.fetch_add(1, Ordering::Relaxed);
        return;
    };
    let Some(rel_path) = rel_path.to_str() else {
        state.skipped.fetch_add(1, Ordering::Relaxed);
        return;
    };

    let terms = match grammar {
        Some(grammar) => extract::extract_identifiers(grammar, &bytes),
        None => extract::extract_identifiers_fallback(&String::from_utf8_lossy(&bytes)),
    };

    let embedding = state
        .embedding_model
        .map(|model| model.embed(&String::from_utf8_lossy(&bytes)));

    state.indexed.fetch_add(1, Ordering::Relaxed);
    state.record_file(terms, rel_path.to_string(), embedding);
}
