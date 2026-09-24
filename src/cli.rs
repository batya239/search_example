use std::path::PathBuf;

use clap::Parser;

/// Default exclusion globs, matched relative to the indexed root.
pub const DEFAULT_EXCLUDES: &[&str] = &[
    "out", "build", "system", "config", "target", "node_modules", ".git",
];

#[derive(Parser, Debug)]
#[command(name = "local_indexer", about = "Local multithreaded identifier indexer")]
pub struct Cli {
    /// Directory to index, recursively.
    pub root: PathBuf,

    /// Root directory under which per-project index directories are stored.
    /// Defaults to a local-data directory (e.g. ~/Library/Application Support/local_indexer on macOS).
    #[arg(long)]
    pub storage_root: Option<PathBuf>,

    /// Additional exclusion globs, on top of the built-in defaults.
    #[arg(long = "exclude")]
    pub exclude: Vec<String>,

    /// Number of walker/worker threads. Defaults to the available parallelism.
    #[arg(long)]
    pub threads: Option<usize>,

    /// Local directory holding the `potion-code-16M-v2` embedding model
    /// (`config.json` + `tokenizer.json` + `model.safetensors`, downloaded
    /// ahead of time -- this tool never fetches it itself). When omitted,
    /// embeddings indexing is skipped entirely and no `embeddings.bin` is
    /// written.
    #[arg(long)]
    pub embedding_model: Option<PathBuf>,
}

pub fn default_storage_root() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("local_indexer")
        .join("projects")
}
