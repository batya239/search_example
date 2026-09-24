mod cli;
mod embeddings;
mod extract;
mod hash;
mod languages;
mod pipeline;
mod project;
mod store;

use clap::Parser;

use embeddings::EmbeddingModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = cli::Cli::parse();

    let storage_root = args
        .storage_root
        .clone()
        .unwrap_or_else(cli::default_storage_root);
    std::fs::create_dir_all(&storage_root)?;

    let project = project::resolve(&args.root, &storage_root)?;

    let mut excludes: Vec<String> = cli::DEFAULT_EXCLUDES.iter().map(|s| s.to_string()).collect();
    excludes.extend(args.exclude.iter().cloned());

    let embedding_model = args
        .embedding_model
        .as_deref()
        .map(EmbeddingModel::load)
        .transpose()?;

    let stats = pipeline::run(
        &project.canonical_root,
        &project.index_dir,
        &excludes,
        args.threads,
        embedding_model.as_ref(),
    )?;

    println!(
        "indexed {} files ({} skipped) from {} into {}",
        stats.files_indexed,
        stats.files_skipped,
        project.canonical_root.display(),
        project.index_dir.display(),
    );

    Ok(())
}
