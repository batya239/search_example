use std::io;
use std::path::{Path, PathBuf};

use crate::hash::fnv1a64;

const PROJECT_PATH_FILE: &str = "PROJECT_PATH";

/// A resolved indexing target: the canonicalized project root, and the
/// index directory (holding `terms.fst` / `postings.bin` / `docs.bin`)
/// dedicated to it.
pub struct Project {
    pub canonical_root: PathBuf,
    pub index_dir: PathBuf,
}

/// Resolves `root` to its own project directory under `storage_root`.
///
/// Project identity is the canonicalized absolute path of `root` (v1
/// placeholder — see DESIGN.md's "Project isolation" section for the known
/// limitation that a moved/renamed project root is treated as a new
/// project). The directory name is a hex FNV-1a64 hash of that path, with a
/// `PROJECT_PATH` sidecar file recording the real path so a hash collision
/// between two different roots is detected rather than silently mixing
/// their indexes.
pub fn resolve(root: &Path, storage_root: &Path) -> io::Result<Project> {
    let canonical_root = std::fs::canonicalize(root)?;
    let root_display = canonical_root.to_string_lossy().into_owned();

    let hash = fnv1a64(root_display.as_bytes());
    let project_dir = storage_root.join(format!("{hash:016x}"));
    std::fs::create_dir_all(&project_dir)?;

    let sidecar_path = project_dir.join(PROJECT_PATH_FILE);
    match std::fs::read_to_string(&sidecar_path) {
        Ok(recorded) if recorded.trim() == root_display => {}
        Ok(recorded) => {
            return Err(io::Error::other(format!(
                "project directory hash collision: {} is recorded for {:?} but was resolved for {:?}",
                sidecar_path.display(),
                recorded.trim(),
                root_display,
            )));
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            std::fs::write(&sidecar_path, &root_display)?;
        }
        Err(e) => return Err(e),
    }

    Ok(Project {
        canonical_root,
        index_dir: project_dir,
    })
}
