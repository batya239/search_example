use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use fst::MapBuilder;
use half::f16;

const TERMS_FILE: &str = "terms.fst";
const POSTINGS_FILE: &str = "postings.bin";
const DOCS_FILE: &str = "docs.bin";
const EMBEDDINGS_FILE: &str = "embeddings.bin";
const FORWARD_FILE: &str = "forward.bin";
const FORWARD_OFFSETS_FILE: &str = "forward_offsets.bin";

/// Writes `v` as an unsigned LEB128 varint (7 payload bits per byte, high
/// bit set on every byte but the last). Returns the number of bytes
/// written, so callers can track postings-block byte offsets without a
/// separate seek.
fn write_uvarint(w: &mut impl Write, mut v: u64) -> io::Result<u64> {
    let mut buf = [0u8; 10];
    let mut i = 0;
    loop {
        let mut byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        buf[i] = byte;
        i += 1;
        if v == 0 {
            break;
        }
    }
    w.write_all(&buf[..i])?;
    Ok(i as u64)
}

/// Reads a varint written by `write_uvarint`.
fn read_uvarint(r: &mut impl Read) -> io::Result<u64> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        let mut byte = [0u8; 1];
        r.read_exact(&mut byte)?;
        result |= u64::from(byte[0] & 0x7f) << shift;
        if byte[0] & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    Ok(result)
}

/// Writes one thread's locally-sorted `(term, file_id, freq)` triples to a
/// fresh spill file, as a flat sequence of `varint(term_len) ++ term_bytes
/// ++ varint(file_id) ++ varint(freq)` records. `entries` must already be
/// sorted by `(term, file_id)` -- that's what makes the phase-2 k-way merge
/// possible. `freq` is that term's occurrence count within `file_id`,
/// carried through untouched for a future BM25 searcher (see DESIGN.md's
/// "Storage" section).
pub fn write_spill_file(path: &Path, entries: &[(String, u64, u32)]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for (term, file_id, freq) in entries {
        write_uvarint(&mut w, term.len() as u64)?;
        w.write_all(term.as_bytes())?;
        write_uvarint(&mut w, *file_id)?;
        write_uvarint(&mut w, *freq as u64)?;
    }
    w.flush()
}

/// A cursor over one sorted spill file, used as a k-way merge input.
struct SpillRun {
    reader: BufReader<File>,
    current: Option<(String, u64, u32)>,
}

impl SpillRun {
    fn open(path: &Path) -> io::Result<SpillRun> {
        let mut reader = BufReader::new(File::open(path)?);
        let current = Self::read_entry(&mut reader)?;
        Ok(SpillRun { reader, current })
    }

    fn read_entry(reader: &mut BufReader<File>) -> io::Result<Option<(String, u64, u32)>> {
        let len = match read_uvarint(reader) {
            Ok(len) => len,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        };
        let mut buf = vec![0u8; len as usize];
        reader.read_exact(&mut buf)?;
        let term = String::from_utf8(buf).map_err(io::Error::other)?;
        let file_id = read_uvarint(reader)?;
        let freq = read_uvarint(reader)? as u32;
        Ok(Some((term, file_id, freq)))
    }

    fn advance(&mut self) -> io::Result<()> {
        self.current = Self::read_entry(&mut self.reader)?;
        Ok(())
    }
}

/// Encodes one identifier's postings block: the doc count, then each
/// `(file_id, freq)` pair -- the file id as a delta from the previous one
/// (the first id is a delta from 0), immediately followed by its term
/// frequency in that document. `docs` must already be sorted ascending by
/// `file_id` and deduplicated. Returns the number of bytes written, so the
/// caller can track byte offsets.
fn write_postings_block(w: &mut impl Write, docs: &[(u64, u32)]) -> io::Result<u64> {
    let mut n = write_uvarint(w, docs.len() as u64)?;
    let mut prev = 0u64;
    for (i, &(id, freq)) in docs.iter().enumerate() {
        let delta = if i == 0 { id } else { id - prev };
        n += write_uvarint(w, delta)?;
        n += write_uvarint(w, freq as u64)?;
        prev = id;
    }
    Ok(n)
}

/// Phase 2: a k-way merge over every thread's sorted spill runs, grouping
/// entries by term. Each distinct term is visited in ascending
/// lexicographic order -- which is both what the merge naturally produces
/// and what `fst::MapBuilder` requires -- so as each term's group closes,
/// this writes its postings block and inserts the term into the builder at
/// that block's starting offset.
///
/// A given `(term, file_id)` pair can only ever appear in one spill file
/// (each file is visited by exactly one worker thread, which drains that
/// file's whole term->frequency map into its buffer in one call before any
/// flush can split it), so `(file_id, freq)` pairs within one term's group
/// are already unique-by-`file_id` and strictly increasing as the merge
/// produces them -- no extra dedup pass is needed here.
fn merge_spill_runs(
    spill_paths: &[PathBuf],
    postings_writer: &mut impl Write,
    fst_builder: &mut MapBuilder<impl Write>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut runs = spill_paths
        .iter()
        .map(|p| SpillRun::open(p))
        .collect::<io::Result<Vec<_>>>()?;

    let mut heap: BinaryHeap<Reverse<(String, u64, usize)>> = BinaryHeap::new();
    for (i, run) in runs.iter().enumerate() {
        if let Some((term, file_id, _freq)) = &run.current {
            heap.push(Reverse((term.clone(), *file_id, i)));
        }
    }

    let mut offset = 0u64;
    let mut current_term: Option<String> = None;
    let mut current_docs: Vec<(u64, u32)> = Vec::new();

    while let Some(Reverse((term, file_id, run_idx))) = heap.pop() {
        let freq = runs[run_idx]
            .current
            .as_ref()
            .expect("heap entry must match this run's current record")
            .2;
        runs[run_idx].advance()?;
        if let Some((next_term, next_id, _)) = &runs[run_idx].current {
            heap.push(Reverse((next_term.clone(), *next_id, run_idx)));
        }

        if current_term.as_deref() != Some(term.as_str()) {
            if let Some(prev_term) = current_term.take() {
                let block_start = offset;
                offset += write_postings_block(postings_writer, &current_docs)?;
                fst_builder.insert(prev_term.as_bytes(), block_start)?;
                current_docs.clear();
            }
            current_term = Some(term);
        }
        current_docs.push((file_id, freq));
    }
    if let Some(prev_term) = current_term.take() {
        let block_start = offset;
        write_postings_block(postings_writer, &current_docs)?;
        fst_builder.insert(prev_term.as_bytes(), block_start)?;
    }

    Ok(())
}

/// `docs.bin`: one entry per indexed file, in `file_id` order --
/// `varint(path_byte_len) ++ path_bytes ++ varint(doc_length)`, where
/// `doc_length` is that file's total identifier-occurrence count (the sum
/// of its term->frequency map's values) -- the per-document length a
/// future BM25 searcher needs for length normalization.
fn write_docs_file(path: &Path, docs: &[(String, u32)]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for (p, doc_length) in docs {
        write_uvarint(&mut w, p.len() as u64)?;
        w.write_all(p.as_bytes())?;
        write_uvarint(&mut w, *doc_length as u64)?;
    }
    w.flush()
}

/// Encodes one file's forward-index block (see DESIGN.md's "Component 8"):
/// `varint(term_count)` then, per term in ascending order, `varint(term_len)
/// ++ term_bytes ++ varint(freq)`. Unlike `postings.bin`, terms are stored
/// directly rather than referencing `terms.fst` -- this index has no
/// dependency on the term dictionary's build order, so it's built entirely
/// within phase 1, right alongside the term->frequency map itself.
pub fn encode_forward_block(terms: &BTreeMap<String, u32>) -> Vec<u8> {
    let mut buf = Vec::new();
    write_uvarint(&mut buf, terms.len() as u64).expect("writing to a Vec<u8> never fails");
    for (term, freq) in terms {
        write_uvarint(&mut buf, term.len() as u64).expect("writing to a Vec<u8> never fails");
        buf.extend_from_slice(term.as_bytes());
        write_uvarint(&mut buf, u64::from(*freq)).expect("writing to a Vec<u8> never fails");
    }
    buf
}

/// Writes `forward.bin` (each file's block, from `blocks`, concatenated in
/// `file_id` order) and `forward_offsets.bin` (`N + 1` fixed-stride `u64`
/// (LE) byte offsets into it -- one per `file_id`, plus a trailing sentinel
/// equal to `forward.bin`'s total length, so a block's length is always
/// `offsets[i + 1] - offsets[i]` with no special case for the last one).
/// See DESIGN.md's "Component 8".
fn write_forward_files(forward_path: &Path, offsets_path: &Path, blocks: &[Vec<u8>]) -> io::Result<()> {
    let mut forward_writer = BufWriter::new(File::create(forward_path)?);
    let mut offsets_writer = BufWriter::new(File::create(offsets_path)?);
    let mut offset = 0u64;
    for block in blocks {
        offsets_writer.write_all(&offset.to_le_bytes())?;
        forward_writer.write_all(block)?;
        offset += block.len() as u64;
    }
    offsets_writer.write_all(&offset.to_le_bytes())?;
    forward_writer.flush()?;
    offsets_writer.flush()
}

/// `embeddings.bin`: a fixed-stride array, one record per `file_id` --
/// `u32 dim` (LE) header, then `count x dim` `f16` values (LE). Every
/// record is the same size, so looking one up needs no varint framing --
/// see DESIGN.md's "Component 7."
fn write_embeddings_file(path: &Path, dim: usize, vectors: &[Vec<f32>]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(&(dim as u32).to_le_bytes())?;
    for vector in vectors {
        debug_assert_eq!(vector.len(), dim);
        for &value in vector {
            w.write_all(&f16::from_f32(value).to_le_bytes())?;
        }
    }
    w.flush()
}

/// Builds a project's index (`terms.fst` + `postings.bin` + `docs.bin` +
/// `forward.bin` + `forward_offsets.bin`, plus `embeddings.bin` when
/// embedding vectors are supplied) from the spill runs and per-file data
/// gathered during the parallel walk, then atomically publishes it: every
/// file is written under `build_dir` first, and only once all of them have
/// been flushed successfully are they renamed into `project_dir` (a plain
/// rename, since both directories are on the same filesystem) -- so a
/// crash or interrupt during the build never touches whatever index was
/// already there.
pub fn build_index(
    build_dir: &Path,
    project_dir: &Path,
    spill_paths: &[PathBuf],
    docs: &[(String, u32)],
    forward_blocks: &[Vec<u8>],
    embeddings: Option<(usize, &[Vec<f32>])>,
) -> Result<(), Box<dyn std::error::Error>> {
    let terms_tmp = build_dir.join(TERMS_FILE);
    let postings_tmp = build_dir.join(POSTINGS_FILE);
    let docs_tmp = build_dir.join(DOCS_FILE);
    let forward_tmp = build_dir.join(FORWARD_FILE);
    let forward_offsets_tmp = build_dir.join(FORWARD_OFFSETS_FILE);

    {
        let mut postings_writer = BufWriter::new(File::create(&postings_tmp)?);
        let mut fst_builder = MapBuilder::new(BufWriter::new(File::create(&terms_tmp)?))?;
        merge_spill_runs(spill_paths, &mut postings_writer, &mut fst_builder)?;
        postings_writer.flush()?;
        fst_builder.finish()?;
    }
    write_docs_file(&docs_tmp, docs)?;
    write_forward_files(&forward_tmp, &forward_offsets_tmp, forward_blocks)?;

    let mut file_names = vec![
        TERMS_FILE,
        POSTINGS_FILE,
        DOCS_FILE,
        FORWARD_FILE,
        FORWARD_OFFSETS_FILE,
    ];
    if let Some((dim, vectors)) = embeddings {
        let embeddings_tmp = build_dir.join(EMBEDDINGS_FILE);
        write_embeddings_file(&embeddings_tmp, dim, vectors)?;
        file_names.push(EMBEDDINGS_FILE);
    }

    for file_name in file_names {
        fs::rename(build_dir.join(file_name), project_dir.join(file_name))?;
    }
    Ok(())
}
