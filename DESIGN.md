# Local Code Indexer — Design Doc

## Goal

A CLI tool that recursively indexes a directory of source files and builds a
persistent, on-disk **inverted index**: identifier → the set of files that
contain it. The index is a compact, read-optimized, mmap-friendly static
structure — an `fst`-backed term dictionary plus an append-only postings
file — built fresh from scratch on every run rather than mutated in place.

Postings now carry per-document **term frequency** (not just presence), and
per-document **length** is recorded alongside its path, so a future BM25
ranking searcher can be built directly on top of this data without another
on-disk format change — see "Storage" and "Non-goals." A second, optional
index sits alongside it: one fixed-size **embedding vector per file**,
produced by a small static code-embedding model, for future similarity
search — see "Component 7."

## Non-goals (v1)

- **No relevance ranking / no BM25 searcher.** Postings now store per-document
  term frequency and `docs.bin` stores per-document length (see "Storage"),
  because those are exactly the inputs a BM25 scorer needs — but *scoring*
  itself (computing document frequency across the corpus, an IDF term, the
  actual BM25 formula, ranking/sorting results) is not implemented here.
  This doc only covers making the data available.
- **No embeddings similarity search.** Same split as above: Component 7
  computes and persists one vector per file; consuming those vectors for
  nearest-neighbor/ANN search is a follow-up, not built here.
- **No incremental/watch mode** (inotify/fsevents). Every run does a full
  walk *and* a full rebuild — nothing about this on-disk format supports an
  in-place update (see "Storage"), so re-running against the same project
  always replaces the whole index, atomically, never diffs it.
- **No cross-file semantic analysis** — each file is parsed and indexed
  independently.
- **No stable project identity across a move/rename of the project root** —
  see "Project isolation" limitation below.
- **No query/search command.** This doc only covers *building* the indexes.
  A follow-up `search <term>` subcommand (term lookup, or later BM25-ranked)
  and a `similar <path>` subcommand (embedding nearest-neighbor) are the
  natural next steps these formats are deliberately shaped for, but neither
  is designed here.

## Pipeline

```
                 ┌────────────────────┐
                 │   ignore::WalkParallel   │  recursive walk, N worker threads,
                 │  (dir exclusions applied) │  built-in to the `ignore` crate
                 └──────────┬─────────┘
                            │ DirEntry (files only)
                            ▼
                 ┌────────────────────┐
                 │ extension allowlist │  skip unless extension is one of the
                 │       check         │  12 supported (see "File selection")
                 └──────────┬─────────┘
                            │ extension recognized
                            ▼
                 ┌────────────────────┐
                 │  binary sniff check │  safety net: skip if content doesn't
                 │  (defense-in-depth) │  actually look like text despite the
                 └──────────┬─────────┘  extension (corrupt/mislabeled file)
                            │ text file
                            ▼
                 ┌────────────────────┐
                 │  tree-sitter parse  │  per-language grammar; .txt has none
                 │   OR word-tokenize  │  and .md's grammar doesn't tokenize
                 └──────────┬─────────┘  prose — both use the fallback instead
                            │ syntax tree / token stream
                            ▼
                 ┌────────────────────┐
                 │ walk tree, collect  │  per-language node-kind list
                 │ identifier terms,   │  (see "File selection" table);
                 │ count occurrences   │  counted into a
                 │    per file         │  BTreeMap<String, u32>
                 └──────────┬─────────┘
                            │ file_id (from a shared AtomicU64) + term→freq map
                            ▼
                 ┌────────────────────┐
                 │ per-thread local     │  (term, file_id, freq) triples,
                 │  sorted spill runs   │  sorted and flushed to their own
                 │   (own temp files)    │  temp file once a size threshold
                 └──────────┬─────────┘  is passed
                            │ N sorted spill files, once WalkParallel finishes
                            ▼
                 ┌────────────────────┐
                 │    k-way merge       │  single-threaded; groups entries by
                 │   (BinaryHeap)       │  term, dedups by file_id (frequency
                 └──────────┬─────────┘  travels along, doesn't need merging)
                            │ (term, [(file_id, freq), ...]) in term order
                            ▼
                 ┌────────────────────┐    ┌─────────────────────┐
                 │     terms.fst        │    │     postings.bin      │
                 │ term → byte offset   │───▶│ varint(count), then    │
                 │   into postings.bin  │    │ delta-id/freq pairs    │
                 └────────────────────┘    └─────────────────────┘
                       plus  docs.bin  (file_id → path + doc length, id order)
```

Each pipeline stage through "count occurrences per file" runs on a
`WalkParallel` worker thread; the spill/merge/build stages after that are
described in full under "Storage" and "Concurrency model."

**Embeddings side-path** (Component 7): on the same worker thread, for the
same file, independently of the term-frequency path above —

```
        raw file bytes (already read + binary-sniffed, decoded lossily as UTF-8)
                            │
                            ▼
                 ┌────────────────────┐
                 │  tokenizers crate    │  add_special_tokens=false,
                 │  (tokenizer.json)     │  unk-token ids dropped
                 └──────────┬─────────┘
                            │ token ids
                            ▼
                 ┌────────────────────┐
                 │ embedding table      │  potion-code-16M-v2's [63457, 256]
                 │  row lookup + mean    │  f16 matrix, decoded once at
                 │      pooling          │  startup; per-row lookup + average
                 └──────────┬─────────┘
                            │ one 256-dim f32 vector
                            ▼
                 ┌────────────────────┐
                 │  L2 normalize        │  per the model's own config
                 └──────────┬─────────┘  (`normalize: true`)
                            ▼
                     embeddings.bin  (file_id → 256×f16 vector, fixed stride)
```

This path only runs when `--embedding-model <dir>` is supplied (see
"Config"); when it's omitted, no tokenization/embedding work happens and no
`embeddings.bin` is written.

## File selection

File selection is **by extension, against an explicit allowlist** — not
"try to parse it and see." A file is only considered for indexing if its
extension is one of the 12 below; everything else is skipped. This
replaces the old "extension → grammar lookup, skip if unmapped" step:
unmapped now just means "not in this table."

**Definition — "identifier":** not a language keyword (reserved words like
`if`/`class`/`return` are explicitly excluded). An identifier is any token
that names or references something: variable/function/class/type/field/
property names in code, and — extending the same idea to non-code formats
— key/tag/attribute names in JSON/TOML/XML, since those also just name a
value rather than being a reserved word.

| Ext | Language | tree-sitter crate | Identifier = node kind(s) |
|---|---|---|---|
| `.rs` | Rust | `tree-sitter-rust` | `identifier`, `type_identifier`, `field_identifier`, `shorthand_field_identifier` |
| `.java` | Java | `tree-sitter-java` | `identifier`, `type_identifier` |
| `.kt` | Kotlin | `tree-sitter-kotlin-ng` (the original `tree-sitter-kotlin` is unmaintained) | `identifier` |
| `.go` | Go | `tree-sitter-go` | `identifier`, `type_identifier`, `field_identifier`, `package_identifier` |
| `.py` | Python | `tree-sitter-python` | `identifier` |
| `.js` | JavaScript | `tree-sitter-javascript` | `identifier`, `property_identifier`, `shorthand_property_identifier`, `shorthand_property_identifier_pattern`, `private_property_identifier`, `statement_identifier` |
| `.ts` | TypeScript | `tree-sitter-typescript` | same as `.js`, plus `type_identifier` |
| `.json` | JSON | `tree-sitter-json` | none — see structural extraction note below |
| `.toml` | TOML | `tree-sitter-toml-ng` (the original `tree-sitter-toml` is unmaintained) | `bare_key`, `quoted_key` (table/property keys) |
| `.xml` | XML | `tree-sitter-xml` | `Name` (element and attribute names) |
| `.md` | Markdown | *(no grammar used)* | none directly — see note below |
| `.txt` | Plain text | *(no grammar exists)* | none — see note below |

> Node-kind names above were verified against each grammar's generated
> `node-types.json` during implementation. Composite/dotted-path wrapper
> kinds (e.g. Rust's `scoped_identifier`, Kotlin's `qualified_identifier`,
> TypeScript's `nested_identifier`, Python's `dotted_name`) are deliberately
> excluded: they only ever wrap the leaf identifier kinds above as children,
> and a full-tree walk already visits those leaves directly, so including
> the wrapper too would just add a redundant, less-precise hash of the whole
> dotted path. Python has no separate `type_identifier`/`field_identifier`
> kinds the way Rust or Go do — it's dynamically typed, and attribute access
> (`obj.attr`) is just a plain `identifier` on the right of a `.`.
>
> **JSON is structural, not kind-based:** JSON has no identifier-kind node
> at all — an object key is a plain `string` node, indistinguishable by
> kind from a string *value*. So instead of a flat node-kind allowlist,
> extraction walks every `pair` node and reads its `key` field specifically
> (concatenating the `string_content` children of that `string` node, to
> handle keys containing escape sequences).

**Note on `.md` and `.txt`:** neither fits the "parse with tree-sitter,
collect identifier nodes" model:

- `.txt` has no tree-sitter grammar at all — there's no tree to walk.
- `.md`'s grammar is *structural* (headings, lists, code fences, block
  quotes...); the prose itself ends up as opaque `inline`/`text` leaf nodes
  with no word-level breakdown, so walking its tree for identifier-kind
  nodes would yield almost nothing useful.

Fallback for both, confirmed in the implementation: a plain word-tokenizer
over the raw text (split on whitespace/punctuation boundaries), treating
each resulting word as an identifier-equivalent token, used *instead of*
tree-sitter for these two extensions.

## Components

### 1. Directory walker — `ignore` crate

Use `ignore::WalkBuilder` + `.build_parallel()` (`WalkParallel`), the same
walker ripgrep uses. It already gives us:

- a real thread pool for the walk itself (`.threads(n)`)
- gitignore-style pattern matching for exclusions, via
  `WalkBuilder::overrides` or a synthetic ignore file, so the default
  exclusion list is just patterns:

  ```
  out/
  build/
  system/
  config/
  target/       # Rust build output
  node_modules/
  .git/
  ```

  This list is a config default (see "Config"), not hardcoded — the user
  supplies additional excludes via CLI flag.

- `WalkParallel::run()` takes a closure returning a `WalkState` per entry,
  which is where we dispatch file-level work — processed inline on that same
  worker thread (see "Concurrency model").

### 2. Binary-file detection

Read the first ~8 KiB of the file; if it contains a NUL byte or fails UTF-8
validation heuristically, treat it as binary and skip. Use the
`content_inspector` crate (does exactly this, no need to hand-roll) rather
than reimplementing a sniffing heuristic.

### 3. Language detection

A static `HashMap<&str, Language>` keyed by the 12 supported extensions,
per the "File selection" table above (`"rs"` → `tree_sitter_rust::LANGUAGE`,
`"json"` → `tree_sitter_json::LANGUAGE`, etc.) — this map *is* the
allowlist; there's no separate "supported extensions" list to keep in
sync. `.txt` and `.md` are handled by the word-tokenizer fallback instead
of an entry in this map.

### 4. Identifier extraction

Two extraction strategies, dispatched by extension:

- **Tree-sitter languages** (`.rs .java .kt .go .py .js .ts .json .toml .xml`):
  walk the parsed tree (via `tree_sitter::TreeCursor`) and collect the text
  of every node whose `.kind()` is in that language's identifier node-kind
  list from the "File selection" table. For the programming languages
  that's literal identifier nodes; for JSON/TOML/XML (which have no
  "identifier" node in the traditional sense) it's key/tag/attribute-name
  nodes instead, per the broadened definition above — reserved words and
  literal values are never included either way.
- **Word-tokenizer fallback** (`.txt .md`): split the raw file content on
  whitespace/punctuation boundaries and treat each resulting word as a
  token, with no tree-sitter parse involved.

Either way the result is collapsed into that file's own term→frequency map
(`BTreeMap<String, u32>`: each distinct identifier mapped to how many times
it occurred in this file) before it becomes input to the indexing step
below, rather than a plain dedup set — this is what lets a future BM25
searcher get per-document term frequency straight out of the index with no
second pass over the source. The map's *values* summed together give that
file's total token count, which travels alongside its path into `docs.bin`
(see "Storage") as the per-document length BM25 needs for its length-
normalization term. This is also where the old design's per-file hashing
step used to sit; see "Storage" for why that's gone.

### 5. Storage — a static `fst` term dictionary + append-only postings

Unlike a KV store, `fst::Map` has no "insert into an existing index" API —
keys must be written once, in strictly increasing lexicographic order, in a
single build pass (`MapBuilder::insert` returns `Error::OutOfOrder`
otherwise). So re-indexing a project always means building a brand-new
index from scratch and atomically swapping it in; there's no partial or
incremental update path, by construction of the format, not by policy
choice. That resolves the old design's open "re-run semantics" question —
see "Open questions."

**On-disk layout**, per project (see "Project isolation" for how the
directory itself is chosen):

```
<project index dir>/
  terms.fst     — fst::Map<&[u8], u64>: identifier text -> byte offset
                  into postings.bin
  postings.bin  — append-only; one block per unique identifier, at the
                  offset stored for it in terms.fst
  docs.bin      — file_id -> relative path, in file_id order
  PROJECT_PATH  — sidecar audit file, unchanged from before
```

**`postings.bin` block format** — one block per unique identifier,
self-describing (no separate length side-table needed):

```
varint(doc_count)
varint(file_id[0])                 — first id, stored absolute
varint(tf[0])                      — that document's term frequency
varint(file_id[1] - file_id[0])    — every subsequent id stored as a
varint(tf[1])                        delta from the one before it, with
varint(file_id[2] - file_id[1])      its term frequency interleaved right
varint(tf[2])                        after it
...
```

IDs within one block are sorted ascending and deduped, so every delta is
`>= 1`. Since `file_id`s are dense integers assigned `0..N` (not, say,
random UUIDs), these deltas stay small even for identifiers that appear in
a large fraction of the project — exactly the case varint encoding is
good at. `tf[i]` is that identifier's occurrence count within document
`file_id[i]`, carried straight through from the term→frequency map built in
Component 4 — nothing to compute during the merge, it just rides alongside
its `file_id` the whole way. Varint is the standard unsigned LEB128 scheme
(7 payload bits + a high continuation bit per byte): hand-rolled, no new
crate for it, same reasoning as the existing FNV-1a64 hash.

**`docs.bin` format**: one entry per indexed file, in `file_id` order —
`varint(path_byte_len)` followed by that many UTF-8 bytes, followed by
`varint(doc_length)` (that file's total identifier-occurrence count — the
sum of its term→frequency map's values, i.e. the document length a BM25
searcher needs for length normalization and for computing the corpus's
average document length). This file is small relative to `postings.bin`
(one entry per *file*, not per identifier occurrence), so a reader just
loads it fully into memory once, up front.

**Why `fst`**: its insert-order requirement is exactly what a k-way merge
over sorted runs produces naturally, so there's no data-shape mismatch to
work around. The resulting structure is a byte-level trie (finite-state
transducer) — shared key prefixes are stored once, so it's extremely
compact — and since `fst::Map::new` accepts any `D: AsRef<[u8]>`, it's
mmap-friendly out of the box: a future query command can `memmap2::Mmap`
the file directly with zero deserialization rather than loading it into a
`HashMap` first. That's a capability the RocksDB draft never gave us for
identifier lookup — its LSM tree is optimized for point/range lookups *by
file path*, so "which files contain identifier X" would have needed a full
scan.

**Build process** (one pass, two phases):

1. **Phase 1 — parallel walk + local spill.** Unchanged up through "count
   occurrences per file" (see the pipeline diagram above). For every file
   it indexes, a `WalkParallel` worker thread grabs the next `file_id` from
   a shared `AtomicU64` counter and buffers `(file_id, path, doc_length)`
   plus each `(term, file_id, freq)` triple *locally* — no cross-thread
   hand-off per file. Once a thread's local buffer passes a size threshold,
   the thread sorts it (by `(term, file_id)`, `freq` just riding along) in
   place and writes it out as one sorted run to its own temp file under the
   project's build directory, then clears the buffer and keeps going. When
   that thread's share of the walk is exhausted, a `Drop` impl on a small
   guard value captured in its visitor closure flushes the final partial
   buffer the same way, then hands the thread's list of spill-file paths
   and its local `(file_id, path, doc_length)` triples to the main thread
   through a small shared `Mutex<Vec<ThreadOutput>>` — one hand-off per
   *thread* (a handful of times, total), not one per file, so that lock
   sees essentially no contention.
2. **Phase 2 — merge**, single-threaded, run after `WalkParallel::run`
   returns and every thread has joined: a k-way merge (a `BinaryHeap` of
   per-run cursors) over all the spill files, each already sorted by
   `(term, file_id)`. As the merge advances, consecutive entries sharing a
   term are grouped into `(file_id, freq)` pairs (already unique and sorted
   by `file_id`, since a given `(term, file_id)` pair can only come from one
   spill file — see the doc comment on `merge_spill_runs`) and written as
   one `postings.bin` block; the term and that block's starting offset are
   then `insert`ed into the `fst::MapBuilder`, in the increasing order the
   merge already guarantees. `docs.bin` is written separately and once,
   from the concatenated per-thread `(file_id, path, doc_length)` triples
   gathered in phase 1 (scattered into one `Vec` sized to the final file
   count, indexed by `file_id`, since the total isn't known until the walk
   finishes).
3. The whole build happens inside a fresh temporary directory. Only once
   `MapBuilder::finish()` and both other files are flushed successfully
   does the tool atomically swap that directory in over the project's real
   index directory — so a crash or interrupt mid-build leaves the
   *previous* good index (if one exists) untouched, never a half-written
   one.

### 6. Project isolation — one index directory per project

Each indexed project gets its own, fully independent index directory (its
own `terms.fst` / `postings.bin` / `docs.bin` / `PROJECT_PATH`) — never
shared. Keeping projects in separate directories means each one's build
(temp dir → atomic swap, per "Storage" above) is independent of every other
project's, and there's no risk of one project's terms/postings ever mixing
into another's.

**Project identity, for now:** the project's absolute, *canonicalized* root
path — `std::fs::canonicalize(--root)`, which resolves symlinks and `.`/
`..` segments. Without canonicalizing, `/a/b` and `/a/./b` (or a symlinked
path) would be treated as two different projects when they're the same
directory.

**Index directory location:** `<storage-root>/<hex(FNV-1a-64(canonical_path))>`.
`storage-root` is a config value (see "Config"). Hashing the path keeps the
directory name short and filesystem-safe regardless of how deep the
project path is. (This FNV-1a64 hash is unrelated to identifier terms —
those are now stored as raw text in `terms.fst`, not hashed at all; this is
the one remaining use of that hash function, purely for naming a
directory.)

A hash alone isn't auditable, so each project's index directory also gets a
sidecar `PROJECT_PATH` file (plain UTF-8 text: the canonical path) written
next to it. This lets you inspect `storage-root` on disk to see which
directory belongs to which project, and lets the tool detect the
(extremely unlikely) hash collision by comparing the sidecar's recorded
path against the one just canonicalized before building.

**Known v1 limitation** (flagged, not solved): keying by path means moving
or renaming a project's root directory makes it look like a brand-new,
never-indexed project on the next run — the old index directory is
orphaned on disk with nothing to garbage-collect it. That's acceptable for
the "for now" scope you gave; a stable project-id mechanism (assigned once,
independent of path) would be the fix if orphaned directories become a
real problem — see "Open questions."

### 7. Embeddings indexer — `minishlab/potion-code-16M-v2`

An optional, second index built alongside the term index: one fixed-size
vector per file, from a small static code-embedding model, for future
similarity search (see "Non-goals" — the search side isn't built here).

**The model.** [`minishlab/potion-code-16M-v2`](https://huggingface.co/minishlab/potion-code-16M-v2)
is a **Model2Vec** model — a *static* embedding model, meaning there's no
attention or any other input-dependent computation at inference time: it's
a fixed per-token embedding table plus a pooling step, nothing more. That
makes it realistic to reimplement natively rather than needing a Python/
ONNX runtime, unlike a real transformer encoder.

Verified directly (repo file listing, the `model.safetensors` header bytes,
and the raw `config.json`/`modules.json` — not just the model card prose,
which doesn't spell out the exact formula):

- `tokenizer.json` (1.02 MB) — a standard HuggingFace *fast* tokenizer,
  loadable as-is by the Rust `tokenizers` crate (the same crate, same file
  format; no Python involved).
- `model.safetensors` (32.5 MB) — exactly **one** tensor, named
  `embeddings`, shape `[63457, 256]`, dtype `F16`. `63457 × 256 × 2 =
  32,489,984` bytes, matching the file's declared data length exactly. One
  tensor and nothing else means there's no `token_mapping` (vocabulary
  quantization) and no per-token `weights` tensor for this particular
  model — lookup is a plain, unweighted row fetch per token id, which
  simplifies the port.
- `config.json` (59 bytes) — `{"normalize": true, "embedding_dtype":
  "float16"}`.
- `modules.json` — sentence-transformers wrapper metadata listing two
  modules, `StaticEmbedding` then `Normalize`, confirming the same
  two-step shape as `config.json`.

**Inference algorithm**, verified against the `model2vec` Python package's
`StaticModel` source (again, not the model card, since it doesn't document
this):

1. Tokenize with `add_special_tokens=False` — no BOS/EOS/CLS/SEP added.
2. Drop any `<unk>` token ids from the resulting sequence.
3. For very long input, truncate *characters* (not tokens) to `max_length ×
   median_token_length` before tokenizing at all, where `median_token_length`
   is the median byte-length of every token string in the vocabulary — a
   cheap way to bound tokenization cost without needing a first tokenize
   pass just to find out the input was too long.
4. Mean-pool: look up each remaining token id's row in the embedding table
   and average them elementwise into one 256-dim vector. (No weighting —
   this model has no `weights` tensor, per above.)
5. L2-normalize the result (`v / (‖v‖₂ + 1e-32)`), per `config.json`'s
   `normalize: true`.

That's simple enough, and now fully pinned down instead of guessed, to
reimplement natively in Rust — consistent with how this project has always
preferred a small hand-rolled piece (varint, FNV-1a64, the postings merge)
over pulling in a heavyweight runtime. Two crates carry the parts that
*aren't* worth hand-rolling: `tokenizers` (official, pure Rust, reads this
exact `tokenizer.json` format) for step 1-3, and `safetensors` (official,
pure Rust) to parse `model.safetensors`'s header and tensor data. The
embedding table is decoded once at startup (via the `half` crate, for its
`F16` values) into an in-memory matrix shared read-only across worker
threads; the lookup + mean + normalize (steps 4-5) is a couple dozen lines
of hand-rolled code, no crate needed.

**Granularity: whole file**, not chunk/function-level — matching the
granularity already used everywhere else in this pipeline (one `docs.bin`
entry per file, one term→frequency map per file). The same file bytes
already read and binary-sniffed for term extraction (Component 4) are
decoded lossily as UTF-8 and fed to the tokenizer as one string; the result
is one 256-dim vector per file. Chunk-level embedding (e.g. one per
function) would need an AST-based chunker this project doesn't have — left
as a future improvement, see "Open questions."

**Model provisioning: no automatic network download.** Every other part of
this tool is fully offline; a hidden HTTP fetch on first run would be a
real behavior change, not an implementation detail. Instead, `--embedding-
model <dir>` (see "Config") points at a local directory already containing
`config.json` + `tokenizer.json` + `model.safetensors` — populated ahead of
time by the user (e.g. `huggingface-cli download minishlab/potion-code-
16M-v2 --local-dir ...`, or a manual clone). Passing this flag is what turns
embeddings indexing on at all; omitting it means the indexer behaves
exactly as it did before this component existed, and no `embeddings.bin`
is written.

**On-disk format**: `embeddings.bin` is a fixed-stride array, one record
per `file_id`, so looking one up is `header_len + file_id × dim × 2` — no
varint framing needed, since every record is the same size:

```
u32 dim (LE)                — 256 for this model, but read, not hardcoded
then, for file_id in 0..N:
  dim × f16 values (LE)     — that file's normalized embedding vector
```

Stored as `f16`, matching the model's own weight dtype and `config.json`'s
`embedding_dtype`, not upcast to `f32`: half the size for the same fidelity
the model natively produces, which matters at scale — 512 bytes/file vs.
1024, e.g. roughly 256 MB vs. 512 MB across intellij's ~500K files. A
future consumer decodes each row to `f32` (via `half`) only when it
actually needs to do math on it (cosine similarity, etc.) — decode-on-read
keeps the on-disk format itself simple and dtype-honest.

**Failure handling**: an empty file (zero tokens after tokenization) has no
rows to mean-pool; store an all-zero vector for it rather than failing the
run, consistent with "never abort the whole run for one bad file" (see
"Error handling").

**Integration**: this computation slots into the *same* per-thread walk
loop as term extraction (Component 4) — same file, same thread, one more
step before `ThreadState::record_file`, no new per-file I/O. The model
(`Arc<EmbeddingModel>`, loaded once up front) is shared read-only across
worker threads, which is sound since nothing about applying it mutates any
shared state. Like `docs.bin`, `embeddings.bin` is written once,
single-threaded, during phase 2, from the same per-thread `(file_id,
vector)` pairs gathered via `ThreadOutput` — the identical "scatter into a
`Vec` sized to the final file count, indexed by `file_id`" pattern already
used for `docs.bin`'s paths.

## Concurrency model

- `WalkParallel` owns the traversal threads (configurable count, defaults to
  available parallelism).
- Each thread does the full read → binary-sniff → parse/tokenize → extract
  → dedup pipeline locally, for every file it's handed — unaffected by the
  storage rewrite.
- What changed is what happens *after* that: previously, every file's result
  crossed a channel to one dedicated writer thread. Now each worker thread
  keeps its own local spill buffer and touches shared state only twice per
  file at most — a lock-free `AtomicU64` fetch-add to claim a `file_id` —
  and only a handful of times *total*, at thread teardown, to hand off its
  finished spill-file list (see "Storage," phase 1). So `crossbeam-channel`
  is no longer pulled in; a plain `std::sync::Mutex` around a small `Vec` is
  more than sufficient given how rarely it's touched.
- The merge phase (phase 2 of "Storage") is single-threaded in v1 — the
  simplest correct option. Parallelizing the k-way merge (e.g. a merge tree
  sized to the thread count) is a plausible follow-up if it turns out to
  dominate wall-clock time on very large projects, but isn't designed here.
- The embeddings model (Component 7), when enabled, is loaded once before
  the walk starts and shared across worker threads behind an `Arc` — safe
  without any locking, since applying it (tokenize + table lookup + mean +
  normalize) never mutates the shared table. Each thread's embedding
  vectors travel back to the main thread the same way its `docs.bin`
  triples already do: bundled into `ThreadOutput`, handed off once per
  thread at teardown through the existing `Mutex<Vec<ThreadOutput>>` — no
  new shared state or synchronization primitive needed for this feature.

## Config

CLI flags (exact surface TBD, not the focus of this doc):

- `--root <path>` — project directory to index (required). Canonicalized
  internally to derive project identity (see "Project isolation").
- `--storage-root <path>` — base directory holding all per-project index
  directories (default: an OS-appropriate data dir via the `dirs` crate,
  e.g. `~/.local/share/local-indexer` on Linux, `~/Library/Application
  Support/local-indexer` on macOS). The per-project path underneath it is
  derived automatically, never passed directly — there's no `--index-dir`
  flag in v1.
- `--exclude <pattern>` — additional exclusion glob, repeatable, merged with
  the default list above
- `--threads <n>` — walker thread count (default: `num_cpus`)
- `--embedding-model <dir>` — optional; local directory holding
  `potion-code-16M-v2`'s `config.json` + `tokenizer.json` +
  `model.safetensors` (see "Component 7"). Omit it (the default) to skip
  embeddings indexing entirely — no model load, no `embeddings.bin`.

## Error handling

- Unreadable file (permissions, race with deletion during walk): log a
  warning, skip, continue. Never abort the whole run for one bad file.
- Parse failure / tree-sitter error node: still index whatever identifiers
  were successfully extracted from the partial tree rather than discarding
  the file — tree-sitter is error-tolerant by design and produces a best-
  effort tree even for invalid syntax.
- Crash or interrupt mid-build: harmless by construction (see "Storage,"
  phase 3) — the previous good index, if any, is only ever replaced by an
  atomic directory/file swap after a build finishes successfully, so a
  partial build never corrupts or half-overwrites it. The orphaned temp
  build directory itself is not automatically cleaned up in v1.

## Crate dependencies (proposed)

| Purpose | Crate |
|---|---|
| Directory walk + exclusions | `ignore` |
| Binary sniffing | `content_inspector` |
| Parsing | `tree-sitter` + `tree-sitter-<lang>` per language |
| Term dictionary | `fst` |
| CLI parsing | `clap` |
| OS-appropriate default storage root | `dirs` |
| Tokenization (embeddings, Component 7) | `tokenizers` |
| Model weight loading (embeddings) | `safetensors` |
| `f16` ⟷ `f32` conversion (embeddings) | `half` |

`memmap2` isn't listed: it's the natural way a future *query* command would
mmap `terms.fst`/`postings.bin`, but v1 only builds the index (`fst::MapBuilder`
just needs an `io::Write`), so it isn't a dependency yet.

## Open questions

1. **Path key encoding**: relative-to-root vs. absolute path stored in
   `docs.bin` — relative is more portable if the index is ever moved
   alongside the source tree, absolute is simpler to implement first.
2. **`.md`/`.txt` fallback**: the word-tokenizer approach above is confirmed
   working (validated against the previous, RocksDB-backed implementation);
   carries over unchanged to this storage rewrite.
3. **Orphaned project index directories**: v1 has no command to list known
   projects or garbage-collect index directories whose project root no
   longer exists or has moved. Worth a `--list-projects` / `--gc` command
   later, or is manual cleanup of `storage-root` fine for now?
4. **Query/search interface**: out of scope per "Non-goals," but worth
   deciding when it becomes the next priority — the on-disk format above
   was chosen specifically to make it cheap to add later.
5. **BM25 searcher and embeddings similarity search**: both are now
   supported by the data this doc builds (term frequency + doc length in
   "Storage"; per-file vectors in "Component 7"), but neither the scoring
   formula nor a nearest-neighbor/ANN search is implemented — see
   "Non-goals."
6. **Chunk-level embeddings**: v1 embeds whole files (Component 7); once
   embeddings actually get *consumed* by something, function/chunk-level
   granularity is likely more useful for code search than whole-file, but
   needs an AST-based chunker this project doesn't have yet.
7. **Embedding model provisioning**: v1 requires the user to have already
   downloaded the model locally and point `--embedding-model` at it (see
   Component 7). Auto-downloading it on first use (cached under
   `storage-root`, presumably behind its own explicit opt-in flag) would be
   more convenient but was deliberately not built, to keep this tool's
   default behavior fully offline.
