use std::collections::HashMap;
use std::fs;
use std::path::Path;

use half::f16;
use safetensors::tensor::Dtype;
use safetensors::SafeTensors;
use tokenizers::Tokenizer;

const EMBEDDINGS_TENSOR: &str = "embeddings";
const DEFAULT_MAX_LENGTH: usize = 512;
const UNK_TOKEN: &str = "[UNK]";

/// A loaded `potion-code-16M-v2`-shaped Model2Vec static embedding model:
/// a tokenizer plus a flat, in-memory embedding table (`vocab_size x dim`,
/// row-major, decoded to `f32`). See DESIGN.md's "Component 7" for the
/// verified on-disk format and inference algorithm this reimplements.
pub struct EmbeddingModel {
    tokenizer: Tokenizer,
    table: Vec<f32>,
    dim: usize,
    normalize: bool,
    max_length: usize,
    median_token_length: usize,
    unk_id: Option<u32>,
}

impl EmbeddingModel {
    /// Loads `config.json` + `tokenizer.json` + `model.safetensors` from
    /// `dir` (see DESIGN.md's `--embedding-model` flag).
    pub fn load(dir: &Path) -> Result<EmbeddingModel, Box<dyn std::error::Error>> {
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;

        let config_bytes = fs::read(dir.join("config.json"))?;
        let config: HashMap<String, serde_json::Value> = serde_json::from_slice(&config_bytes)?;
        let normalize = config
            .get("normalize")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let weights = fs::read(dir.join("model.safetensors"))?;
        let tensors = SafeTensors::deserialize(&weights)?;
        let view = tensors.tensor(EMBEDDINGS_TENSOR)?;
        if view.dtype() != Dtype::F16 {
            return Err(format!(
                "expected {EMBEDDINGS_TENSOR} tensor dtype F16, got {:?}",
                view.dtype()
            )
            .into());
        }
        let (vocab_size, dim) = match view.shape() {
            [v, d] => (*v, *d),
            shape => {
                return Err(format!("expected a 2D {EMBEDDINGS_TENSOR} tensor, got shape {shape:?}").into());
            }
        };
        let data = view.data();
        if data.len() != vocab_size * dim * 2 {
            return Err("embeddings tensor byte length doesn't match its declared shape".into());
        }
        let mut table = Vec::with_capacity(vocab_size * dim);
        for chunk in data.as_chunks::<2>().0 {
            table.push(f16::from_le_bytes(*chunk).to_f32());
        }

        let unk_id = tokenizer.token_to_id(UNK_TOKEN);
        let median_token_length = median_token_length(&tokenizer);

        Ok(EmbeddingModel {
            tokenizer,
            table,
            dim,
            normalize,
            max_length: DEFAULT_MAX_LENGTH,
            median_token_length,
            unk_id,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Computes one file's embedding: tokenize (no special tokens, `[UNK]`
    /// ids dropped), mean-pool the looked-up rows, then L2-normalize if the
    /// model config asks for it (see DESIGN.md's "Component 7" for the
    /// verified algorithm this mirrors). Empty/all-unk input yields an
    /// all-zero vector rather than a failure, per "Error handling."
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let char_limit = self.max_length * self.median_token_length;
        let truncated: String = if text.chars().count() > char_limit {
            text.chars().take(char_limit).collect()
        } else {
            text.to_string()
        };

        let mut sum = vec![0f32; self.dim];
        let mut count = 0u32;
        if let Ok(encoding) = self.tokenizer.encode(truncated.as_str(), false) {
            for &id in encoding.get_ids() {
                if Some(id) == self.unk_id {
                    continue;
                }
                let row_start = id as usize * self.dim;
                let Some(row) = self.table.get(row_start..row_start + self.dim) else {
                    continue;
                };
                for (s, v) in sum.iter_mut().zip(row) {
                    *s += v;
                }
                count += 1;
            }
        }

        if count > 0 {
            let count = count as f32;
            for v in &mut sum {
                *v /= count;
            }
        }

        if self.normalize {
            let norm = sum.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-32;
            for v in &mut sum {
                *v /= norm;
            }
        }

        sum
    }
}

/// Median character length of every token string in the tokenizer's
/// vocabulary -- used, like the upstream `model2vec` Python package, to
/// bound tokenization cost on very long input by truncating *characters*
/// before ever calling the tokenizer, rather than tokenizing first and
/// truncating the (already fully computed) token list.
fn median_token_length(tokenizer: &Tokenizer) -> usize {
    let mut lengths: Vec<usize> = tokenizer
        .get_vocab(false)
        .keys()
        .map(|t| t.chars().count())
        .collect();
    lengths.sort_unstable();
    lengths.get(lengths.len() / 2).copied().unwrap_or(1).max(1)
}
