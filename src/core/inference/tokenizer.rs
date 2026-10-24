use std::path::Path;
use ndarray::Array2;
use tokenizers::Tokenizer;
use crate::shared::error::AppError;

pub const DEFAULT_MAX_QUERY_LEN: usize = 512;
pub const DEFAULT_MAX_TOOL_LEN: usize = 96;

/// Tokenizer especializado para la familia multilingual-e5.
/// Implementa rigurosamente el protocolo de prefijos asimétricos ("query: " / "passage: ").
pub struct E5Tokenizer {
    inner: Tokenizer,
}

pub struct TokenizedBatch {
    pub input_ids: Array2<i64>,
    pub attention_mask: Array2<i64>,
    pub seq_len: usize,
}

impl E5Tokenizer {
    /// Carga el tokenizer desde un archivo `tokenizer.json`.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, AppError> {
        let tokenizer = Tokenizer::from_file(path.as_ref())
            .map_err(|e| AppError::Tokenizer(format!("Fallo al cargar tokenizer.json: {}", e)))?;
        Ok(Self { inner: tokenizer })
    }

    /// Codifica una consulta aplicando el prefijo "query: " y truncando a max_len.
    pub fn encode_query(&self, text: &str, max_len: usize) -> Result<TokenizedBatch, AppError> {
        let formatted = if text.starts_with("query: ") {
            text.to_string()
        } else {
            format!("query: {}", text.trim())
        };
        self.encode_single(&formatted, max_len)
    }

    /// Codifica la definición de una herramienta aplicando el prefijo "passage: ".
    pub fn encode_passage(&self, text: &str, max_len: usize) -> Result<TokenizedBatch, AppError> {
        let formatted = if text.starts_with("passage: ") {
            text.to_string()
        } else {
            format!("passage: {}", text.trim())
        };
        self.encode_single(&formatted, max_len)
    }

    /// Codificación interna hacia tensores contiguos Array2<i64> de dimensión (1, seq_len).
    fn encode_single(&self, text: &str, max_len: usize) -> Result<TokenizedBatch, AppError> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| AppError::Tokenizer(format!("Error de codificación: {}", e)))?;

        let raw_ids = encoding.get_ids();
        let seq_len = raw_ids.len().min(max_len);

        let mut input_ids = Array2::<i64>::zeros((1, seq_len));
        let mut attention_mask = Array2::<i64>::zeros((1, seq_len));

        for i in 0..seq_len {
            input_ids[[0, i]] = raw_ids[i] as i64;
            attention_mask[[0, i]] = 1i64;
        }

        Ok(TokenizedBatch {
            input_ids,
            attention_mask,
            seq_len,
        })
    }
}
