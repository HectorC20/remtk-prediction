use std::path::Path;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use tokenizers::Tokenizer;

pub struct E5Embedder {
    pub tokenizer: Tokenizer,
    pub model: BertModel,
    pub device: Device,
}

impl E5Embedder {
    pub fn load(model_dir: &Path, dev: &Device) -> Result<Self> {
        let tok_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenizer error: {}", e)))?;

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| candle_core::Error::Msg(format!("Read config: {}", e)))?;
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| candle_core::Error::Msg(format!("Parse config: {}", e)))?;

        let weights_path = model_dir.join("model.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, dev)? };
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            tokenizer,
            model,
            device: dev.clone(),
        })
    }

    pub fn encode_single(&self, text: &str, prefix: &str, max_len: usize) -> Result<(Tensor, Tensor)> {
        let clean = text.trim();
        let full_text = if clean.starts_with(prefix) {
            clean.to_string()
        } else {
            format!("{}{}", prefix, clean)
        };

        let encoding = self.tokenizer.encode(full_text.as_str(), true)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenizer encode error: {}", e)))?;

        let mut ids: Vec<u32> = encoding.get_ids().to_vec();
        let mut att_mask: Vec<f32> = encoding.get_attention_mask().iter().map(|&x| x as f32).collect();

        // Truncate to max_len
        if ids.len() > max_len {
            ids.truncate(max_len);
            att_mask.truncate(max_len);
        } else {
            // Pad with pad_id = 0 to max_len
            while ids.len() < max_len {
                ids.push(0);
                att_mask.push(0.0);
            }
        }

        let input_ids = Tensor::from_slice(&ids, (1, max_len), &self.device)?;
        let token_type_ids = Tensor::zeros((1, max_len), DType::U32, &self.device)?;
        let mask_tensor = Tensor::from_slice(&att_mask, (1, max_len, 1), &self.device)?; // [1, max_len, 1]

        let out = self.model.forward(&input_ids, &token_type_ids, None)?; // [1, max_len, 384]

        // F.normalize(out, p=2, dim=-1) * mask
        let eps_scalar = Tensor::new(&[1e-6f32], &self.device)?;
        let norms = out.sqr()?.sum_keepdim(2)?.sqrt()?.broadcast_maximum(&eps_scalar)?;
        let normed_tokens = out.broadcast_div(&norms)?.broadcast_mul(&mask_tensor)?;

        // lens = mask.sum(dim=1).clamp(min=1.0)
        let one_scalar = Tensor::new(&[1.0f32], &self.device)?;
        let lens = mask_tensor.sum_keepdim(1)?.broadcast_maximum(&one_scalar)?; // [1, 1, 1]

        // pooled = F.normalize((normed_tokens * mask).sum(dim=1) / lens, p=2, dim=-1)
        let sum_tokens = normed_tokens.sum_keepdim(1)?; // [1, 1, 384]
        let mean_pooled = sum_tokens.broadcast_div(&lens)?.squeeze(1)?; // [1, 384]
        let p_norm = mean_pooled.sqr()?.sum_keepdim(1)?.sqrt()?.broadcast_maximum(&eps_scalar)?;
        let pooled = mean_pooled.broadcast_div(&p_norm)?; // [1, 384]

        Ok((normed_tokens, pooled))
    }
}
