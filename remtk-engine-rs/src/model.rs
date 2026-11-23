use std::collections::HashMap;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, LayerNorm, Module};

pub fn silu(xs: &Tensor) -> Result<Tensor> {
    xs.mul(&candle_nn::ops::sigmoid(xs)?)
}

pub struct SwiGLUProjector {
    pub w_gate: Linear,
    pub w_up: Linear,
    pub w_down: Linear,
    pub norm: LayerNorm,
}

impl SwiGLUProjector {
    pub fn load(prefix: &str, weights: &HashMap<String, Tensor>, _dev: &Device) -> Result<Self> {
        let wg_w = weights.get(&format!("{}.w_gate.weight", prefix)).unwrap();
        let wg_b = weights.get(&format!("{}.w_gate.bias", prefix)).cloned();
        let wu_w = weights.get(&format!("{}.w_up.weight", prefix)).unwrap();
        let wu_b = weights.get(&format!("{}.w_up.bias", prefix)).cloned();
        let wd_w = weights.get(&format!("{}.w_down.weight", prefix)).unwrap();
        let wd_b = weights.get(&format!("{}.w_down.bias", prefix)).cloned();
        let n_w = weights.get(&format!("{}.norm.weight", prefix)).unwrap();
        let n_b = weights.get(&format!("{}.norm.bias", prefix)).unwrap();

        Ok(Self {
            w_gate: Linear::new(wg_w.clone(), wg_b),
            w_up: Linear::new(wu_w.clone(), wu_b),
            w_down: Linear::new(wd_w.clone(), wd_b),
            norm: LayerNorm::new(n_w.clone(), n_b.clone(), 1e-5),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = silu(&self.w_gate.forward(x)?)?;
        let up = self.w_up.forward(x)?;
        let h = gate.mul(&up)?;
        let down = self.w_down.forward(&h)?;
        self.norm.forward(&down)
    }
}

pub struct CandidateDifferentialInjector {
    pub proj: SwiGLUProjector,
}

impl CandidateDifferentialInjector {
    pub fn load(prefix: &str, weights: &HashMap<String, Tensor>, dev: &Device) -> Result<Self> {
        let proj = SwiGLUProjector::load(&format!("{}.proj", prefix), weights, dev)?;
        Ok(Self { proj })
    }

    pub fn forward(&self, q: &Tensor, t: &Tensor) -> Result<Tensor> {
        // q: [B, 384], t: [B, K, 384]
        let (_b, _k, _d) = t.dims3()?;
        let q_exp = q.unsqueeze(1)?.broadcast_as(t.shape())?; // [B, K, 384]
        let q_diff = q_exp.sub(t)?;
        let q_prod = q_exp.mul(t)?;
        let diff = Tensor::cat(&[&q_exp, t, &q_diff, &q_prod], 2)?; // [B, K, 1536]
        self.proj.forward(&diff)
    }
}

pub struct PreLNSetAttentionLayer {
    pub norm1: LayerNorm,
    pub in_proj_w: Tensor,
    pub in_proj_b: Tensor,
    pub out_proj: Linear,
    pub norm2: LayerNorm,
    pub ffn_0: Linear,
    pub ffn_3: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl PreLNSetAttentionLayer {
    pub fn load(prefix: &str, weights: &HashMap<String, Tensor>, _dev: &Device) -> Result<Self> {
        let n1_w = weights.get(&format!("{}.norm1.weight", prefix)).unwrap();
        let n1_b = weights.get(&format!("{}.norm1.bias", prefix)).unwrap();
        let in_w = weights.get(&format!("{}.attn.in_proj_weight", prefix)).unwrap();
        let in_b = weights.get(&format!("{}.attn.in_proj_bias", prefix)).unwrap();
        let out_w = weights.get(&format!("{}.attn.out_proj.weight", prefix)).unwrap();
        let out_b = weights.get(&format!("{}.attn.out_proj.bias", prefix)).cloned();
        let n2_w = weights.get(&format!("{}.norm2.weight", prefix)).unwrap();
        let n2_b = weights.get(&format!("{}.norm2.bias", prefix)).unwrap();
        let f0_w = weights.get(&format!("{}.ffn.0.weight", prefix)).unwrap();
        let f0_b = weights.get(&format!("{}.ffn.0.bias", prefix)).cloned();
        let f3_w = weights.get(&format!("{}.ffn.3.weight", prefix)).unwrap();
        let f3_b = weights.get(&format!("{}.ffn.3.bias", prefix)).cloned();

        let d_moe = 2048;
        let num_heads = 8;
        let head_dim = d_moe / num_heads;

        Ok(Self {
            norm1: LayerNorm::new(n1_w.clone(), n1_b.clone(), 1e-5),
            in_proj_w: in_w.clone(),
            in_proj_b: in_b.clone(),
            out_proj: Linear::new(out_w.clone(), out_b),
            norm2: LayerNorm::new(n2_w.clone(), n2_b.clone(), 1e-5),
            ffn_0: Linear::new(f0_w.clone(), f0_b),
            ffn_3: Linear::new(f3_w.clone(), f3_b),
            num_heads,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normed = self.norm1.forward(x)?;
        let (b, s, d) = normed.dims3()?;

        // in_proj
        let qkv = normed.broadcast_matmul(&self.in_proj_w.t()?)?.broadcast_add(&self.in_proj_b)?;
        let chunks = qkv.chunk(3, 2)?;
        let (q, k, v) = (&chunks[0], &chunks[1], &chunks[2]);

        // multi-head reshape: [B, num_heads, S, head_dim]
        let q_h = q.reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k_h = k.reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v_h = v.reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q_h.matmul(&k_h.transpose(2, 3)?)? * scale)?;
        let attn_weights = candle_nn::ops::softmax(&scores, 3)?;
        let attn_out = attn_weights.matmul(&v_h)?;
        let attn_flat = attn_out.transpose(1, 2)?.reshape((b, s, d))?;
        let proj_out = self.out_proj.forward(&attn_flat)?;

        let res1 = (x + proj_out)?;
        let normed2 = self.norm2.forward(&res1)?;
        let ffn_h = silu(&self.ffn_0.forward(&normed2)?)?;
        let ffn_out = self.ffn_3.forward(&ffn_h)?;
        &res1 + ffn_out
    }
}

pub struct UltraSparseVectorizedMoEBank {
    pub norm: LayerNorm,
    pub router: Linear,
    pub w_gate: Tensor, // [128, 2048, 4528] (F32 for CPU calculation)
    pub w_up: Tensor,   // [128, 2048, 4528]
    pub w_down: Tensor, // [128, 4528, 2048]
    pub top_k: usize,
}

impl UltraSparseVectorizedMoEBank {
    pub fn load(prefix: &str, weights: &HashMap<String, Tensor>, _dev: &Device) -> Result<Self> {
        let n_w = weights.get(&format!("{}.norm.weight", prefix)).unwrap();
        let n_b = weights.get(&format!("{}.norm.bias", prefix)).unwrap();
        let r_w = weights.get(&format!("{}.router.weight", prefix)).unwrap();
        let wg = weights.get(&format!("{}.w_gate", prefix)).unwrap().clone();
        let wu = weights.get(&format!("{}.w_up", prefix)).unwrap().clone();
        let wd = weights.get(&format!("{}.w_down", prefix)).unwrap().clone();

        Ok(Self {
            norm: LayerNorm::new(n_w.clone(), n_b.clone(), 1e-5),
            router: Linear::new(r_w.clone(), None),
            w_gate: wg,
            w_up: wu,
            w_down: wd,
            top_k: 4,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, d) = x.dims3()?;
        let normed = self.norm.forward(x)?;
        let x_flat = normed.reshape((b * s, d))?;

        // Router logits: [B*S, 128]
        let router_logits = self.router.forward(&x_flat)?;
        let router_probs = candle_nn::ops::softmax(&router_logits, 1)?;

        let num_tokens = b * s;
        let mut out_flat = Tensor::zeros((num_tokens, d), DType::F32, x.device())?;

        // Extract top-4 experts per token in Rust
        let probs_vec = router_probs.to_vec2::<f32>()?;
        let x_vec = x_flat.to_vec2::<f32>()?;

        // Vectorized MoE execution per expert
        for t_idx in 0..num_tokens {
            let p_row = &probs_vec[t_idx];
            let mut indexed_probs: Vec<(usize, f32)> = p_row.iter().cloned().enumerate().collect();
            indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let top4 = &indexed_probs[0..self.top_k];
            let sum_w: f32 = top4.iter().map(|(_, w)| *w).sum::<f32>() + 1e-6;

            let token_tensor = Tensor::from_slice(&x_vec[t_idx], (1, d), x.device())?;
            let mut token_accum = Tensor::zeros((1, d), DType::F32, x.device())?;

            for &(exp_id, weight) in top4 {
                let norm_weight = weight / sum_w;
                let wg = self.w_gate.get(exp_id)?.to_dtype(DType::F32)?; // [2048, 4528]
                let wu = self.w_up.get(exp_id)?.to_dtype(DType::F32)?;   // [2048, 4528]
                let wd = self.w_down.get(exp_id)?.to_dtype(DType::F32)?; // [4528, 2048]

                let g = silu(&token_tensor.matmul(&wg)?)?;
                let u = token_tensor.matmul(&wu)?;
                let h = g.mul(&u)?;
                let expert_out = h.matmul(&wd)?;
                let scaled = (expert_out * (norm_weight as f64))?;
                token_accum = (token_accum + scaled)?;
            }

            // Scatter back
            out_flat = out_flat.slice_assign(&[t_idx..t_idx+1, 0..d], &token_accum)?;
        }

        let moe_seq = out_flat.reshape((b, s, d))?;
        x + moe_seq
    }
}

pub struct JudgmentCore {
    pub judgment_net: SwiGLUProjector,
    pub judgment_head: Linear,
}

impl JudgmentCore {
    pub fn load(prefix: &str, weights: &HashMap<String, Tensor>, dev: &Device) -> Result<Self> {
        let j_net = SwiGLUProjector::load(&format!("{}.judgment_net", prefix), weights, dev)?;
        let jh_w = weights.get(&format!("{}.judgment_head.weight", prefix)).unwrap();
        let jh_b = weights.get(&format!("{}.judgment_head.bias", prefix)).cloned();
        Ok(Self {
            judgment_net: j_net,
            judgment_head: Linear::new(jh_w.clone(), jh_b),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let h = self.judgment_net.forward(x)?;
        let logits = self.judgment_head.forward(&h)?;
        let probs = candle_nn::ops::softmax(&logits, 1)?;
        Ok((h, probs))
    }
}

pub struct ColBERTSubwordLateInteraction {
    pub q_proj: Linear,
    pub t_proj: Linear,
}

impl ColBERTSubwordLateInteraction {
    pub fn load(prefix: &str, weights: &HashMap<String, Tensor>, _dev: &Device) -> Result<Self> {
        let qw = weights.get(&format!("{}.q_proj.weight", prefix)).unwrap();
        let tw = weights.get(&format!("{}.t_proj.weight", prefix)).unwrap();
        Ok(Self {
            q_proj: Linear::new(qw.clone(), None),
            t_proj: Linear::new(tw.clone(), None),
        })
    }

    pub fn score(&self, q_tokens: &Tensor, t_tokens: &Tensor) -> Result<Tensor> {
        // q_tokens: [B, Q, 384], t_tokens: [B, K, T, 384]
        let eps_s = Tensor::new(&[1e-6f32], q_tokens.device())?;
        let q_p = self.q_proj.forward(q_tokens)?; // [B, Q, 128]
        let q_norm = q_p.broadcast_div(&q_p.sqr()?.sum_keepdim(2)?.sqrt()?.broadcast_maximum(&eps_s)?)?;

        let t_p = self.t_proj.forward(t_tokens)?; // [B, K, T, 128]
        let t_norm = t_p.broadcast_div(&t_p.sqr()?.sum_keepdim(3)?.sqrt()?.broadcast_maximum(&eps_s)?)?;

        // sim = einsum('b q d, b k t d -> b k q t')
        let (b, k, _t, _d) = t_norm.dims4()?;
        let (_b, _q, _) = q_norm.dims3()?;

        let mut colbert_scores = Vec::with_capacity(k);
        for k_idx in 0..k {
            let t_cand = t_norm.get(0)?.get(k_idx)?; // [T, 128]
            let sim_qt = q_norm.get(0)?.matmul(&t_cand.t()?)?; // [Q, T]
            let max_t = sim_qt.max(1)?; // [Q]
            let mean_q = max_t.mean(0)?; // scalar
            colbert_scores.push(mean_q.to_scalar::<f32>()?);
        }

        Tensor::from_slice(&colbert_scores, (b, k), q_tokens.device())
    }
}

pub struct RemtkModel {
    pub noul_sink_token: Tensor,
    pub colbert_weight: f32,
    pub colbert: ColBERTSubwordLateInteraction,
    pub diff_injector: CandidateDifferentialInjector,
    pub h1_to_moe: Linear,
    pub sink_context_proj: Linear,
    pub set_attn_layers: Vec<PreLNSetAttentionLayer>,
    pub moe_bank: UltraSparseVectorizedMoEBank,
    pub candidate_logit_head: Linear,
    pub sink_logit_head: Linear,
    pub judgment_core: JudgmentCore,
    pub noul_intent_head_0: Linear,
    pub noul_intent_head_2: Linear,
    pub noul_coverage_head_0: Linear,
    pub noul_coverage_head_2: Linear,
    pub route_head_0: Linear,
    pub route_head_1: LayerNorm,
    pub route_head_3: Linear,
}

impl RemtkModel {
    pub fn load_from_safetensors(path: &std::path::Path, dev: &Device) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let mmap = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };
        println!("[Model] Safetensors mapped in {:.2}ms", t0.elapsed().as_secs_f64() * 1000.0);

        let mut weights = HashMap::new();
        for (name, _) in mmap.tensors() {
            let tensor = mmap.load(&name, dev)?;
            weights.insert(name, tensor);
        }
        println!("[Model] Loaded {} tensors into memory-mapped structures", weights.len());
        Self::load(&weights, dev)
    }

    pub fn load(weights: &HashMap<String, Tensor>, dev: &Device) -> Result<Self> {
        let sink = weights.get("noul_sink_token").unwrap().clone();
        let cw = weights.get("colbert_weight").map(|t| t.to_scalar::<f32>().unwrap_or(2.5)).unwrap_or(2.5);

        let colbert = ColBERTSubwordLateInteraction::load("colbert", weights, dev)?;
        let diff_injector = CandidateDifferentialInjector::load("diff_injector", weights, dev)?;

        let h1_w = weights.get("h1_to_moe.weight").unwrap();
        let h1_b = weights.get("h1_to_moe.bias").cloned();
        let h1_to_moe = Linear::new(h1_w.clone(), h1_b);

        let sc_w = weights.get("sink_context_proj.weight").unwrap();
        let sc_b = weights.get("sink_context_proj.bias").cloned();
        let sink_context_proj = Linear::new(sc_w.clone(), sc_b);

        let mut set_attn_layers = Vec::with_capacity(6);
        for i in 0..6 {
            set_attn_layers.push(PreLNSetAttentionLayer::load(&format!("set_attn_layers.{}", i), weights, dev)?);
        }

        let moe_bank = UltraSparseVectorizedMoEBank::load("moe_bank", weights, dev)?;

        let cl_w = weights.get("candidate_logit_head.weight").unwrap();
        let cl_b = weights.get("candidate_logit_head.bias").cloned();
        let candidate_logit_head = Linear::new(cl_w.clone(), cl_b);

        let sl_w = weights.get("sink_logit_head.weight").unwrap();
        let sl_b = weights.get("sink_logit_head.bias").cloned();
        let sink_logit_head = Linear::new(sl_w.clone(), sl_b);

        let judgment_core = JudgmentCore::load("judgment_core", weights, dev)?;

        let ni0_w = weights.get("noul_intent_head.0.weight").unwrap();
        let ni0_b = weights.get("noul_intent_head.0.bias").cloned();
        let ni2_w = weights.get("noul_intent_head.2.weight").unwrap();
        let ni2_b = weights.get("noul_intent_head.2.bias").cloned();
        let noul_intent_head_0 = Linear::new(ni0_w.clone(), ni0_b);
        let noul_intent_head_2 = Linear::new(ni2_w.clone(), ni2_b);

        let nc0_w = weights.get("noul_coverage_head.0.weight").unwrap();
        let nc0_b = weights.get("noul_coverage_head.0.bias").cloned();
        let nc2_w = weights.get("noul_coverage_head.2.weight").unwrap();
        let nc2_b = weights.get("noul_coverage_head.2.bias").cloned();
        let noul_coverage_head_0 = Linear::new(nc0_w.clone(), nc0_b);
        let noul_coverage_head_2 = Linear::new(nc2_w.clone(), nc2_b);

        let rh0_w = weights.get("route_head.0.weight").unwrap();
        let rh0_b = weights.get("route_head.0.bias").cloned();
        let rh1_w = weights.get("route_head.1.weight").unwrap();
        let rh1_b = weights.get("route_head.1.bias").unwrap();
        let rh3_w = weights.get("route_head.3.weight").unwrap();
        let rh3_b = weights.get("route_head.3.bias").cloned();
        let route_head_0 = Linear::new(rh0_w.clone(), rh0_b);
        let route_head_1 = LayerNorm::new(rh1_w.clone(), rh1_b.clone(), 1e-5);
        let route_head_3 = Linear::new(rh3_w.clone(), rh3_b);

        Ok(Self {
            noul_sink_token: sink,
            colbert_weight: cw,
            colbert,
            diff_injector,
            h1_to_moe,
            sink_context_proj,
            set_attn_layers,
            moe_bank,
            candidate_logit_head,
            sink_logit_head,
            judgment_core,
            noul_intent_head_0,
            noul_intent_head_2,
            noul_coverage_head_0,
            noul_coverage_head_2,
            route_head_0,
            route_head_1,
            route_head_3,
        })
    }

    pub fn forward(
        &self,
        q_tokens: &Tensor,
        q_pooled: &Tensor,
        t_tokens: &Tensor,
        t_pooled: &Tensor,
    ) -> Result<RemtkOutput> {
        let (b, _k, _d) = t_pooled.dims3()?;

        // ColBERT scores: [B, K]
        let colbert_scores = self.colbert.score(q_tokens, t_tokens)?;

        // Candidate differential embeddings
        let cand_moe_in = self.diff_injector.forward(q_pooled, t_pooled)?; // [B, K, 2048]
        let h1_dummy = Tensor::zeros((b, 1024), DType::F32, q_pooled.device())?;
        let h1_proj = self.h1_to_moe.forward(&h1_dummy)?.unsqueeze(1)?; // [B, 1, 2048]
        let cand_moe_in = cand_moe_in.broadcast_add(&h1_proj)?;

        let disc_probs = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 0.0, 0.0], (b, 5), q_pooled.device())?;
        let red_prob = Tensor::zeros((b, 1), DType::F32, q_pooled.device())?;
        let sink_ctx_in = Tensor::cat(&[&disc_probs, &red_prob], 1)?; // [B, 6]
        let sink_ctx = self.sink_context_proj.forward(&sink_ctx_in)?.unsqueeze(1)?; // [B, 1, 2048]

        let sink_token = (self.noul_sink_token.broadcast_as((b, 1, 2048))? + &h1_proj)? + &sink_ctx;
        let mut set_seq = Tensor::cat(&[&cand_moe_in, &sink_token?], 1)?; // [B, K+1, 2048]

        for layer in &self.set_attn_layers {
            set_seq = layer.forward(&set_seq)?;
        }

        let moe_seq = self.moe_bank.forward(&set_seq)?;
        let (_b, s, _d) = moe_seq.dims3()?;
        let cand_moe_out = moe_seq.narrow(1, 0, s - 1)?;
        let sink_moe_out = moe_seq.narrow(1, s - 1, 1)?.squeeze(1)?;

        let raw_cand_l = (self.candidate_logit_head.forward(&cand_moe_out)?.squeeze(2)? + (&colbert_scores * (self.colbert_weight as f64))?)?;
        let sink_l = self.sink_logit_head.forward(&sink_moe_out)?;

        let all_logits = Tensor::cat(&[&raw_cand_l, &sink_l], 1)?;
        let all_probs = candle_nn::ops::softmax(&all_logits, 1)?;
        let p_sink = all_probs.narrow(1, s - 1, 1)?;

        let cand_max_pool = cand_moe_out.max(1)?;
        let cand_mean_pool = cand_moe_out.mean(1)?;
        let colbert_max = colbert_scores.max(1)?.unsqueeze(1)?;

        let j_in = Tensor::cat(&[
            &cand_max_pool, &cand_mean_pool, &sink_moe_out, &h1_dummy,
            &disc_probs, &red_prob, &p_sink, &colbert_max
        ], 1)?;

        let (j_h, j_probs) = self.judgment_core.forward(&j_in)?;

        let noul_intent_in = Tensor::cat(&[&j_h, &disc_probs, &red_prob, &j_probs], 1)?;
        let p_noul_intent = candle_nn::ops::sigmoid(&self.noul_intent_head_2.forward(&silu(&self.noul_intent_head_0.forward(&noul_intent_in)?)?)?)?;

        let noul_cov_in = Tensor::cat(&[&j_h, &sink_moe_out, &p_sink, &colbert_max], 1)?;
        let p_noul_cov_raw = candle_nn::ops::sigmoid(&self.noul_coverage_head_2.forward(&silu(&self.noul_coverage_head_0.forward(&noul_cov_in)?)?)?)?;
        let one = Tensor::new(&[[1.0f32]], q_pooled.device())?;
        let p_noul_cov = p_noul_cov_raw.mul(&one.sub(&p_sink)?)?;

        let route_in = Tensor::cat(&[&j_h, &disc_probs, &red_prob, &j_probs, &p_sink, &p_noul_intent, &p_noul_cov], 1)?;
        let r_h = silu(&self.route_head_1.forward(&self.route_head_0.forward(&route_in)?)?)?;
        let route_probs = candle_nn::ops::softmax(&self.route_head_3.forward(&r_h)?, 1)?;
        let p_route_tool = route_probs.narrow(1, 0, 1)?;

        let raw_aff = candle_nn::ops::sigmoid(&raw_cand_l)?;
        let master_gate = p_route_tool.mul(&p_noul_intent)?.mul(&p_noul_cov)?;
        let gated_scores = raw_aff.broadcast_mul(&master_gate)?;

        Ok(RemtkOutput {
            gated_scores: gated_scores.squeeze(0)?.to_vec1::<f32>()?,
            raw_affinities: raw_aff.squeeze(0)?.to_vec1::<f32>()?,
            master_gate: master_gate.to_vec2::<f32>()?[0][0],
            p_sink: p_sink.to_vec2::<f32>()?[0][0],
            p_noul_intent: p_noul_intent.to_vec2::<f32>()?[0][0],
            p_noul_coverage: p_noul_cov.to_vec2::<f32>()?[0][0],
            redundancy_prob: 0.0,
        })
    }
}

pub struct RemtkOutput {
    pub gated_scores: Vec<f32>,
    pub raw_affinities: Vec<f32>,
    pub master_gate: f32,
    pub p_sink: f32,
    pub p_noul_intent: f32,
    pub p_noul_coverage: f32,
    pub redundancy_prob: f32,
}
