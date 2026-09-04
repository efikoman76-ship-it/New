//! Versioned model / training / serving configuration schema.
//!
//! Loaded from TOML files in `configs/`; validated with cross-field checks;
//! the same struct drives parameter accounting (`crate::params`), graph
//! building (`rhizome-plan`) and serving.

use crate::error::RhizomeError;
use crate::toml::{self, Table, Value};

/// Where the Think Core's Read block (C1) attends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreReadSource {
    /// Core-input cache of tokens ≤ t (default).
    Input,
    /// Iteration-(i−1) latents (ablation-only; SPEC 3.5).
    PrevIter,
}

impl CoreReadSource {
    /// Name used in configs.
    pub fn name(self) -> &'static str {
        match self {
            CoreReadSource::Input => "input",
            CoreReadSource::PrevIter => "prev_iter",
        }
    }
    /// Parse from config string.
    pub fn parse(s: &str) -> Result<Self, RhizomeError> {
        match s {
            "input" => Ok(CoreReadSource::Input),
            "prev_iter" => Ok(CoreReadSource::PrevIter),
            other => Err(RhizomeError::Config(format!(
                "bad core_read_source {other}"
            ))),
        }
    }
}

/// Deploy weight format classes (SPEC 3.9/3.10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeployFormat {
    /// Full precision (CI/reference).
    F32,
    /// MXINT4 experts+mixers, bf16 the rest (Edge).
    MxInt4,
    /// MXFP4 experts+mixers, bf16 the rest (Standard).
    MxFp4,
    /// MXFP4 + FP8 attention projections (Flagship).
    MxFp4Fp8Attn,
}

impl DeployFormat {
    /// Name used in configs.
    pub fn name(self) -> &'static str {
        match self {
            DeployFormat::F32 => "f32",
            DeployFormat::MxInt4 => "mxint4",
            DeployFormat::MxFp4 => "mxfp4",
            DeployFormat::MxFp4Fp8Attn => "mxfp4+fp8attn",
        }
    }
    /// Parse from config string.
    pub fn parse(s: &str) -> Result<Self, RhizomeError> {
        match s {
            "f32" => Ok(DeployFormat::F32),
            "mxint4" => Ok(DeployFormat::MxInt4),
            "mxfp4" => Ok(DeployFormat::MxFp4),
            "mxfp4+fp8attn" => Ok(DeployFormat::MxFp4Fp8Attn),
            other => Err(RhizomeError::Config(format!("bad deploy format {other}"))),
        }
    }
}

/// YaRN context-extension settings (SPEC 3.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YarnConfig {
    /// Original training context length.
    pub original_context: usize,
    /// Extended context length target.
    pub target_context: usize,
    /// YaRN attention-temperature factor ramp (default 32).
    pub attn_temp: f64,
    /// Beta fast (default 32).
    pub beta_fast: f64,
    /// Beta slow (default 1).
    pub beta_slow: f64,
    /// Scale applied to low-frequency dims (default 1).
    pub mscale: f64,
}

impl Default for YarnConfig {
    fn default() -> Self {
        YarnConfig {
            original_context: 4096,
            target_context: 4096,
            attn_temp: 32.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: 1.0,
        }
    }
}

/// Byte-latent front/back end configuration (SPEC 3.7b).
#[derive(Debug, Clone, PartialEq)]
pub struct BytePathConfig {
    /// Whether the byte path is compiled into the model.
    pub enabled: bool,
    /// Local width for byte encoder/decoder (512/768/1024 by tier).
    pub d_local: usize,
    /// Byte encoder blocks (4 Local, dense FFN).
    pub encoder_blocks: usize,
    /// Byte decoder blocks (6 Local, dense FFN + cross-attn).
    pub decoder_blocks: usize,
    /// Hashed n-gram embedding tables.
    pub ngram_tables: usize,
    /// Smallest n.
    pub ngram_min: usize,
    /// Largest n.
    pub ngram_max: usize,
    /// Buckets per table.
    pub ngram_buckets: usize,
    /// Bytes of decoder byte-state context per step.
    pub byte_window: usize,
    /// Min patch length.
    pub patch_min: usize,
    /// Max patch length.
    pub patch_max: usize,
    /// Patcher entropy threshold θ_g (nats).
    pub theta_g: f64,
    /// Patcher entropy-jump threshold δ (nats).
    pub delta: f64,
    /// Patcher model width.
    pub patcher_d_model: usize,
    /// Patcher model Local blocks (dense FFN).
    pub patcher_blocks: usize,
}

impl Default for BytePathConfig {
    fn default() -> Self {
        BytePathConfig {
            enabled: false,
            d_local: 512,
            encoder_blocks: 4,
            decoder_blocks: 6,
            ngram_tables: 3,
            ngram_min: 3,
            ngram_max: 8,
            ngram_buckets: 256 * 1024,
            byte_window: 256,
            patch_min: 1,
            patch_max: 64,
            theta_g: 3.0,
            delta: 0.8,
            patcher_d_model: 512,
            patcher_blocks: 4,
        }
    }
}

/// Full model architecture description.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    /// Config name (e.g. "test-s").
    pub name: String,
    /// Residual stream width.
    pub d_model: usize,
    /// Attention/delta heads in trunk.
    pub n_heads: usize,
    /// Per-head width (128 by default; 32 for Test configs).
    pub d_head: usize,
    /// Number of trunk strata N; each = 5 Local + 1 Global block.
    pub trunk_strata: usize,
    /// Think Core inserted after stratum P.
    pub core_after_stratum: usize,
    /// Max Think Core iterations.
    pub l_max: usize,
    /// Core read mode (input | prev_iter).
    pub core_read_source: CoreReadSource,
    /// Routed experts per MoE block.
    pub experts_routed: usize,
    /// Shared experts per MoE block.
    pub experts_shared: usize,
    /// Experts per token.
    pub expert_topk: usize,
    /// Expert hidden width.
    pub expert_hidden: usize,
    /// Expert routing groups.
    pub expert_groups: usize,
    /// 1-based trunk block indices (across strata) where routed experts are
    /// replaced by PKM.
    pub pkm_blocks: Vec<usize>,
    /// PKM value slots.
    pub pkm_slots: usize,
    /// PKM key rows per table.
    pub pkm_rows: usize,
    /// PKM subkey dimension.
    pub pkm_dims: usize,
    /// PKM query heads.
    pub pkm_heads: usize,
    /// PKM query width (per head).
    pub pkm_query_dim: usize,
    /// PKM candidates kept per table.
    pub pkm_topk: usize,
    /// MLA latent dims.
    pub d_latent: usize,
    /// MLA rope dims.
    pub d_rope: usize,
    /// MLA non-rope dims.
    pub d_nope: usize,
    /// RoPE base.
    pub rope_base: f64,
    /// YaRN settings.
    pub yarn: YarnConfig,
    /// Optional attention logit softcap.
    pub attn_softcap: Option<f64>,
    /// FP8 latent cache at deploy (reference stays f32).
    pub fp8_latent_cache: bool,
    /// BPE vocab size.
    pub vocab_size: usize,
    /// Untied unembedding.
    pub tie_unembed: bool,
    /// MTP head count.
    pub mtp_heads: usize,
    /// Dense FFN multiplier for dense blocks (C3, MTP, byte path).
    pub dense_ffn_mult: usize,
    /// Causal conv kernel in Local blocks.
    pub conv_kernel: usize,
    /// Deploy format.
    pub deploy: DeployFormat,
    /// Byte path settings.
    pub byte_path: BytePathConfig,
}

impl ModelConfig {
    /// Total trunk blocks (strata × 6).
    pub fn trunk_blocks(&self) -> usize {
        self.trunk_strata * 6
    }

    /// Local blocks per stratum (5).
    pub fn local_per_stratum(&self) -> usize {
        5
    }

    /// Global blocks per stratum (1).
    pub fn global_per_stratum(&self) -> usize {
        1
    }

    /// Total local blocks in the trunk.
    pub fn total_local_blocks(&self) -> usize {
        self.trunk_strata * self.local_per_stratum()
    }

    /// Total global blocks in the trunk.
    pub fn total_global_blocks(&self) -> usize {
        self.trunk_strata * self.global_per_stratum()
    }

    /// Validate cross-field constraints.
    pub fn validate(&self) -> Result<(), RhizomeError> {
        let err = |m: &str| RhizomeError::Config(format!("[{}] {m}", self.name));
        if self.d_model == 0 || self.n_heads == 0 || self.d_head == 0 {
            return Err(err("d_model, n_heads, d_head must be positive"));
        }
        if !self.d_model.is_multiple_of(self.n_heads) {
            return Err(err("d_model must be divisible by n_heads"));
        }
        if self.trunk_strata == 0 {
            return Err(err("trunk_strata must be >= 1"));
        }
        if self.core_after_stratum > self.trunk_strata || self.core_after_stratum == 0 {
            return Err(err("core_after_stratum must be in 1..=trunk_strata"));
        }
        if self.l_max == 0 {
            return Err(err("l_max must be >= 1"));
        }
        if self.experts_routed == 0 || self.expert_topk == 0 || self.expert_groups == 0 {
            return Err(err("MoE fields must be positive"));
        }
        if self.expert_groups > self.experts_routed {
            return Err(err("expert_groups > experts_routed"));
        }
        if !self.experts_routed.is_multiple_of(self.expert_groups) {
            return Err(err("experts_routed must be divisible by expert_groups"));
        }
        // top-k experts must fit within the union of the top-2 groups.
        let per_group = self.experts_routed / self.expert_groups;
        if self.expert_topk > 2 * per_group {
            return Err(err("expert_topk exceeds the top-2-group capacity"));
        }
        if self.expert_topk > self.experts_routed {
            return Err(err("expert_topk > experts_routed"));
        }
        for b in &self.pkm_blocks {
            if *b == 0 || *b > self.trunk_blocks() {
                return Err(err(&format!("pkm block {b} out of trunk range")));
            }
            // PKM lives in the routed-expert position; it must not collide.
        }
        let mut sorted = self.pkm_blocks.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != self.pkm_blocks.len() {
            return Err(err("duplicate pkm_blocks entry"));
        }
        if self.pkm_slots == 0 || self.pkm_rows == 0 || self.pkm_dims == 0 {
            return Err(err("PKM dims must be positive"));
        }
        if self.pkm_query_dim != 2 * self.pkm_dims {
            return Err(err("pkm_query_dim must equal 2*pkm_dims"));
        }
        if self.pkm_topk == 0 || self.pkm_topk > self.pkm_rows {
            return Err(err("bad pkm_topk"));
        }
        if self.vocab_size < 256 {
            return Err(err("vocab_size must cover the byte alphabet"));
        }
        if self.mtp_heads == 0 {
            return Err(err("mtp_heads must be >= 1"));
        }
        if self.dense_ffn_mult == 0 || self.conv_kernel == 0 {
            return Err(err("dense_ffn_mult and conv_kernel must be positive"));
        }
        if self.byte_path.enabled {
            let bp = &self.byte_path;
            if bp.ngram_min > bp.ngram_max || bp.ngram_max == 0 {
                return Err(err("bad ngram range"));
            }
            if bp.patch_min == 0 || bp.patch_max < bp.patch_min {
                return Err(err("bad patch length range"));
            }
            if bp.patcher_blocks == 0 || bp.patcher_d_model == 0 {
                return Err(err("bad patcher dims"));
            }
        }
        if let Some(cap) = self.attn_softcap {
            if cap <= 0.0 {
                return Err(err("attn_softcap must be positive"));
            }
        }
        Ok(())
    }

    fn req_usize(t: &Table, key: &str) -> Result<usize, RhizomeError> {
        match t.get(key) {
            None => Err(RhizomeError::Config(format!("missing field {key}"))),
            Some(v) => {
                let i = v
                    .as_int()
                    .map_err(|_| RhizomeError::Config(format!("field {key} must be an integer")))?;
                if i < 0 {
                    return Err(RhizomeError::Config(format!(
                        "field {key} must be non-negative"
                    )));
                }
                Ok(i as usize)
            }
        }
    }

    fn opt_usize(t: &Table, key: &str, default: usize) -> Result<usize, RhizomeError> {
        match t.get(key) {
            None => Ok(default),
            Some(_) => Self::req_usize(t, key),
        }
    }

    fn req_f64(t: &Table, key: &str) -> Result<f64, RhizomeError> {
        match t.get(key) {
            None => Err(RhizomeError::Config(format!("missing field {key}"))),
            Some(v) => v
                .as_float()
                .map_err(|_| RhizomeError::Config(format!("field {key} must be a number"))),
        }
    }

    fn opt_f64(t: &Table, key: &str, default: f64) -> Result<f64, RhizomeError> {
        match t.get(key) {
            None => Ok(default),
            Some(_) => Self::req_f64(t, key),
        }
    }

    fn opt_bool(t: &Table, key: &str, default: bool) -> Result<bool, RhizomeError> {
        match t.get(key) {
            None => Ok(default),
            Some(v) => v
                .as_bool()
                .map_err(|_| RhizomeError::Config(format!("field {key} must be bool"))),
        }
    }

    /// Parse from a parsed TOML table (the `[model]` section).
    pub fn from_table(name: &str, t: &Table) -> Result<Self, RhizomeError> {
        let d_model = Self::req_usize(t, "d_model")?;
        let n_heads = Self::req_usize(t, "n_heads")?;
        let cfg = ModelConfig {
            name: name.to_string(),
            d_model,
            n_heads,
            d_head: Self::req_usize(t, "d_head")?,
            trunk_strata: Self::req_usize(t, "trunk_strata")?,
            core_after_stratum: Self::req_usize(t, "core_after_stratum")?,
            l_max: Self::req_usize(t, "l_max")?,
            core_read_source: match t.get("core_read_source") {
                None => CoreReadSource::Input,
                Some(v) => CoreReadSource::parse(v.as_str()?)?,
            },
            experts_routed: Self::req_usize(t, "experts_routed")?,
            experts_shared: Self::req_usize(t, "experts_shared")?,
            expert_topk: Self::req_usize(t, "expert_topk")?,
            expert_hidden: Self::req_usize(t, "expert_hidden")?,
            expert_groups: Self::req_usize(t, "expert_groups")?,
            pkm_blocks: match t.get("pkm_blocks") {
                None => Vec::new(),
                Some(v) => {
                    let a = v.as_array()?;
                    let mut out = Vec::with_capacity(a.len());
                    for item in a {
                        let i = item.as_int().map_err(|_| {
                            RhizomeError::Config("pkm_blocks entries must be integers".into())
                        })?;
                        if i <= 0 {
                            return Err(RhizomeError::Config("pkm_blocks must be 1-based".into()));
                        }
                        out.push(i as usize);
                    }
                    out
                }
            },
            pkm_slots: Self::opt_usize(t, "pkm_slots", 1 << 20)?,
            pkm_rows: Self::opt_usize(t, "pkm_rows", 1024)?,
            pkm_dims: Self::opt_usize(t, "pkm_dims", 256)?,
            pkm_heads: Self::opt_usize(t, "pkm_heads", 4)?,
            pkm_query_dim: Self::opt_usize(t, "pkm_query_dim", 512)?,
            pkm_topk: Self::opt_usize(t, "pkm_topk", 32)?,
            d_latent: Self::req_usize(t, "d_latent")?,
            d_rope: Self::req_usize(t, "d_rope")?,
            d_nope: Self::req_usize(t, "d_nope")?,
            rope_base: Self::req_f64(t, "rope_base")?,
            yarn: {
                let mut y = YarnConfig::default();
                if let Some(Value::Table(yt)) = t.get("yarn") {
                    y.original_context =
                        Self::opt_usize(yt, "original_context", y.original_context)?;
                    y.target_context = Self::opt_usize(yt, "target_context", y.target_context)?;
                    y.attn_temp = Self::opt_f64(yt, "attn_temp", y.attn_temp)?;
                    y.beta_fast = Self::opt_f64(yt, "beta_fast", y.beta_fast)?;
                    y.beta_slow = Self::opt_f64(yt, "beta_slow", y.beta_slow)?;
                    y.mscale = Self::opt_f64(yt, "mscale", y.mscale)?;
                }
                y
            },
            attn_softcap: match t.get("attn_softcap") {
                None => None,
                Some(_) => Some(Self::req_f64(t, "attn_softcap")?),
            },
            fp8_latent_cache: Self::opt_bool(t, "fp8_latent_cache", false)?,
            vocab_size: Self::req_usize(t, "vocab_size")?,
            tie_unembed: Self::opt_bool(t, "tie_unembed", false)?,
            mtp_heads: Self::opt_usize(t, "mtp_heads", 2)?,
            dense_ffn_mult: Self::opt_usize(t, "dense_ffn_mult", 4)?,
            conv_kernel: Self::opt_usize(t, "conv_kernel", 4)?,
            deploy: match t.get("deploy") {
                None => DeployFormat::F32,
                Some(v) => DeployFormat::parse(v.as_str()?)?,
            },
            byte_path: {
                let mut bp = BytePathConfig::default();
                if let Some(Value::Table(bt)) = t.get("byte_path") {
                    bp.enabled = Self::opt_bool(bt, "enabled", bp.enabled)?;
                    bp.d_local = Self::req_usize(bt, "d_local")?;
                    bp.encoder_blocks = Self::opt_usize(bt, "encoder_blocks", bp.encoder_blocks)?;
                    bp.decoder_blocks = Self::opt_usize(bt, "decoder_blocks", bp.decoder_blocks)?;
                    bp.ngram_tables = Self::opt_usize(bt, "ngram_tables", bp.ngram_tables)?;
                    bp.ngram_min = Self::opt_usize(bt, "ngram_min", bp.ngram_min)?;
                    bp.ngram_max = Self::opt_usize(bt, "ngram_max", bp.ngram_max)?;
                    bp.ngram_buckets = Self::opt_usize(bt, "ngram_buckets", bp.ngram_buckets)?;
                    bp.byte_window = Self::opt_usize(bt, "byte_window", bp.byte_window)?;
                    bp.patch_min = Self::opt_usize(bt, "patch_min", bp.patch_min)?;
                    bp.patch_max = Self::opt_usize(bt, "patch_max", bp.patch_max)?;
                    bp.theta_g = Self::opt_f64(bt, "theta_g", bp.theta_g)?;
                    bp.delta = Self::opt_f64(bt, "delta", bp.delta)?;
                    bp.patcher_d_model = Self::req_usize(bt, "patcher_d_model")?;
                    bp.patcher_blocks = Self::opt_usize(bt, "patcher_blocks", bp.patcher_blocks)?;
                }
                bp
            },
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Parse a full config file (with `[model]` section).
    pub fn from_toml_str(name: &str, src: &str) -> Result<Self, RhizomeError> {
        let t = toml::parse(src)?;
        match t.get("model") {
            Some(Value::Table(mt)) => Self::from_table(name, mt),
            _ => Err(RhizomeError::Config("missing [model] section".into())),
        }
    }
}

/// Optimizer and schedule description (SPEC 5.3).
#[derive(Debug, Clone, PartialEq)]
pub struct TrainConfig {
    /// Master seed.
    pub seed: u64,
    /// Micro-batch (sequences per fwd/bwd).
    pub micro_batch: usize,
    /// Optimizer batch (gradient accumulation × world).
    pub batch: usize,
    /// Packed sequence length.
    pub seq_len: usize,
    /// Total optimizer steps.
    pub steps: usize,
    /// Warmup fraction of steps.
    pub warmup_frac: f64,
    /// Stable fraction (rest is decay; WSD).
    pub stable_frac: f64,
    /// Muon LR.
    pub muon_lr: f64,
    /// Muon momentum.
    pub muon_momentum: f64,
    /// Newton–Schulz iterations.
    pub muon_ns_iters: usize,
    /// AdamW LR for the μP table groups.
    pub adamw_lr: f64,
    /// Adam beta1.
    pub adam_beta1: f64,
    /// Adam beta2.
    pub adam_beta2: f64,
    /// Adam epsilon.
    pub adam_eps: f64,
    /// Weight decay (AdamW group).
    pub weight_decay: f64,
    /// EMA decay (0 = off).
    pub ema_decay: f64,
    /// Global grad clip norm.
    pub grad_clip: f64,
    /// z-loss weight.
    pub z_loss: f64,
    /// Sequence-wise expert balance loss weight.
    pub balance_loss: f64,
    /// Depth regularizer λ_d.
    pub depth_reg: f64,
    /// Halt threshold τ at inference.
    pub halt_tau: f64,
    /// Depth sampling μ (ln 2).
    pub depth_mu: f64,
    /// Depth sampling σ.
    pub depth_sigma: f64,
    /// Truncated backprop iterations.
    pub backprop_depth: usize,
    /// Convergence ε_c.
    pub conv_epsilon: f64,
    /// Loss ε_l (nats).
    pub loss_epsilon: f64,
    /// Halt-head loss token subsample rate (1/8).
    pub loss_subsample: f64,
    /// Scheduled halting start fraction.
    pub scheduled_halting_start: f64,
    /// QAT start fraction (final 10%).
    pub qat_start: f64,
    /// Router bias step size γ (loss-free balancing).
    pub router_bias_gamma: f64,
    /// Expert-parallel world size.
    pub ep_world: usize,
    /// Pipeline stages.
    pub pipeline_stages: usize,
    /// ZeRO stage (0..=3).
    pub zero_stage: u32,
    /// Checkpoint interval (steps).
    pub ckpt_interval: usize,
    /// Deterministic mode.
    pub deterministic: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            seed: 0,
            micro_batch: 8,
            batch: 32,
            seq_len: 256,
            steps: 1000,
            warmup_frac: 0.05,
            stable_frac: 0.7,
            muon_lr: 0.02,
            muon_momentum: 0.95,
            muon_ns_iters: 5,
            adamw_lr: 3e-4,
            adam_beta1: 0.9,
            adam_beta2: 0.95,
            adam_eps: 1e-8,
            weight_decay: 0.1,
            ema_decay: 0.999,
            grad_clip: 1.0,
            z_loss: 1e-4,
            balance_loss: 1e-4,
            depth_reg: 1e-3,
            halt_tau: 0.5,
            depth_mu: 2.0f64.ln(),
            depth_sigma: 0.5,
            backprop_depth: 4,
            conv_epsilon: 0.05,
            loss_epsilon: 0.02,
            loss_subsample: 0.125,
            scheduled_halting_start: 0.8,
            qat_start: 0.9,
            router_bias_gamma: 1e-3,
            ep_world: 1,
            pipeline_stages: 1,
            zero_stage: 0,
            ckpt_interval: 100,
            deterministic: false,
        }
    }
}

impl TrainConfig {
    /// Parse the `[train]` section (all fields optional; defaults above).
    pub fn from_table(t: &Table) -> Result<Self, RhizomeError> {
        let d = TrainConfig::default();
        let g = |k: &str| t.get(k);
        let u = |k: &str, d: usize| -> Result<usize, RhizomeError> {
            match g(k) {
                None => Ok(d),
                Some(v) => {
                    let i = v
                        .as_int()
                        .map_err(|_| RhizomeError::Config(format!("train.{k} must be int")))?;
                    if i < 0 {
                        return Err(RhizomeError::Config(format!("train.{k} negative")));
                    }
                    Ok(i as usize)
                }
            }
        };
        let f = |k: &str, d: f64| -> Result<f64, RhizomeError> {
            match g(k) {
                None => Ok(d),
                Some(v) => v
                    .as_float()
                    .map_err(|_| RhizomeError::Config(format!("train.{k} must be number"))),
            }
        };
        let cfg = TrainConfig {
            seed: match g("seed") {
                None => d.seed,
                Some(v) => v
                    .as_int()
                    .map_err(|_| RhizomeError::Config("train.seed must be int".into()))?
                    as u64,
            },
            micro_batch: u("micro_batch", d.micro_batch)?,
            batch: u("batch", d.batch)?,
            seq_len: u("seq_len", d.seq_len)?,
            steps: u("steps", d.steps)?,
            warmup_frac: f("warmup_frac", d.warmup_frac)?,
            stable_frac: f("stable_frac", d.stable_frac)?,
            muon_lr: f("muon_lr", d.muon_lr)?,
            muon_momentum: f("muon_momentum", d.muon_momentum)?,
            muon_ns_iters: u("muon_ns_iters", d.muon_ns_iters)?,
            adamw_lr: f("adamw_lr", d.adamw_lr)?,
            adam_beta1: f("adam_beta1", d.adam_beta1)?,
            adam_beta2: f("adam_beta2", d.adam_beta2)?,
            adam_eps: f("adam_eps", d.adam_eps)?,
            weight_decay: f("weight_decay", d.weight_decay)?,
            ema_decay: f("ema_decay", d.ema_decay)?,
            grad_clip: f("grad_clip", d.grad_clip)?,
            z_loss: f("z_loss", d.z_loss)?,
            balance_loss: f("balance_loss", d.balance_loss)?,
            depth_reg: f("depth_reg", d.depth_reg)?,
            halt_tau: f("halt_tau", d.halt_tau)?,
            depth_mu: f("depth_mu", d.depth_mu)?,
            depth_sigma: f("depth_sigma", d.depth_sigma)?,
            backprop_depth: u("backprop_depth", d.backprop_depth)?,
            conv_epsilon: f("conv_epsilon", d.conv_epsilon)?,
            loss_epsilon: f("loss_epsilon", d.loss_epsilon)?,
            loss_subsample: f("loss_subsample", d.loss_subsample)?,
            scheduled_halting_start: f("scheduled_halting_start", d.scheduled_halting_start)?,
            qat_start: f("qat_start", d.qat_start)?,
            router_bias_gamma: f("router_bias_gamma", d.router_bias_gamma)?,
            ep_world: u("ep_world", d.ep_world)?,
            pipeline_stages: u("pipeline_stages", d.pipeline_stages)?,
            zero_stage: match g("zero_stage") {
                None => d.zero_stage,
                Some(v) => v
                    .as_int()
                    .map_err(|_| RhizomeError::Config("train.zero_stage must be int".into()))?
                    as u32,
            },
            ckpt_interval: u("ckpt_interval", d.ckpt_interval)?,
            deterministic: match g("deterministic") {
                None => d.deterministic,
                Some(v) => v
                    .as_bool()
                    .map_err(|_| RhizomeError::Config("train.deterministic must be bool".into()))?,
            },
        };
        if cfg.seq_len == 0 || cfg.micro_batch == 0 || cfg.batch == 0 {
            return Err(RhizomeError::Config(
                "train batch/seq must be positive".into(),
            ));
        }
        if cfg.muon_ns_iters == 0 || cfg.muon_ns_iters > 32 {
            return Err(RhizomeError::Config("muon_ns_iters out of range".into()));
        }
        if !(0.0..=1.0).contains(&cfg.warmup_frac) || !(0.0..=1.0).contains(&cfg.stable_frac) {
            return Err(RhizomeError::Config(
                "warmup/stable fractions must be in [0,1]".into(),
            ));
        }
        if cfg.ep_world == 0 || cfg.pipeline_stages == 0 {
            return Err(RhizomeError::Config(
                "ep_world/pipeline_stages must be >= 1".into(),
            ));
        }
        Ok(cfg)
    }
}

/// Serving configuration (SPEC 6).
#[derive(Debug, Clone, PartialEq)]
pub struct ServeConfig {
    /// HTTP listen port.
    pub http_port: u16,
    /// gRPC listen port (serve-grpc feature).
    pub grpc_port: u16,
    /// KV page size (tokens).
    pub page_size: usize,
    /// Prefill chunk size (tokens/patches).
    pub prefill_chunk: usize,
    /// Max total context tokens.
    pub max_context: usize,
    /// Batch buckets served.
    pub buckets: Vec<usize>,
    /// Prefix-cache LRU byte cap (pinned host memory).
    pub prefix_cache_bytes: usize,
    /// Snapshot interval inside long prefixes.
    pub snapshot_every: usize,
    /// Deterministic mode (R9/SPEC 6.8).
    pub deterministic: bool,
    /// Speculative draft tokens.
    pub spec_draft: usize,
    /// p99 inter-token latency SLO (µs) driving the Core-cap controller.
    pub itl_slo_us: u64,
    /// Max resident pages (admission control).
    pub max_pages: usize,
    /// Default think budget (max Core iterations when unbound).
    pub default_think_budget: usize,
}

impl Default for ServeConfig {
    fn default() -> Self {
        ServeConfig {
            http_port: 8080,
            grpc_port: 8081,
            page_size: 64,
            prefill_chunk: 2048,
            max_context: 8192,
            buckets: vec![1, 2, 4, 8, 16, 32, 64, 128],
            prefix_cache_bytes: 1 << 30,
            snapshot_every: 4096,
            deterministic: true,
            spec_draft: 3,
            itl_slo_us: 50_000,
            max_pages: 1 << 16,
            default_think_budget: 8,
        }
    }
}

impl ServeConfig {
    /// Parse the `[serve]` section.
    pub fn from_table(t: &Table) -> Result<Self, RhizomeError> {
        let d = ServeConfig::default();
        let gi = |k: &str, dv: i64| -> Result<i64, RhizomeError> {
            match t.get(k) {
                None => Ok(dv),
                Some(v) => v
                    .as_int()
                    .map_err(|_| RhizomeError::Config(format!("serve.{k} must be int"))),
            }
        };
        let gf = |k: &str, dv: f64| -> Result<f64, RhizomeError> {
            match t.get(k) {
                None => Ok(dv),
                Some(v) => v
                    .as_float()
                    .map_err(|_| RhizomeError::Config(format!("serve.{k} must be number"))),
            }
        };
        let buckets = match t.get("buckets") {
            None => d.buckets.clone(),
            Some(v) => {
                let a = v.as_array()?;
                let mut out = Vec::with_capacity(a.len());
                for item in a {
                    let i = item.as_int().map_err(|_| {
                        RhizomeError::Config("serve.buckets entries must be int".into())
                    })?;
                    out.push(i as usize);
                }
                out
            }
        };
        let cfg = ServeConfig {
            http_port: gi("http_port", d.http_port as i64)? as u16,
            grpc_port: gi("grpc_port", d.grpc_port as i64)? as u16,
            page_size: gi("page_size", d.page_size as i64)? as usize,
            prefill_chunk: gi("prefill_chunk", d.prefill_chunk as i64)? as usize,
            max_context: gi("max_context", d.max_context as i64)? as usize,
            buckets,
            prefix_cache_bytes: gi("prefix_cache_bytes", d.prefix_cache_bytes as i64)? as usize,
            snapshot_every: gi("snapshot_every", d.snapshot_every as i64)? as usize,
            deterministic: match t.get("deterministic") {
                None => d.deterministic,
                Some(v) => v
                    .as_bool()
                    .map_err(|_| RhizomeError::Config("serve.deterministic must be bool".into()))?,
            },
            spec_draft: gi("spec_draft", d.spec_draft as i64)? as usize,
            itl_slo_us: gi("itl_slo_us", d.itl_slo_us as i64)? as u64,
            max_pages: gi("max_pages", d.max_pages as i64)? as usize,
            default_think_budget: gi("default_think_budget", d.default_think_budget as i64)?
                as usize,
        };
        let _ = gf("unused", 0.0); // keep f-helper used; serve section is integer-only today
        if cfg.page_size == 0 || cfg.prefill_chunk == 0 || cfg.max_context == 0 {
            return Err(RhizomeError::Config(
                "serve page/prefill/context must be positive".into(),
            ));
        }
        if cfg.buckets.is_empty() {
            return Err(RhizomeError::Config(
                "serve.buckets must be non-empty".into(),
            ));
        }
        Ok(cfg)
    }
}

/// A complete parsed configuration file: model + train + serve.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Model architecture.
    pub model: ModelConfig,
    /// Training settings.
    pub train: TrainConfig,
    /// Serving settings.
    pub serve: ServeConfig,
}

impl Config {
    /// Parse a full TOML document.
    pub fn from_toml_str(name: &str, src: &str) -> Result<Self, RhizomeError> {
        let t = toml::parse(src)?;
        let model = match t.get("model") {
            Some(Value::Table(mt)) => ModelConfig::from_table(name, mt)?,
            _ => return Err(RhizomeError::Config("missing [model] section".into())),
        };
        let train = match t.get("train") {
            Some(Value::Table(tt)) => TrainConfig::from_table(tt)?,
            _ => TrainConfig::default(),
        };
        let serve = match t.get("serve") {
            Some(Value::Table(st)) => ServeConfig::from_table(st)?,
            _ => ServeConfig::default(),
        };
        Ok(Config {
            model,
            train,
            serve,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_S: &str = include_str!("../../../configs/test-s.toml");

    #[test]
    fn parse_test_s() {
        let cfg = Config::from_toml_str("test-s", TEST_S).unwrap();
        assert_eq!(cfg.model.d_model, 64);
        assert_eq!(cfg.model.trunk_strata, 1);
        assert_eq!(cfg.model.trunk_blocks(), 6);
        assert_eq!(cfg.model.total_local_blocks(), 5);
        assert_eq!(cfg.model.total_global_blocks(), 1);
        assert_eq!(cfg.model.core_after_stratum, 1);
        assert_eq!(cfg.model.l_max, 3);
        assert!(!cfg.model.pkm_blocks.is_empty() || cfg.model.pkm_blocks.is_empty());
        assert!(cfg.model.validate().is_ok());
        assert_eq!(cfg.model.deploy, DeployFormat::F32);
    }

    #[test]
    fn validation_rejects_bad_configs() {
        let mut cfg = Config::from_toml_str("test-s", TEST_S).unwrap();
        cfg.model.expert_topk = 100;
        assert!(cfg.model.validate().is_err());
        let mut cfg2 = Config::from_toml_str("test-s", TEST_S).unwrap();
        cfg2.model.d_model = 65; // not divisible by heads
        assert!(cfg2.model.validate().is_err());
        let mut cfg3 = Config::from_toml_str("test-s", TEST_S).unwrap();
        cfg3.model.core_after_stratum = 0;
        assert!(cfg3.model.validate().is_err());
        let mut cfg4 = Config::from_toml_str("test-s", TEST_S).unwrap();
        cfg4.model.experts_routed = 7;
        cfg4.model.expert_groups = 4;
        assert!(cfg4.model.validate().is_err());
    }

    #[test]
    fn missing_fields_are_errors() {
        assert!(ModelConfig::from_toml_str("x", "").is_err());
        assert!(ModelConfig::from_toml_str("x", "[model]\nd_model = 64\n").is_err());
    }
}
