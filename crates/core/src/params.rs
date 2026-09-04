//! Exact parameter accounting, FLOPs/token and deploy-memory estimates for a
//! [`ModelConfig`]. `rhizome params --all --format=markdown` emits this table;
//! CI asserts it equals the table in SPEC.md (R7).

use crate::config::{DeployFormat, ModelConfig};

/// Per-component parameter breakdown (element counts).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ParamBreakdown {
    /// Token embedding (V×d).
    pub embedding_in: u64,
    /// Unembedding (V×d when untied).
    pub embedding_out: u64,
    /// Trunk Local mixers (qkv+gate proj, conv, out proj, norms, logits).
    pub trunk_local_mixers: u64,
    /// Trunk MLA attentions.
    pub trunk_global_attn: u64,
    /// Trunk MoE experts (shared + routed) and routers.
    pub trunk_moe: u64,
    /// Trunk PKM blocks (keys, values, query, out proj, shared experts).
    pub trunk_pkm: u64,
    /// Trunk norms (pre + sandwich).
    pub trunk_norms: u64,
    /// Think Core (adapter, embeddings, C1/C2/C3, halt head).
    pub core: u64,
    /// MTP heads (blocks + untied unembeddings).
    pub mtp: u64,
    /// Byte path (patcher, encoder, pooling, decoder).
    pub byte_path: u64,
}

impl ParamBreakdown {
    /// Sum of all components.
    pub fn total(&self) -> u64 {
        self.embedding_in
            + self.embedding_out
            + self.trunk_local_mixers
            + self.trunk_global_attn
            + self.trunk_moe
            + self.trunk_pkm
            + self.trunk_norms
            + self.core
            + self.mtp
            + self.byte_path
    }
}

fn local_mixer_params(m: &ModelConfig, in_dim: usize) -> u64 {
    let hd = m.n_heads * m.d_head;
    let proj = (in_dim * 4 * hd) as u64;
    let conv = (3 * hd * m.conv_kernel) as u64;
    let out = (hd * m.d_model) as u64;
    let logits = (2 * m.n_heads + m.n_heads) as u64;
    let head_norm = hd as u64;
    proj + conv + out + logits + head_norm
}

fn mla_params(m: &ModelConfig) -> u64 {
    let h = m.n_heads;
    let wq = (m.d_model * h * (m.d_nope + m.d_rope)) as u64;
    let q_norm = (h * (m.d_nope + m.d_rope) + h * m.d_nope) as u64;
    let w_dk = (m.d_model * m.d_latent) as u64;
    let a_norm = m.d_latent as u64;
    let w_kr = (m.d_model * m.d_rope) as u64;
    let w_uk = (m.d_latent * h * m.d_nope) as u64;
    let w_uv = (m.d_latent * h * m.d_nope) as u64;
    let w_o = (h * m.d_nope * m.d_model) as u64;
    wq + q_norm + w_dk + a_norm + w_kr + w_uk + w_uv + w_o
}

fn moe_params(m: &ModelConfig) -> u64 {
    let per_expert = 3 * m.d_model as u64 * m.expert_hidden as u64;
    let experts = (m.experts_routed + m.experts_shared) as u64 * per_expert;
    let router = m.d_model as u64 * m.experts_routed as u64 + m.experts_routed as u64;
    experts + router
}

fn shared_experts_params(m: &ModelConfig) -> u64 {
    m.experts_shared as u64 * 3 * m.d_model as u64 * m.expert_hidden as u64
}

fn active_experts_params(m: &ModelConfig) -> u64 {
    (m.expert_topk + m.experts_shared) as u64 * 3 * m.d_model as u64 * m.expert_hidden as u64
}

fn pkm_params(m: &ModelConfig) -> u64 {
    let keys = 2 * m.pkm_rows as u64 * m.pkm_dims as u64;
    let values = m.pkm_slots as u64 * m.d_model as u64;
    let query = m.d_model as u64 * m.pkm_heads as u64 * m.pkm_query_dim as u64;
    let q_norm = (m.pkm_heads * m.pkm_query_dim) as u64;
    let out = (m.d_model * m.d_model) as u64;
    keys + values + query + q_norm + out
}

fn dense_ffn_params(m: &ModelConfig, d: usize) -> u64 {
    (3 * d * d * m.dense_ffn_mult) as u64
}

fn core_params(m: &ModelConfig) -> u64 {
    let adapter = (m.d_model * 2 * m.d_model) as u64;
    let e_iter = (m.l_max * m.d_model) as u64;
    let e_budget = (m.l_max * m.d_model) as u64;
    // C1 = MLA block with MoE FFN; C2 = Local block with MoE FFN; C3 dense.
    let c1 = mla_params(m) + moe_params(m);
    let c2 = local_mixer_params(m, m.d_model) + moe_params(m);
    let c3 = dense_ffn_params(m, m.d_model);
    let halt = (m.d_model + m.d_model) as u64;
    adapter + e_iter + e_budget + c1 + c2 + c3 + halt
}

fn mtp_params(m: &ModelConfig) -> u64 {
    // MTP heads share the trunk's output unembedding table (docs/mup.md).
    let mut total = 0;
    for _ in 0..m.mtp_heads {
        let fusion = (2 * m.d_model * m.d_model) as u64;
        let block = local_mixer_params(m, 2 * m.d_model) + dense_ffn_params(m, m.d_model);
        total += fusion + block;
    }
    total
}

fn byte_local_block(m: &ModelConfig, d: usize) -> u64 {
    // Local block (delta mixer at width d, dense FFN).
    let heads = (d / 32).max(1);
    let hd = d;
    let proj = (d * 4 * hd) as u64;
    let conv = (3 * hd * m.conv_kernel) as u64;
    let out = (hd * d) as u64;
    let logits = (3 * heads) as u64;
    let norms = (3 * d + hd) as u64;
    proj + conv + out + logits + norms + dense_ffn_params(m, d)
}

fn byte_path_params(m: &ModelConfig) -> u64 {
    if !m.byte_path.enabled {
        return 0;
    }
    let bp = &m.byte_path;
    let d = bp.d_local;
    // Patcher: byte embedding + blocks + byte unembed head.
    let pd = bp.patcher_d_model;
    let patcher = (256 * pd + 256 * pd) as u64 + bp.patcher_blocks as u64 * byte_local_block(m, pd);
    // Byte encoder: byte embedding + ngram tables + blocks.
    let encoder = (256 * d) as u64
        + (bp.ngram_tables * bp.ngram_buckets * d) as u64
        + bp.encoder_blocks as u64 * byte_local_block(m, d);
    // Patch pooling: query, K/V projections, out to d_model.
    let pooling = (d + d * d + d * d + d * m.d_model) as u64;
    // Decoder: blocks + per-block cross-attn + byte head.
    let cross = (2 * (m.d_model + d) * d) as u64;
    let decoder = bp.decoder_blocks as u64 * (byte_local_block(m, d) + cross) + (d * 256) as u64;
    patcher + encoder + pooling + decoder
}

/// Compute the full breakdown for a model config.
pub fn breakdown(m: &ModelConfig) -> ParamBreakdown {
    let mut b = ParamBreakdown {
        embedding_in: (m.vocab_size * m.d_model) as u64,
        embedding_out: if m.tie_unembed {
            0
        } else {
            (m.vocab_size * m.d_model) as u64
        },
        ..ParamBreakdown::default()
    };
    let pkm_set: std::collections::HashSet<usize> = m.pkm_blocks.iter().copied().collect();
    for block in 1..=m.trunk_blocks() {
        let is_global = block % 6 == 0;
        let has_pkm = pkm_set.contains(&block);
        b.trunk_norms += (3 * m.d_model) as u64;
        if is_global {
            b.trunk_global_attn += mla_params(m);
        } else {
            b.trunk_local_mixers += local_mixer_params(m, m.d_model);
        }
        if has_pkm {
            b.trunk_pkm += pkm_params(m) + shared_experts_params(m);
        } else {
            b.trunk_moe += moe_params(m);
        }
    }
    b.core = core_params(m);
    b.mtp = mtp_params(m);
    b.byte_path = byte_path_params(m);
    b
}

/// Active (compute) parameter counts: weight elements multiplied per token,
/// excluding embedding/unembedding table lookups. At L=1.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ActiveBreakdown {
    /// Trunk mixers + attention.
    pub trunk: u64,
    /// Active experts at L=1 (top-k + shared).
    pub trunk_experts: u64,
    /// PKM gather cost (query + keys + values read per token; values counted
    /// as the top-32 rows read, not the whole table).
    pub trunk_pkm: u64,
    /// Think Core blocks (C1+C2+C3) per iteration.
    pub core_per_iter: u64,
    /// Core adapter W_in (applied each iteration; reported separately because
    /// it is an injection rather than a block).
    pub core_adapter: u64,
    /// Total at L=1.
    pub total_l1: u64,
}

/// Active-parameter estimate for L=1 (see docs/mup.md for the accounting
/// convention: table lookups are excluded).
pub fn active(m: &ModelConfig) -> ActiveBreakdown {
    let mut a = ActiveBreakdown::default();
    let pkm_set: std::collections::HashSet<usize> = m.pkm_blocks.iter().copied().collect();
    for block in 1..=m.trunk_blocks() {
        let is_global = block % 6 == 0;
        let has_pkm = pkm_set.contains(&block);
        if is_global {
            a.trunk += mla_params(m);
        } else {
            a.trunk += local_mixer_params(m, m.d_model);
        }
        a.trunk += (3 * m.d_model) as u64;
        if has_pkm {
            // Per token: query projections + key dot products + 32 value rows.
            a.trunk_pkm += m.d_model as u64 * m.pkm_heads as u64 * m.pkm_query_dim as u64
                + 2 * m.pkm_rows as u64 * m.pkm_dims as u64
                + m.pkm_topk as u64 * m.d_model as u64
                + m.d_model as u64 * m.d_model as u64
                + shared_experts_params(m);
        } else {
            a.trunk_experts +=
                active_experts_params(m) + m.d_model as u64 * m.experts_routed as u64;
        }
    }
    // Core per-iteration cost: C1 active, C2 active, C3. The adapter and
    // halt head are injection/lookup-scale and counted separately.
    a.core_per_iter = mla_params(m)
        + active_experts_params(m)
        + local_mixer_params(m, m.d_model)
        + active_experts_params(m)
        + dense_ffn_params(m, m.d_model);
    a.core_adapter = (2 * m.d_model * m.d_model + 2 * m.d_model) as u64;
    a.total_l1 = a.trunk + a.trunk_experts + a.trunk_pkm + a.core_per_iter + a.core_adapter;
    a
}

/// Approximate FLOPs per token at a given Core depth (multiply-accumulate
/// pairs ×2), including attention quadratics for context `ctx`.
pub fn flops_per_token(m: &ModelConfig, depth: usize, ctx: usize) -> u64 {
    let a = active(m);
    let mut f = 2 * (a.trunk + a.trunk_experts + a.trunk_pkm);
    f += 2 * a.core_per_iter * depth as u64;
    // Attention quadratic terms: global blocks score the full context with
    // q·k of dim (d_nope+d_rope) plus value merge of d_nope.
    let heads = m.n_heads as u64;
    let qk = heads * (m.d_nope + m.d_rope) as u64;
    let vo = heads * m.d_nope as u64;
    let globals = m.total_global_blocks() as u64 + 1; // + core C1
    f += 2 * globals * ctx as u64 * (qk + vo);
    f
}

/// Deploy memory in bytes for the given format class (SPEC 3.9): experts and
/// mixer projections in MX, embeddings/router/norms/halt/PKM keys bf16,
/// PKM values int8, attention projections fp8 on Flagship.
pub fn deploy_bytes(m: &ModelConfig, b: &ParamBreakdown) -> u64 {
    let mx_elems = |x: u64| x / 2 + x / 32; // 4-bit code + e8m0 scale
    let bf16_elems = |x: u64| x * 2;
    let mx_body = |b: &ParamBreakdown| -> u64 {
        // Experts + mixer projections (in_proj/out_proj share of mixers,
        // approximated as all mixer params except norms/logits) in MX.
        let proj_frac = |x: u64| x * 9 / 10; // norms/logits are ~<10%
        let mx = mx_elems(proj_frac(b.trunk_local_mixers))
            + mx_elems(b.trunk_moe)
            + mx_elems(proj_frac(b.core));
        let bf16 =
            bf16_elems(b.embedding_in + b.embedding_out + b.mtp + b.byte_path + b.trunk_norms)
                + bf16_elems(b.trunk_global_attn)
                + bf16_elems(b.trunk_local_mixers - proj_frac(b.trunk_local_mixers))
                + bf16_elems(b.core - proj_frac(b.core));
        // PKM: keys bf16, values int8; only when PKM blocks exist.
        let n_pkm = m.pkm_blocks.len() as u64;
        let pkm_keys = n_pkm * 2 * m.pkm_rows as u64 * m.pkm_dims as u64;
        let pkm_vals = n_pkm * m.pkm_slots as u64 * m.d_model as u64;
        let pkm_rest = b.trunk_pkm.saturating_sub(pkm_keys + pkm_vals);
        mx + bf16 + bf16_elems(pkm_keys) + pkm_vals + bf16_elems(pkm_rest)
    };
    match m.deploy {
        DeployFormat::F32 => b.total() * 4,
        DeployFormat::MxInt4 | DeployFormat::MxFp4 => mx_body(b),
        DeployFormat::MxFp4Fp8Attn => {
            // Same body with attention projections in fp8 instead of bf16.
            let base = mx_body(b);
            base - b.trunk_global_attn * 2 + b.trunk_global_attn // fp8 = 1 byte/elem
        }
    }
}

/// Render the SPEC.md parameter table for the given (name, config) pairs.
pub fn markdown_table(rows: &[(String, ModelConfig)]) -> String {
    let mut out = String::from("| Field |");
    for (name, _) in rows {
        out.push_str(&format!(" {name} |"));
    }
    out.push('\n');
    out.push_str("|---|");
    for _ in rows {
        out.push_str("---|");
    }
    out.push('\n');
    let mut line = |field: &str, vals: Vec<String>| {
        out.push_str(&format!("| {field} |"));
        for v in vals {
            out.push_str(&format!(" {v} |"));
        }
        out.push('\n');
    };
    let g = |f: fn(&ModelConfig) -> String| rows.iter().map(|(_, m)| f(m)).collect();
    line("d_model", g(|m| m.d_model.to_string()));
    line(
        "trunk_strata (blocks)",
        g(|m| format!("{} ({})", m.trunk_strata, m.trunk_blocks())),
    );
    line(
        "core_after_stratum",
        g(|m| m.core_after_stratum.to_string()),
    );
    line("l_max", g(|m| m.l_max.to_string()));
    line(
        "n_heads (d_head)",
        g(|m| format!("{} ({})", m.n_heads, m.d_head)),
    );
    line(
        "experts routed/shared/topk",
        g(|m| {
            format!(
                "{}/{}/{}",
                m.experts_routed, m.experts_shared, m.expert_topk
            )
        }),
    );
    line("expert_hidden", g(|m| m.expert_hidden.to_string()));
    line("expert_groups", g(|m| m.expert_groups.to_string()));
    line(
        "pkm_blocks",
        g(|m| {
            if m.pkm_blocks.is_empty() {
                "none".into()
            } else {
                format!("{:?}", m.pkm_blocks)
            }
        }),
    );
    line(
        "front_end",
        g(|m| {
            if m.byte_path.enabled {
                "both".into()
            } else {
                "BPE".into()
            }
        }),
    );
    line(
        "approx_total_params",
        g(|m| format!("~{:.2}B", breakdown(m).total() as f64 / 1e9)),
    );
    line(
        "approx_active_l1",
        g(|m| format!("~{:.2}B", active(m).total_l1 as f64 / 1e9)),
    );
    line(
        "per_core_iter",
        g(|m| format!("~{:.2}B", active(m).core_per_iter as f64 / 1e9)),
    );
    line("deploy", g(|m| m.deploy.name().to_string()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn load(name: &str) -> ModelConfig {
        let path = format!("{}/../../configs/{name}.toml", env!("CARGO_MANIFEST_DIR"));
        let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        Config::from_toml_str(name, &src).unwrap().model
    }

    #[test]
    fn tier_totals_within_10pct_of_spec_targets() {
        // SPEC 3.10 targets (approx, ±10% asserted per R7/spec header).
        // Flagship active is asserted at ±25%: see ADR-0006 (the table's
        // flagship active target is inconsistent with its own row under any
        // uniform accounting convention).
        let cases = [("edge", 2.5e9), ("standard", 30.0e9), ("flagship", 70.0e9)];
        for (name, target) in cases {
            let m = load(name);
            let b = breakdown(&m);
            let total = b.total() as f64;
            let rel = (total - target) / target;
            assert!(
                rel.abs() <= 0.10,
                "{name}: total {total:.3e} vs target {target:.3e} (rel {rel:+.3})"
            );
        }
    }

    #[test]
    fn active_l1_within_10pct_of_spec_targets() {
        // Edge and Standard match the topk+shared convention within ±10%;
        // Flagship is widened to ±25% per ADR-0006 (table-internal
        // inconsistency in SPEC 3.10: topk=6 across 39 MoE blocks at
        // h_e=1536 implies ≥11.6B active under any uniform convention).
        let cases = [
            ("edge", 0.45e9, 0.10),
            ("standard", 3.5e9, 0.10),
            ("flagship", 9.5e9, 0.25),
        ];
        for (name, target, tol) in cases {
            let m = load(name);
            let a = active(&m);
            let total = a.total_l1 as f64;
            let rel = (total - target) / target;
            assert!(
                rel.abs() <= tol,
                "{name}: active {total:.3e} vs target {target:.3e} (rel {rel:+.3}, tol {tol})"
            );
        }
    }

    #[test]
    fn per_core_iteration_within_10pct() {
        // Flagship widened to ±25% per ADR-0006.
        let cases = [
            ("edge", 0.05e9, 0.10),
            ("standard", 0.3e9, 0.10),
            ("flagship", 0.6e9, 0.25),
        ];
        for (name, target, tol) in cases {
            let m = load(name);
            let a = active(&m);
            let rel = (a.core_per_iter as f64 - target) / target;
            assert!(
                rel.abs() <= tol,
                "{name}: core/iter {} rel {rel:+.3}",
                a.core_per_iter
            );
        }
    }

    #[test]
    fn test_configs_are_small_for_ci() {
        let s = load("test-s");
        assert!(
            breakdown(&s).total() < 3_000_000,
            "test-s too big: {}",
            breakdown(&s).total()
        );
        let l = load("test-l");
        assert!(
            breakdown(&l).total() < 120_000_000,
            "test-l too big: {}",
            breakdown(&l).total()
        );
    }

    #[test]
    fn flops_grow_with_depth_and_context() {
        let m = load("test-s");
        let f1 = flops_per_token(&m, 1, 128);
        let f4 = flops_per_token(&m, 4, 128);
        let fc = flops_per_token(&m, 1, 1024);
        // 3 extra iterations add ~3x the per-iteration cost.
        let per_iter = 2 * active(&m).core_per_iter;
        assert!(f4 - f1 > 2 * per_iter && f4 - f1 < 4 * per_iter);
        assert!(fc > f1);
    }

    #[test]
    fn deploy_bytes_smaller_than_f32() {
        let m = load("standard");
        let b = breakdown(&m);
        assert!(deploy_bytes(&m, &b) < b.total() * 4);
        let e = load("edge");
        let be = breakdown(&e);
        assert!(deploy_bytes(&e, &be) < be.total() * 4);
    }

    #[test]
    fn markdown_table_renders() {
        let m = load("test-s");
        let md = markdown_table(&[("test-s".to_string(), m)]);
        assert!(md.contains("test-s"));
        assert!(md.contains("| d_model |"));
    }

    #[test]
    fn config_files_parse_and_validate() {
        for name in ["test-s", "test-m", "test-l", "edge", "standard", "flagship"] {
            let m = load(name);
            assert!(m.validate().is_ok(), "{name} invalid");
        }
    }
}
