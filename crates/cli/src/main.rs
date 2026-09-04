//! `rhizome` command-line interface.
//!
//! Subcommands arrive with their milestones (docs/milestones.md). M0 ships
//! `version` and `params` (R7/M0 acceptance).

#![deny(missing_docs)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use rhizome_core::config::Config;
use rhizome_core::json::{self, Json, ObjBuilder};
use rhizome_core::params as rp;
use std::path::{Path, PathBuf};

const USAGE: &str = "usage: rhizome <version|params|train|serve|convert|eval|edit-memory|bench|verify|inspect|data|tokenizer> [args]";

fn repo_root() -> PathBuf {
    let mut p = std::path::absolute(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
        .unwrap_or_else(|_| PathBuf::from("."));
    while !p.join("Cargo.lock").exists() && p.parent().is_some() {
        p = p.parent().map(Path::to_path_buf).unwrap_or_default();
    }
    p
}

fn configs_root() -> PathBuf {
    let root = repo_root();
    let direct = root.join("configs");
    if direct.is_dir() {
        return direct;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs")
}

fn load_config(name: &str) -> Result<Config, String> {
    let path = configs_root().join(format!("{name}.toml"));
    let src =
        std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Config::from_toml_str(name, &src).map_err(|e| format!("parse {}: {e}", path.display()))
}

fn config_json(name: &str, cfg: &Config) -> Json {
    let m = &cfg.model;
    let b = rp::breakdown(m);
    let a = rp::active(m);
    ObjBuilder::new()
        .str_field("name", name)
        .int_field("d_model", m.d_model as i64)
        .int_field("trunk_strata", m.trunk_strata as i64)
        .int_field("trunk_blocks", m.trunk_blocks() as i64)
        .int_field("l_max", m.l_max as i64)
        .field("total_params", Json::int(b.total() as i64))
        .field(
            "breakdown",
            Json::object(vec![
                ("embedding_in", Json::int(b.embedding_in as i64)),
                ("embedding_out", Json::int(b.embedding_out as i64)),
                ("trunk_local_mixers", Json::int(b.trunk_local_mixers as i64)),
                ("trunk_global_attn", Json::int(b.trunk_global_attn as i64)),
                ("trunk_moe", Json::int(b.trunk_moe as i64)),
                ("trunk_pkm", Json::int(b.trunk_pkm as i64)),
                ("trunk_norms", Json::int(b.trunk_norms as i64)),
                ("core", Json::int(b.core as i64)),
                ("mtp", Json::int(b.mtp as i64)),
                ("byte_path", Json::int(b.byte_path as i64)),
            ]),
        )
        .field(
            "active_l1",
            Json::object(vec![
                ("trunk", Json::int(a.trunk as i64)),
                ("trunk_experts", Json::int(a.trunk_experts as i64)),
                ("trunk_pkm", Json::int(a.trunk_pkm as i64)),
                ("core_per_iter", Json::int(a.core_per_iter as i64)),
                ("total", Json::int(a.total_l1 as i64)),
            ]),
        )
        .float_field(
            "flops_per_token_l1_4k",
            rp::flops_per_token(m, 1, 4096) as f64,
        )
        .float_field(
            "flops_per_token_lmax_4k",
            rp::flops_per_token(m, m.l_max, 4096) as f64,
        )
        .float_field("deploy_bytes", rp::deploy_bytes(m, &b) as f64)
        .str_field("deploy_format", m.deploy.name())
        .build()
}

fn cmd_params(args: &[String]) -> i32 {
    let all = args.iter().any(|a| a == "--all");
    let fmt = args
        .iter()
        .position(|a| a == "--format")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "table".to_string());
    let names: Vec<String> = if all {
        let mut n = vec!["test-s", "test-m", "test-l", "edge", "standard", "flagship"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        n.sort_unstable();
        n
    } else {
        let idx = args.iter().position(|a| a == "--config");
        match idx.and_then(|i| args.get(i + 1)) {
            Some(n) => vec![n.clone()],
            None => {
                eprintln!("params: pass --config <name> or --all");
                return 2;
            }
        }
    };
    let mut configs = Vec::new();
    for name in &names {
        match load_config(name) {
            Ok(c) => configs.push((name.clone(), c)),
            Err(e) => {
                eprintln!("params: {e}");
                return 1;
            }
        }
    }
    match fmt.as_str() {
        "markdown" => {
            let rows: Vec<(String, rhizome_core::config::ModelConfig)> = configs
                .iter()
                .map(|(n, c)| (n.clone(), c.model.clone()))
                .collect();
            println!("{}", rp::markdown_table(&rows));
        }
        "json" => {
            let arr = Json::Array(configs.iter().map(|(n, c)| config_json(n, c)).collect());
            println!("{}", json::to_string_pretty(&arr));
        }
        "table" => {
            for (name, cfg) in &configs {
                let b = rp::breakdown(&cfg.model);
                let a = rp::active(&cfg.model);
                println!("{name}:");
                println!("  total params      : {}", b.total());
                println!("  active @L=1       : {}", a.total_l1);
                println!("  per core iter     : {}", a.core_per_iter);
                println!(
                    "  flops/token @L=1  : {}",
                    rp::flops_per_token(&cfg.model, 1, cfg.train.seq_len)
                );
                println!(
                    "  deploy            : {} ({} bytes)",
                    cfg.model.deploy.name(),
                    rp::deploy_bytes(&cfg.model, &b)
                );
            }
        }
        other => {
            eprintln!("params: unknown format {other:?} (markdown|json|table)");
            return 2;
        }
    }
    0
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("version") => {
            println!("rhizome {}", env!("CARGO_PKG_VERSION"));
            0
        }
        Some("params") => cmd_params(&args[1..]),
        Some(other) => {
            eprintln!("rhizome: {other:?} lands with its milestone (docs/milestones.md)");
            eprintln!("{USAGE}");
            2
        }
        None => {
            eprintln!("{USAGE}");
            2
        }
    };
    std::process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_all_markdown_matches_xtask_generation() {
        // Same source of truth: the CLI and xtask must agree (R7).
        let rows: Vec<(String, rhizome_core::config::ModelConfig)> =
            ["edge", "flagship", "standard", "test-l", "test-m", "test-s"]
                .iter()
                .map(|n| {
                    let c = load_config(n).unwrap_or_else(|e| panic!("{e}"));
                    (n.to_string(), c.model)
                })
                .collect();
        let table = rp::markdown_table(&rows);
        assert!(table.contains("| d_model |"));
        assert!(table.contains("flagship"));
    }

    #[test]
    fn json_output_roundtrips() {
        let cfg = load_config("test-s").expect("test-s");
        let v = config_json("test-s", &cfg);
        let s = json::to_string(&v);
        let back = json::parse_str(&s).expect("reparse");
        assert_eq!(back.get("name").and_then(|j| j.as_str()), Some("test-s"));
    }
}
