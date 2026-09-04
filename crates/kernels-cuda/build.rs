//! Build script for the CUDA backend: compiles `src/kernels/*.cu` with nvcc
//! to PTX and cubin (sm_80, sm_90) when the `cuda` feature is enabled, and
//! emits a manifest consumed by `RHIZOME_CUDA_KERNELS` in lib.rs.
//!
//! Never downloads anything (R5); a missing nvcc under `--features cuda` is
//! a hard error.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels");
    let feature_cuda = env::var_os("CARGO_FEATURE_CUDA").is_some();
    if !feature_cuda {
        // Nothing to compile; lib.rs reports the backend unavailable.
        return;
    }
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set"));
    let src =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir")).join("src/kernels");
    let nvcc = which_nvcc().unwrap_or_else(|| panic!("cuda feature requires nvcc on PATH"));
    let mut manifest = String::new();
    let kernels: Vec<PathBuf> = fs::read_dir(&src)
        .expect("kernel dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|e| e == "cu").unwrap_or(false))
        .collect();
    assert!(!kernels.is_empty(), "no .cu kernels found");
    for cu in &kernels {
        let name = cu.file_stem().expect("stem").to_string_lossy().to_string();
        for arch in ["sm_80", "sm_90"] {
            for kind in ["ptx", "cubin"] {
                let ext = if kind == "ptx" { "ptx" } else { "cubin" };
                let target = out_dir.join(format!("{name}-{arch}.{ext}"));
                let mut cmd = Command::new(&nvcc);
                // Precise math: no fast-math flags (determinism, R9).
                cmd.arg("-O3").arg("-std=c++17");
                if kind == "ptx" {
                    cmd.arg("-ptx");
                } else {
                    cmd.arg("-cubin");
                }
                cmd.arg(format!("-arch={arch}"))
                    .arg("-o")
                    .arg(&target)
                    .arg(cu);
                let status = cmd.status().unwrap_or_else(|e| panic!("run nvcc: {e}"));
                assert!(status.success(), "nvcc failed for {name} {arch} {kind}");
                manifest.push_str(&format!("{kind} {arch} {name}.cu {}\n", target.display()));
            }
        }
        println!("cargo:rerun-if-changed={}", cu.display());
    }
    let manifest_path = out_dir.join("kernels.manifest");
    fs::write(&manifest_path, &manifest).expect("write manifest");
    println!(
        "cargo:rustc-env=RHIZOME_CUDA_KERNELS={}",
        manifest_path.display()
    );
}

fn which_nvcc() -> Option<PathBuf> {
    if let Ok(path) = env::var("NVCC") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(path) = env::var("PATH") {
        for dir in path.split(':') {
            let p = PathBuf::from(dir).join("nvcc");
            if p.exists() {
                return Some(p);
            }
        }
    }
    None
}
