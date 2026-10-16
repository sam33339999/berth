//! Compile the EXL3 shim with the fork's nvcc, then archive it.
//!
//! nvcc is invoked directly, not through the `cc` crate. `cc`'s CUDA mode
//! device-links the archive (`nvcc --device-link`), which is for relocatable
//! device code. The fork's `setup.py` does not pass `-rdc`; each translation
//! unit keeps its own fatbin and the dynamic CUDA runtime registers it. The
//! same `-c` line is used here. cudart is the shared library from the CUDA
//! 13.3 toolkit (`libcudart.so.13`), matching that toolkit rather than the
//! CUDA 12.4 driver at `/usr/bin/nvcc`.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;

const EXT_TREE: &str = "de35ffbe336bdc6d617e49e208cd7c730a789787";

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-env-changed=BERTH_EXL3_SRC");
    println!("cargo:rerun-if-env-changed=BERTH_NVCC");

    let src = exl3_src();
    watch_fork_head(&src);
    let got = ext_tree(&src);
    if got != EXT_TREE {
        panic!(
            "berth-kernels: `{}` tree `exllamav3/exllamav3_ext` is `{got}`, expected `{EXT_TREE}`.\n\
             The fork kernel sources must match the running service (git rev-parse HEAD:exllamav3/exllamav3_ext).\n\
             Set BERTH_EXL3_SRC if the fork is not at the default path. Refusing to build.",
            src.display()
        );
    }
    let dirty = ext_dirty(&src);
    if !dirty.is_empty() {
        panic!(
            "berth-kernels: `{}` has uncommitted changes under `exllamav3/exllamav3_ext`; the tree hash check \
             only covers HEAD, and nvcc would compile the working tree.\n{dirty}Refusing to build.",
            src.display()
        );
    }

    let nvcc = nvcc_path();
    let ext = src.join("exllamav3/exllamav3_ext");
    let quant = ext.join("quant");
    if !quant.join("exl3_gemm_kernel.cuh").is_file() {
        panic!(
            "berth-kernels: {} is missing. BERTH_EXL3_SRC should be the fork repo root.",
            quant.join("exl3_gemm_kernel.cuh").display()
        );
    }

    let csrc = manifest.join("csrc");
    extract_fork_kernels(&ext, &out_dir);
    // Every `.cu` under csrc/, recursively, sorted. One object per file;
    // a file under a subdirectory (`csrc/gdn/x.cu`) becomes `gdn__x.o` in OUT_DIR.
    let units = cuda_units(&csrc);
    let mut objects = Vec::new();
    let mut handles = Vec::new();
    for unit in units {
        let obj = out_dir.join(unit.replace('/', "__")).with_extension("o");
        objects.push(obj.clone());
        let nvcc = nvcc.clone();
        let src_file = csrc.join(unit);
        let ext = ext.clone();
        let quant = quant.clone();
        let csrc = csrc.clone();
        let gen = out_dir.clone();
        handles.push(thread::spawn(move || {
            compile(&nvcc, &src_file, &obj, &ext, &quant, &csrc, &gen);
        }));
    }
    for handle in handles {
        if let Err(payload) = handle.join() {
            if let Some(msg) = payload.downcast_ref::<String>() {
                panic!("{msg}");
            }
            if let Some(msg) = payload.downcast_ref::<&str>() {
                panic!("{msg}");
            }
            panic!("nvcc thread panicked");
        }
    }

    let lib = out_dir.join("libberth_exl3_kernels.a");
    let status = Command::new("ar")
        .arg("crs")
        .arg(&lib)
        .args(&objects)
        .status()
        .unwrap_or_else(|err| panic!("failed to run ar: {err}"));
    if !status.success() {
        panic!("ar failed to archive {}", lib.display());
    }

    let cuda_lib = nvcc
        .parent()
        .and_then(|bin| bin.parent())
        .map(|root| root.join("lib64"))
        .filter(|path| path.is_dir())
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda-13.3/lib64"));

    // Prefill LIN. hgemm.cu's cublasGemmEx must be torch's libcublas, the
    // copy the serve loads (nvidia/cu13/lib/libcublas.so.13 via
    // reuse-torch.pth). The CUDA 13.3 toolkit also ships libcublas.so.13, and
    // a RUNPATH that lists that directory first binds the toolkit copy.
    // The two are different releases (torch ships 13.1, the toolkit 13.6),
    // so binding the toolkit copy would change the numbers.
    //
    // `<target>/cublas-torch` holds symlinks to torch's libcublas and
    // libcublasLt only: `libcublas.so` for `-lcublas` at link time (searched
    // before the toolkit's lib64, which has its own `libcublas.so`), and the
    // `.so.13` sonames for run time. Every binary that links berth-kernels
    // finds that directory through the `$ORIGIN` rpaths in
    // `.cargo/config.toml` (`cargo:rustc-link-arg` would reach only this
    // crate's own tests). libcudart.so.13 is not in it, so cudart still
    // comes from the 13.3 toolkit. libcublas's own RUNPATH is `$ORIGIN`;
    // the Lt symlink sits beside the cublas symlink.
    let cublas = torch_cublas();
    let cublas_dir = cublas.parent().unwrap_or_else(|| {
        panic!("berth-kernels: {} has no parent", cublas.display());
    });
    let cublas_lt = cublas_dir.join("libcublasLt.so.13");
    if !cublas_lt.is_file() {
        panic!(
            "berth-kernels: {} is missing; torch libcublas needs the sibling libcublasLt.so.13",
            cublas_lt.display()
        );
    }
    // OUT_DIR is <target>/<profile>/build/<pkg>-<hash>/out.
    let target_dir = out_dir
        .ancestors()
        .nth(4)
        .unwrap_or_else(|| panic!("berth-kernels: unexpected OUT_DIR {}", out_dir.display()));
    let pin = target_dir.join("cublas-torch");
    std::fs::create_dir_all(&pin)
        .unwrap_or_else(|err| panic!("berth-kernels: cannot create {}: {err}", pin.display()));
    for (name, target) in [
        ("libcublas.so", &cublas),
        ("libcublas.so.13", &cublas),
        ("libcublasLt.so.13", &cublas_lt),
    ] {
        let link = pin.join(name);
        if link.symlink_metadata().is_ok() {
            std::fs::remove_file(&link).unwrap_or_else(|err| {
                panic!("berth-kernels: cannot replace {}: {err}", link.display())
            });
        }
        std::os::unix::fs::symlink(target, &link).unwrap_or_else(|err| {
            panic!("berth-kernels: cannot symlink {}: {err}", link.display())
        });
    }
    println!("cargo:rustc-link-search=native={}", pin.display());
    println!("cargo:rustc-link-lib=dylib=cublas");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=berth_exl3_kernels");
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

/// `libcublas.so.13` from the torch the serve imports.
///
/// `/home/sam/qwen38-exl3-dflash2/env` points at that torch with
/// `reuse-torch.pth` (torch 2.13, `nvidia/cu13/lib`). An older wheel layout
/// (`nvidia/cublas/lib`) is accepted if the pth target has it.
fn torch_cublas() -> PathBuf {
    let pth = PathBuf::from(
        "/home/sam/qwen38-exl3-dflash2/env/lib/python3.13/site-packages/reuse-torch.pth",
    );
    println!("cargo:rerun-if-changed={}", pth.display());
    let mut site = PathBuf::from("/home/sam/qwen38-exl3-dflash2/env/lib/python3.13/site-packages");
    if let Ok(text) = std::fs::read_to_string(&pth) {
        if let Some(line) = text
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty() && !line.starts_with('#'))
        {
            site = PathBuf::from(line);
        }
    }
    let candidates = [
        site.join("nvidia/cu13/lib/libcublas.so.13"),
        site.join("nvidia/cublas/lib/libcublas.so.13"),
        site.join("nvidia/cublas/lib/libcublas.so"),
    ];
    for candidate in &candidates {
        if candidate.is_file() {
            println!("cargo:rerun-if-changed={}", candidate.display());
            return candidate.clone();
        }
    }
    panic!(
        "berth-kernels: torch libcublas not found under {} (tried nvidia/cu13/lib and nvidia/cublas/lib).\n\
         Prefill hgemm has to link the same cuBLAS the fork's hgemm.cu calls through torch.",
        site.display()
    );
}

/// One fork source range: file under `exllamav3_ext/`, first and last line
/// (1-based, inclusive), and a string the range must contain.
struct Part {
    file: &'static str,
    first: usize,
    last: usize,
    contains: &'static str,
}

/// A header written to `OUT_DIR/fork/<name>`, made of fork line ranges.
struct Extract {
    name: &'static str,
    parts: &'static [Part],
}

/// Device code the fork defines inside `.cu` files that also include ATen,
/// so it cannot be `#include`d. The ranges are read at build time from the
/// fork at tree [`EXT_TREE`]; nothing is copied into this repo. A tree change
/// already stops the build, and `contains` catches a wrong range.
const FORK_EXTRACTS: &[Extract] = &[
    Extract {
        name: "gdn_kernels.cuh",
        parts: &[
            Part {
                file: "gdn.cu",
                first: 14,
                last: 51,
                contains: "softplus",
            },
            Part {
                file: "gdn.cu",
                first: 660,
                last: 849,
                contains: "cuda_recurrent_gated_delta_rule_kernel_128",
            },
            Part {
                file: "gdn.cu",
                first: 1309,
                last: 1406,
                contains: "conv1d_update_kernel",
            },
            Part {
                file: "gdn.cu",
                first: 1513,
                last: 1561,
                contains: "gated_delta_net_fused_op_3_kernel",
            },
            Part {
                file: "gdn.cu",
                first: 1630,
                last: 1669,
                contains: "gdn_ba_gemv_kernel",
            },
        ],
    },
    // Rewind kernel bodies only. The host wrappers at 1962–2010 and the job
    // structs in gdn.cuh use ATen, so those stay out of the extract. The
    // launcher in csrc/gdn/gdn_rewind.cu declares the POD jobs itself.
    Extract {
        name: "gdn_rewind_kernels.cuh",
        parts: &[Part {
            file: "gdn.cu",
            first: 1923,
            last: 1960,
            contains: "batched_conv_rewind_kernel",
        }],
    },
    Extract {
        name: "norm_kernels.cuh",
        parts: &[
            Part {
                file: "norm.cu",
                first: 9,
                last: 309,
                contains: "rms_norm_kernel",
            },
            Part {
                file: "norm.cu",
                first: 499,
                last: 598,
                contains: "gated_rms_norm_kernel",
            },
        ],
    },
    Extract {
        name: "deinterleave_qg.cuh",
        parts: &[Part {
            file: "activation.cu",
            first: 721,
            last: 738,
            contains: "deinterleave_qg_kernel",
        }],
    },
    Extract {
        name: "rope_kernel.cuh",
        parts: &[Part {
            file: "rope.cu",
            first: 10,
            last: 260,
            contains: "rope_kernel",
        }],
    },
    // Sampler device code. These .cu files include ATen above the kernels,
    // so the host launchers in csrc/sample include the generated headers.
    Extract {
        name: "rep_pen_kernels.cuh",
        parts: &[
            Part {
                file: "generator/rep_pen.cu",
                first: 9,
                last: 91,
                contains: "apply_rep_pens_kernel",
            },
            Part {
                file: "generator/rep_pen.cu",
                first: 137,
                last: 201,
                contains: "apply_pres_freq_pens_kernel",
            },
        ],
    },
    Extract {
        name: "dry_kernel.cuh",
        parts: &[Part {
            file: "generator/dry.cu",
            first: 30,
            last: 132,
            contains: "dry_penalty_kernel",
        }],
    },
    Extract {
        name: "gumbel_kernels.cuh",
        parts: &[Part {
            file: "generator/gumbel.cu",
            first: 10,
            last: 94,
            contains: "gumbel_noise_kernel_f32",
        }],
    },
    Extract {
        name: "sampling_fused_kernels.cuh",
        parts: &[Part {
            file: "generator/sampling_fused.cu",
            first: 32,
            last: 768,
            contains: "fs_partial_argmax_kernel",
        }],
    },
    // Prefill LIN: rows > 1 EXL3 linear. Device code only; the host side of
    // these .cu files includes ATen. csrc/prefill_lin launches the templates.
    Extract {
        name: "prefill_reconstruct.cuh",
        parts: &[
            Part {
                file: "quant/reconstruct.cu",
                first: 11,
                last: 97,
                contains: "reconstruct_kernel",
            },
            Part {
                file: "quant/reconstruct.cu",
                first: 200,
                last: 369,
                contains: "reconstruct_had_kernel",
            },
        ],
    },
    Extract {
        name: "prefill_hadamard.cuh",
        parts: &[Part {
            file: "quant/hadamard.cu",
            first: 9,
            last: 37,
            contains: "had_hf_r_128_kernel",
        }],
    },
    Extract {
        name: "prefill_f16acc.cuh",
        parts: &[Part {
            file: "hgemm_f16acc.cu",
            first: 25,
            last: 332,
            contains: "gemm_kernel",
        }],
    },
    // GDNP: Gated DeltaNet prefill. fused_op_2 plus a private copy of the
    // device helpers it calls (FUSED_OP_2_THREADS, _sigmoid_fast_exp, softplus,
    // trunc_bf16, as_float). The prefill TU must not include gdn_kernels.cuh.
    Extract {
        name: "gdn_prefill_kernels.cuh",
        parts: &[
            Part {
                file: "gdn.cu",
                first: 19,
                last: 51,
                contains: "FUSED_OP_2_THREADS",
            },
            Part {
                file: "gdn.cu",
                first: 237,
                last: 270,
                contains: "gated_delta_net_fused_op_2_kernel",
            },
        ],
    },
];

fn extract_fork_kernels(ext: &Path, out_dir: &Path) {
    let dir = out_dir.join("fork");
    std::fs::create_dir_all(&dir)
        .unwrap_or_else(|err| panic!("berth-kernels: cannot create {}: {err}", dir.display()));
    for extract in FORK_EXTRACTS {
        let mut text = format!(
            "// Generated by berth-kernels build.rs from the exllamav3 fork, tree {EXT_TREE}.\n\
             // Do not edit; change the ranges in build.rs instead.\n#pragma once\n"
        );
        for part in extract.parts {
            let path = ext.join(part.file);
            let source = std::fs::read_to_string(&path).unwrap_or_else(|err| {
                panic!("berth-kernels: cannot read {}: {err}", path.display())
            });
            let lines: Vec<&str> = source.lines().collect();
            if part.first == 0 || part.last < part.first || part.last > lines.len() {
                panic!(
                    "berth-kernels: range {}:{}-{} is outside the file ({} lines)",
                    part.file,
                    part.first,
                    part.last,
                    lines.len()
                );
            }
            let body = lines[part.first - 1..part.last].join("\n");
            if !body.contains(part.contains) {
                panic!(
                    "berth-kernels: {}:{}-{} does not contain `{}`; the extract ranges in build.rs are wrong",
                    part.file, part.first, part.last, part.contains
                );
            }
            text.push_str(&format!(
                "#line {} \"{}\"\n{body}\n",
                part.first,
                path.display()
            ));
        }
        let out = dir.join(extract.name);
        std::fs::write(&out, text)
            .unwrap_or_else(|err| panic!("berth-kernels: cannot write {}: {err}", out.display()));
    }
}

fn cuda_units(csrc: &Path) -> Vec<String> {
    fn walk(dir: &Path, root: &Path, out: &mut Vec<String>) {
        let entries = std::fs::read_dir(dir)
            .unwrap_or_else(|err| panic!("berth-kernels: cannot read {}: {err}", dir.display()));
        for entry in entries {
            let path = entry.expect("csrc entry").path();
            if path.is_dir() {
                walk(&path, root, out);
            } else if path.extension().is_some_and(|ext| ext == "cu") {
                let rel = path.strip_prefix(root).expect("under csrc");
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    let mut units = Vec::new();
    walk(csrc, csrc, &mut units);
    units.sort();
    units
}

fn exl3_src() -> PathBuf {
    match env::var("BERTH_EXL3_SRC") {
        Ok(path) if !path.is_empty() => PathBuf::from(path),
        _ => PathBuf::from("/home/sam/qwen38-exl3-dflash2/exllamav3"),
    }
}

fn nvcc_path() -> PathBuf {
    let path = match env::var("BERTH_NVCC") {
        Ok(path) if !path.is_empty() => PathBuf::from(path),
        _ => PathBuf::from("/usr/local/cuda-13.3/bin/nvcc"),
    };
    if !path.is_file() {
        panic!(
            "berth-kernels: nvcc not found at {}.\n\
             The fork extension was built with CUDA 13.3. /usr/bin/nvcc is 12.4 and must not be used.\n\
             Set BERTH_NVCC to the CUDA 13.3 nvcc.",
            path.display()
        );
    }
    path
}

fn watch_fork_head(src: &Path) {
    let head = src.join(".git/HEAD");
    if !head.is_file() {
        return;
    }
    println!("cargo:rerun-if-changed={}", head.display());
    let text = std::fs::read_to_string(&head).unwrap_or_default();
    let Some(rest) = text.trim().strip_prefix("ref: ") else {
        return;
    };
    let ref_file = src.join(".git").join(rest);
    if ref_file.is_file() {
        println!("cargo:rerun-if-changed={}", ref_file.display());
    }
}

/// `git status --porcelain` for the kernel tree: modified, staged, or untracked files.
fn ext_dirty(src: &Path) -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(src)
        .args(["status", "--porcelain", "--", "exllamav3/exllamav3_ext"])
        .output()
        .unwrap_or_else(|err| {
            panic!(
                "berth-kernels: failed to run git status in {}: {err}",
                src.display()
            )
        });
    if !output.status.success() {
        panic!("berth-kernels: git status failed in {}", src.display());
    }
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn ext_tree(src: &Path) -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(src)
        .args(["rev-parse", "HEAD:exllamav3/exllamav3_ext"])
        .output()
        .unwrap_or_else(|err| {
            panic!(
                "berth-kernels: failed to run git -C {} rev-parse HEAD:exllamav3/exllamav3_ext: {err}",
                src.display()
            )
        });
    if !output.status.success() {
        panic!(
            "berth-kernels: git rev-parse failed for {} (exit {:?}).\n\
             stderr: {}\n\
             BERTH_EXL3_SRC must be the exllamav3 fork checkout, not a copy of the sources.",
            src.display(),
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn compile(nvcc: &Path, src: &Path, obj: &Path, ext: &Path, quant: &Path, csrc: &Path, gen: &Path) {
    let mut cmd = Command::new(nvcc);
    cmd.args([
        "-c",
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        "-arch=sm_86",
        "-std=c++17",
        "-Xcudafe",
        "--diag_suppress=177",
        "-Xcudafe",
        "--diag_suppress=20012",
        "-Xcompiler",
        "-fPIC",
    ])
    .arg("-I")
    .arg(csrc)
    .arg("-I")
    .arg(gen)
    .arg("-I")
    .arg(ext)
    .arg("-I")
    .arg(quant)
    .arg(src)
    .arg("-o")
    .arg(obj);
    let output = cmd
        .output()
        .unwrap_or_else(|err| panic!("failed to run {}: {err}", nvcc.display()));
    if !output.status.success() {
        panic!(
            "nvcc failed on {}:\n{}{}",
            src.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
