use berth_ptx::{assemble_ids, reject_ids, OPS};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn marked(path: &str, tag: &str) -> BTreeSet<String> {
    let text = std::fs::read_to_string(repo_root().join(path)).unwrap();
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            let rest = line.strip_prefix("// ")?.strip_prefix(tag)?.trim();
            // A marker is `// PROBE ident`, not a sentence that happens to start with the word.
            if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
                Some(rest.to_string())
            } else {
                None
            }
        })
        .collect()
}

#[test]
fn ids_are_unique() {
    let mut seen = BTreeSet::new();
    for op in OPS {
        assert!(seen.insert(op.id), "duplicate id {}", op.id);
        assert!(!op.ptx.is_empty());
    }
}

#[test]
fn probe_comments_match_the_catalog() {
    let on_disk = marked("ptx/sm86_probes.cu", "PROBE");
    let expected: BTreeSet<String> = assemble_ids().map(str::to_string).collect();
    assert_eq!(on_disk, expected);
}

#[test]
fn reject_comments_match_the_catalog() {
    let on_disk = marked("ptx/reject_add_rn_f32_f16.cu", "REJECT");
    let expected: BTreeSet<String> = reject_ids().map(str::to_string).collect();
    assert_eq!(on_disk, expected);
}

#[test]
fn nvcc_accepts_sm86_and_rejects_f32_f16_add() {
    let script = repo_root().join("scripts/ptx_asm_check.sh");
    let status = Command::new("bash").arg(&script).status().unwrap();
    assert!(status.success(), "ptx asm check failed: {status}");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn this_host_is_avx2_without_avx512() {
    // berth's CPU paths assume the i9-9900K: AVX2 + FMA + F16C, no AVX-512.
    assert!(std::arch::is_x86_feature_detected!("avx2"));
    assert!(std::arch::is_x86_feature_detected!("fma"));
    assert!(std::arch::is_x86_feature_detected!("f16c"));
    assert!(!std::arch::is_x86_feature_detected!("avx512f"));
}
