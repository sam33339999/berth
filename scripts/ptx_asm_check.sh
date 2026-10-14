#!/usr/bin/env bash
# Assemble every sm_86 probe, and require the sm_100-only op to be rejected.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
out=$root/ptx/out
mkdir -p "$out"

nvcc -arch=sm_86 -cubin "$root/ptx/sm86_probes.cu" -o "$out/sm86_probes.cubin"

if nvcc -arch=sm_86 -cubin "$root/ptx/reject_add_rn_f32_f16.cu" -o "$out/reject_add_rn_f32_f16.cubin" 2>"$out/reject_add_rn_f32_f16.log"; then
  echo "add.rn.f32.f16 assembled for sm_86; the fallback gate is no longer valid" >&2
  exit 1
fi
grep -q "Unexpected instruction types" "$out/reject_add_rn_f32_f16.log"

echo "sm_86 probes assembled; add.rn.f32.f16 rejected"
