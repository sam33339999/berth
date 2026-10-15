//! One EXL3 linear on the GPU, bit for bit against the fork.
//!
//! Reads each case in `testdata/exl3_ref/meta.json`, loads that linear's
//! trellis and Hadamard scales from the 27B checkpoint with `berth-exl3`,
//! copies them to the device with `berth-cuda`, launches `berth-kernels`, and
//! compares the output bytes with `scripts/dump_exl3_reference.py`'s.
//!
//! Needs the GPU: runs only with `BERTH_GPU_TESTS=1`, and skips when the
//! checkpoint directory is missing.

use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use berth_cuda::{Context, DeviceBuffer, Stream};
use berth_exl3::{Bitrate, Checkpoint, HadamardScale, LinearExl3, TensorInfo};
use berth_kernels::{exl3_linear, last_launch, Codebook, OutDtype};

fn model_dir() -> String {
    std::env::var("BERTH_27B_DIR")
        .unwrap_or_else(|_| "/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3".to_string())
}

fn ref_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/exl3_ref")
}

fn read_tensor(info: &TensorInfo) -> Vec<u8> {
    let mut file = File::open(&info.shard).unwrap();
    file.seek(SeekFrom::Start(info.offset)).unwrap();
    let mut bytes = vec![0u8; info.length as usize];
    file.read_exact(&mut bytes).unwrap();
    bytes
}

fn upload(stream: &Stream, bytes: &[u8]) -> DeviceBuffer<u8> {
    DeviceBuffer::from_host(stream, bytes).unwrap()
}

fn unpacked(scale: &HadamardScale) -> &TensorInfo {
    match scale {
        HadamardScale::Unpacked(info) => info,
        HadamardScale::Packed(_) => panic!("27B linears store fp16 suh / svh"),
    }
}

fn codebook(linear: &LinearExl3) -> Codebook {
    match linear.codebook {
        berth_exl3::Codebook::Mul1 => Codebook::Mul1,
        berth_exl3::Codebook::Mcg => Codebook::Mcg,
        berth_exl3::Codebook::Default => Codebook::Default,
    }
}

#[test]
fn exl3_linears_match_the_fork_bit_for_bit() {
    if std::env::var("BERTH_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping exl3_ref: set BERTH_GPU_TESTS=1");
        return;
    }
    let dir = model_dir();
    if !Path::new(&dir).is_dir() {
        eprintln!("skipping exl3_ref: directory missing: {dir}");
        return;
    }

    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(ref_dir().join("meta.json")).unwrap()).unwrap();
    let ckpt = Checkpoint::open(&dir).unwrap();
    let ctx = Context::new().unwrap();
    let stream = ctx.stream().unwrap();

    let mut failures = Vec::new();
    for case in meta["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let key = case["key"].as_str().unwrap();
        let linear = ckpt
            .linears()
            .iter()
            .find(|linear| linear.prefix == key)
            .unwrap_or_else(|| panic!("{key} is not an EXL3 linear in the checkpoint"));
        let Bitrate::Int(k_bits) = linear.k else {
            panic!("{key}: half-integer rate");
        };
        let size_k = case["x_shape"].as_array().unwrap().last().unwrap();
        let size_k = size_k.as_u64().unwrap() as i32;
        let size_n = case["y_shape"].as_array().unwrap().last().unwrap();
        let size_n = size_n.as_u64().unwrap() as i32;
        let (out, out_bytes) = match case["y_dtype"].as_str().unwrap() {
            "float16" => (OutDtype::F16, 2),
            "float32" => (OutDtype::F32, 4),
            other => panic!("{name}: output dtype {other}"),
        };

        let x = std::fs::read(ref_dir().join(format!("{name}.x.bin"))).unwrap();
        let want = std::fs::read(ref_dir().join(format!("{name}.y.bin"))).unwrap();
        assert_eq!(x.len(), size_k as usize * 2, "{name}: x size");
        assert_eq!(want.len(), size_n as usize * out_bytes, "{name}: y size");

        let a = upload(&stream, &x);
        let trellis = upload(&stream, &read_tensor(&linear.trellis));
        let suh = upload(&stream, &read_tensor(unpacked(&linear.input_scale)));
        let svh = upload(&stream, &read_tensor(unpacked(&linear.output_scale)));
        let a_had = DeviceBuffer::<u8>::zeroed(&stream, x.len()).unwrap();
        let c = DeviceBuffer::<u8>::zeroed(&stream, want.len()).unwrap();

        // SAFETY: every buffer is a live device allocation sized for this
        // shape, and the stream is synchronized before any of them drops.
        unsafe {
            exl3_linear(
                a.ptr().as_u64() as *const c_void,
                trellis.ptr().as_u64() as *const c_void,
                c.ptr().as_u64() as *mut c_void,
                suh.ptr().as_u64() as *const c_void,
                a_had.ptr().as_u64() as *mut c_void,
                svh.ptr().as_u64() as *const c_void,
                1,
                size_k,
                size_n,
                i32::from(k_bits),
                codebook(linear),
                out,
                stream.handle().as_ptr(),
            )
        }
        .unwrap_or_else(|err| panic!("{name}: {err}"));
        let kernel = last_launch();
        stream.synchronize().unwrap();

        let mut got = vec![0u8; want.len()];
        c.copy_to_host(&stream, &mut got).unwrap();
        let differing = got
            .chunks(out_bytes)
            .zip(want.chunks(out_bytes))
            .filter(|(g, w)| g != w)
            .count();
        eprintln!("{name}: {kernel}, {differing} of {size_n} elements differ");
        if differing != 0 {
            failures.push(format!("{name}: {differing} of {size_n} elements differ"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
