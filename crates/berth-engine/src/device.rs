//! Shared device plumbing: the context and stream, checkpoint uploads, and
//! the EXL3 linear every layer uses.

use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use berth_cuda::{ByteBuf, Context, DeviceBuffer, Stream};
use berth_exl3::{Bitrate, Checkpoint, HadamardScale, LinearExl3, TensorInfo};
use berth_kernels::{Codebook, OutDtype};

use crate::{Error, Result};

/// Device 0, the stream every launch goes on, and its SM count.
pub struct Gpu {
    pub ctx: Context,
    pub stream: Stream,
    pub sm_count: i32,
}

impl Gpu {
    /// Open device 0 and a stream on it.
    pub fn new() -> Result<Self> {
        let ctx = Context::new()?;
        let stream = ctx.stream()?;
        let sm_count = ctx.multiprocessor_count()? as i32;
        Ok(Self {
            ctx,
            stream,
            sm_count,
        })
    }

    /// The stream as the `cudaStream_t` the kernel shims take.
    pub fn raw_stream(&self) -> *mut c_void {
        self.stream.handle().as_ptr()
    }

    /// A zeroed device buffer of `bytes` bytes.
    pub fn zeroed(&self, bytes: usize) -> Result<ByteBuf> {
        Ok(DeviceBuffer::zeroed(&self.stream, bytes)?)
    }
}

/// Raw device address of a buffer, for the kernel shims.
pub fn ptr(buf: &ByteBuf) -> *mut c_void {
    buf.ptr().as_u64() as *mut c_void
}

/// Read one tensor's bytes from its shard.
pub fn read_tensor(info: &TensorInfo) -> Result<Vec<u8>> {
    let read = || -> std::io::Result<Vec<u8>> {
        let mut file = File::open(&info.shard)?;
        file.seek(SeekFrom::Start(info.offset))?;
        let mut bytes = vec![0u8; info.length as usize];
        file.read_exact(&mut bytes)?;
        Ok(bytes)
    };
    read().map_err(|source| Error::Read {
        name: info.name.clone(),
        source,
    })
}

/// Copy one checkpoint tensor, as stored, to the device.
pub fn upload_tensor(gpu: &Gpu, ckpt: &Checkpoint, name: &str) -> Result<ByteBuf> {
    let info = ckpt
        .tensors
        .get(name)
        .ok_or_else(|| Error::Weights(format!("tensor {name} is missing")))?;
    let bytes = read_tensor(info)?;
    Ok(DeviceBuffer::from_host(&gpu.stream, &bytes)?)
}

/// One EXL3 linear on the device: trellis, fp16 Hadamard scales, and the
/// fp16 scratch the kernel writes the transformed input to.
pub struct Exl3Linear {
    pub prefix: String,
    trellis: ByteBuf,
    suh: ByteBuf,
    svh: ByteBuf,
    a_had: ByteBuf,
    k_bits: i32,
    codebook: Codebook,
    /// In-features.
    pub size_k: i32,
    /// Out-features.
    pub size_n: i32,
}

impl Exl3Linear {
    /// Load `<prefix>.trellis`, `.suh`, `.svh` for decode (m = 1).
    pub fn load(gpu: &Gpu, ckpt: &Checkpoint, prefix: &str) -> Result<Self> {
        let linear = find_linear(ckpt, prefix)?;
        let Bitrate::Int(k_bits) = linear.k else {
            return Err(Error::Weights(format!("{prefix}: half-integer rate")));
        };
        let codebook = match linear.codebook {
            berth_exl3::Codebook::Mul1 => Codebook::Mul1,
            berth_exl3::Codebook::Mcg => Codebook::Mcg,
            berth_exl3::Codebook::Default => Codebook::Default,
        };
        let (suh, svh) = match (&linear.input_scale, &linear.output_scale) {
            (HadamardScale::Unpacked(suh), HadamardScale::Unpacked(svh)) => (suh, svh),
            _ => {
                return Err(Error::Weights(format!(
                    "{prefix}: packed su / sv signs are not handled yet"
                )))
            }
        };
        let size_k = suh.shape.iter().product::<u64>() as i32;
        let size_n = svh.shape.iter().product::<u64>() as i32;
        Ok(Self {
            prefix: prefix.to_string(),
            trellis: DeviceBuffer::from_host(&gpu.stream, &read_tensor(&linear.trellis)?)?,
            suh: DeviceBuffer::from_host(&gpu.stream, &read_tensor(suh)?)?,
            svh: DeviceBuffer::from_host(&gpu.stream, &read_tensor(svh)?)?,
            a_had: gpu.zeroed(size_k as usize * 2)?,
            k_bits: i32::from(k_bits),
            codebook,
            size_k,
            size_n,
        })
    }

    /// `c = a @ W` for one row. `a` is fp16 `[size_k]`; `c` is `[size_n]` of `out`.
    pub fn forward(
        &mut self,
        gpu: &Gpu,
        a: &ByteBuf,
        c: &mut ByteBuf,
        out: OutDtype,
    ) -> Result<()> {
        let out_bytes = match out {
            OutDtype::F16 => 2,
            OutDtype::F32 => 4,
        };
        if a.num_bytes() < self.size_k as usize * 2
            || c.num_bytes() < self.size_n as usize * out_bytes
        {
            return Err(Error::Input(format!("{}: buffer too small", self.prefix)));
        }
        // SAFETY: every buffer is a live device allocation at least as large
        // as this shape needs, all on `gpu.stream`.
        unsafe {
            berth_kernels::exl3_linear(
                ptr(a),
                ptr(&self.trellis),
                ptr(c),
                ptr(&self.suh),
                ptr(&self.a_had),
                ptr(&self.svh),
                1,
                self.size_k,
                self.size_n,
                self.k_bits,
                self.codebook,
                out,
                gpu.raw_stream(),
            )?;
        }
        Ok(())
    }
}

/// The checkpoint's EXL3 linear with this prefix.
pub fn find_linear<'a>(ckpt: &'a Checkpoint, prefix: &str) -> Result<&'a LinearExl3> {
    ckpt.linears()
        .iter()
        .find(|linear| linear.prefix == prefix)
        .ok_or_else(|| Error::Weights(format!("{prefix} is not an EXL3 linear")))
}
