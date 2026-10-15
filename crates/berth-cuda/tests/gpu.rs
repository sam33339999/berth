//! GPU tests. Each one returns immediately unless `BERTH_GPU_TESTS=1`.

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use berth_cuda::{
    ComputeCapability, Context, Cubin, DeviceBuffer, Dim3, Function, ParamSlot, ParamUpdate,
    PinnedBuf, Stream, PARAM_END,
};

static GPU: Mutex<()> = Mutex::new(());

fn lock_gpu() -> Option<MutexGuard<'static, ()>> {
    if std::env::var("BERTH_GPU_TESTS").ok().as_deref() != Some("1") {
        return None;
    }
    Some(GPU.lock().unwrap_or_else(|poison| poison.into_inner()))
}

#[test]
fn device_reports_the_3090() {
    let Some(_gpu) = lock_gpu() else {
        return;
    };
    device_reports_the_3090_inner().unwrap();
}

#[test]
fn memcpy_round_trip_device_and_pinned() {
    let Some(_gpu) = lock_gpu() else {
        return;
    };
    memcpy_round_trip_inner().unwrap();
}

#[test]
fn event_elapsed_between_streams() {
    let Some(_gpu) = lock_gpu() else {
        return;
    };
    event_elapsed_inner().unwrap();
}

#[test]
fn cubin_launch_writes_the_scalar() {
    let Some(_gpu) = lock_gpu() else {
        return;
    };
    cubin_launch_inner().unwrap();
}

#[test]
fn graph_param_update_changes_one_launch() {
    let Some(_gpu) = lock_gpu() else {
        return;
    };
    graph_param_update_inner().unwrap();
}

fn device_reports_the_3090_inner() -> berth_cuda::Result<()> {
    let ctx = Context::new()?;
    assert_eq!(ctx.ordinal(), 0);
    let cc = ctx.compute_capability()?;
    assert_eq!(
        cc,
        ComputeCapability { major: 8, minor: 6 },
        "device 0 compute capability {cc}"
    );
    assert_eq!(cc.sm_version(), 86);
    assert_eq!(ctx.multiprocessor_count()?, 82);
    let stream = ctx.stream()?;
    let handle = stream.handle();
    assert!(!handle.is_default());
    assert_eq!(handle.as_ptr() as usize, handle.as_u64() as usize);
    Ok(())
}

fn memcpy_round_trip_inner() -> berth_cuda::Result<()> {
    let ctx = Context::new()?;
    let stream = ctx.stream()?;

    let host = [1i32, 2, 3, 4];
    let device = DeviceBuffer::from_host(&stream, &host)?;
    assert_eq!(device.len(), 4);
    assert!(!device.ptr().is_null());
    let mut back = [0i32; 4];
    device.copy_to_host(&stream, &mut back)?;
    assert_eq!(back, host);

    let zeroed = DeviceBuffer::<i32>::zeroed(&stream, 4)?;
    let mut back = [7i32; 4];
    zeroed.copy_to_host(&stream, &mut back)?;
    assert_eq!(back, [0, 0, 0, 0]);

    let empty = DeviceBuffer::<i32>::alloc(&stream, 0)?;
    assert!(empty.is_empty());
    assert!(empty.ptr().is_null());
    let mut nowhere = [0i32; 0];
    empty.copy_to_host(&stream, &mut nowhere)?;

    let mismatch = device.copy_to_host(&stream, &mut [0i32; 2]);
    assert!(matches!(
        mismatch,
        Err(berth_cuda::Error::Length { host: 2, device: 4 })
    ));

    let pinned = PinnedBuf::from_slice(&ctx, &host)?;
    assert_eq!(pinned.as_slice(), &host);
    let mut device = DeviceBuffer::<i32>::alloc(&stream, 4)?;
    device.copy_from_pinned(&stream, &pinned)?;
    let mut pinned_back = PinnedBuf::<i32>::zeroed(&ctx, 4)?;
    unsafe { device.copy_to_pinned_async(&stream, &mut pinned_back)? };
    stream.synchronize()?;
    assert_eq!(pinned_back.as_slice(), &host);
    Ok(())
}

fn event_elapsed_inner() -> berth_cuda::Result<()> {
    let ctx = Context::new()?;
    let producer = ctx.stream()?;
    let consumer = ctx.stream()?;
    let start = ctx.event()?;
    let end = ctx.event()?;

    let bytes = 1usize << 20;
    let host = vec![1u8; bytes];
    let mut device = DeviceBuffer::alloc(&producer, bytes)?;

    start.record(&producer)?;
    unsafe { device.copy_from_host_async(&producer, &host)? };
    end.record(&producer)?;
    consumer.wait_event(&end)?;
    consumer.synchronize()?;

    let ms = start.elapsed_ms(&end)?;
    assert!(ms.is_finite(), "{ms}");
    assert!(ms >= 0.0, "{ms}");
    Ok(())
}

fn cubin_launch_inner() -> berth_cuda::Result<()> {
    let ctx = Context::new()?;
    let stream = ctx.stream()?;
    let function = load_fill(&ctx)?;
    let out = DeviceBuffer::<i32>::zeroed(&stream, 1)?;
    launch_fill(&function, &stream, out.ptr().as_u64(), 42)?;
    let mut host = [0i32];
    out.copy_to_host(&stream, &mut host)?;
    assert_eq!(host, [42]);
    Ok(())
}

fn graph_param_update_inner() -> berth_cuda::Result<()> {
    let ctx = Context::new()?;
    let stream = ctx.stream()?;
    let function = load_fill(&ctx)?;
    let first = DeviceBuffer::<i32>::zeroed(&stream, 1)?;
    let second = DeviceBuffer::<i32>::zeroed(&stream, 1)?;
    let first_ptr = first.ptr().as_u64();
    let second_ptr = second.ptr().as_u64();

    let mut graph = stream.capture(|capture| {
        let slots = [
            ParamSlot::pointer(1, 0),
            ParamSlot::u32_value(2, 1),
            ParamSlot::end(),
        ];
        unsafe { capture.record(&function, &slots)? };
        launch_fill(&function, &stream, first_ptr, 11)?;
        unsafe { capture.record(&function, &slots)? };
        launch_fill(&function, &stream, second_ptr, 22)?;
        Ok(())
    })?;
    assert_eq!(graph.launch_count(), 2);

    graph.launch(&stream)?;
    let mut host_first = [0i32];
    let mut host_second = [0i32];
    first.copy_to_host(&stream, &mut host_first)?;
    second.copy_to_host(&stream, &mut host_second)?;
    assert_eq!(host_first, [11]);
    assert_eq!(host_second, [22]);

    let same = 11i32.to_ne_bytes();
    let updated = 77i32.to_ne_bytes();
    graph.set_params(&[
        ParamUpdate::new(2, &same),
        ParamUpdate::new(PARAM_END, &[]),
        ParamUpdate::new(2, &updated),
    ])?;
    graph.launch(&stream)?;
    first.copy_to_host(&stream, &mut host_first)?;
    second.copy_to_host(&stream, &mut host_second)?;
    assert_eq!(host_first, [11]);
    assert_eq!(host_second, [77]);
    Ok(())
}

fn load_fill(ctx: &Context) -> berth_cuda::Result<Function> {
    let path = compile_fill_cubin();
    let cubin = Cubin::load_file(ctx, &path)?;
    cubin.function("fill_i32")
}

fn launch_fill(
    function: &Function,
    stream: &Stream,
    ptr: u64,
    value: i32,
) -> berth_cuda::Result<()> {
    let mut ptr = ptr;
    let mut value = value;
    let mut params = [
        (&raw mut ptr).cast::<c_void>(),
        (&raw mut value).cast::<c_void>(),
    ];
    unsafe { function.launch(stream, Dim3::one(1), Dim3::one(1), 0, &mut params) }
}

fn compile_fill_cubin() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    let out = dir.join("fill_i32.cubin");
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fill_i32.cu");
    let nvcc =
        std::env::var("BERTH_NVCC").unwrap_or_else(|_| "/usr/local/cuda-13.3/bin/nvcc".to_string());
    let output = std::process::Command::new(&nvcc)
        .args(["-arch=sm_86", "-cubin", "-o"])
        .arg(&out)
        .arg(&src)
        .output()
        .unwrap_or_else(|err| panic!("failed to run {nvcc}: {err}"));
    assert!(
        output.status.success(),
        "nvcc failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    out
}
