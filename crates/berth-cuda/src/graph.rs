//! CUDA graphs whose kernel parameters can be patched after instantiation.
//!
//! This replaces exllamav3 `graph.cu` / `graph.cuh`. The fork records every
//! patched argument with `record_param(kernel, param_id, param_offset, size)`
//! and ends a launch's group with the sentinel `GP_end` (enum value 0), so two
//! launches of the same function stay separate. `capture_end` instantiates the
//! graph with flags 0, lists the nodes, and reads each kernel node's
//! `kernelParams`. `Graph::launch` walks recorded sites against an ordered list
//! of `(param_id, value)` updates: a site whose id does not match is skipped,
//! `GP_end` matches but is not written, and the first matching site consumes
//! the update. Changed bytes are copied into the node-owned `kernelParams[i]`
//! storage (only `size` bytes, because adjacent arguments may be packed) and
//! the node is marked dirty. Dirty nodes are pushed with
//! `cuGraphExecKernelNodeSetParams` before `cudaGraphLaunch`.
//!
//! The mapping here:
//!
//! * [`Capture::record`] is one call per launch. That call is the fork's
//!   `GP_end`-terminated group. Each [`ParamSlot`] is `(key, index, size)`:
//!   `key` is `param_id`, `index` is `param_offset` (the `kernelParams` index).
//! * [`PARAM_END`] is `GP_end` (0). [`Graph::set_params`] walks sites the same
//!   way and does not write a [`PARAM_END`] site. Updates are a subsequence of
//!   the recorded keys, in order. An earlier site with the same key consumes
//!   the update; the walk cannot skip it.
//! * [`Graph::set_launch_param`] addresses one recorded launch's argument
//!   directly.
//! * Capture uses thread-local mode, as the fork does. The fork calls
//!   `cudaDeviceSynchronize` first because other torch work may be in flight,
//!   and it allocates a private stream. This crate only synchronizes the stream
//!   being captured, and launches are recorded on the stream the caller passes.
//! * Launches here go through the driver API, so node parameters are read and
//!   written with the driver v2 entry points (`cuGraphKernelNodeGetParams_v2`,
//!   `cuGraphExecKernelNodeSetParams_v2`). The fork also tries the runtime API
//!   and falls back to the driver for `cuLaunchKernel` nodes. Nothing captured
//!   through this crate needs the runtime fallback.
//! * A new value is a byte slice of the recorded size. The fork stuffs the
//!   value into a `void *` and copies `size` bytes of that pointer. Pass the
//!   native-endian bytes of the argument (`i32::to_ne_bytes` for a 4-byte
//!   scalar, [`crate::DevicePtr::to_ne_bytes`] for an 8-byte pointer).
//!
//! The driver owns `kernelParams` and says it should not be modified directly.
//! The fork writes those bytes in place and then calls
//! `cuGraphExecKernelNodeSetParams`. [`Graph::set_params`] does the same. The
//! pointers stay valid for the life of the graph. Device buffers named by
//! captured pointer arguments are not owned: the caller keeps them alive.
//!
//! cuda-core wraps only begin/end capture. Instantiation, node queries,
//! parameter updates, and launch go through `cuda_core::sys`.

use std::mem::MaybeUninit;
use std::sync::Arc;

use cuda_core::Device as RuntimeDevice;
use cuda_core::IntoResult;
use cuda_core::Stream as RuntimeStream;

use crate::context::Context;
use crate::error::{Error, Result};
use crate::module::Function;
use crate::raw::device_as_c_int;
use crate::stream::Stream;

/// Largest `kernelParams` slot this crate will record, in bytes.
///
/// Rope's position arguments and a device pointer both fit. The cap rejects a
/// size that cannot be a single kernel argument before capture starts.
pub const MAX_PARAM_BYTES: usize = 256;

/// Highest `kernelParams` index this crate will record.
///
/// The check is not a substitute for knowing the kernel's real arity. An index
/// inside this range can still be out of range for a particular kernel, which
/// is why [`Capture::record`] is unsafe.
pub const MAX_PARAM_INDEX: usize = 256;

/// Sentinel key, the fork's `GP_end`.
///
/// A site with this key matches an update and consumes it, and the bytes are
/// not written. Real parameter keys start at 1, as they do in `graph.cuh`.
pub const PARAM_END: u32 = 0;

/// One recorded kernel argument.
///
/// `index` selects `kernelParams[index]`. `size` is the number of bytes that
/// belong to that argument. `key` is the id [`Graph::set_params`] matches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamSlot {
    /// Caller-chosen id. [`PARAM_END`] is the sentinel.
    pub key: u32,
    /// Index in the kernel's `kernelParams` array.
    pub index: usize,
    /// Bytes occupied by this argument.
    pub size: usize,
}

impl ParamSlot {
    /// An 8-byte device pointer argument.
    #[inline]
    pub const fn pointer(key: u32, index: usize) -> Self {
        Self {
            key,
            index,
            size: 8,
        }
    }

    /// A 4-byte scalar argument (`int` / `u32` / `f32` bits).
    #[inline]
    pub const fn u32_value(key: u32, index: usize) -> Self {
        Self {
            key,
            index,
            size: 4,
        }
    }

    /// A [`PARAM_END`] sentinel that closes one launch's group.
    ///
    /// [`Capture::record`] already groups slots by launch. The sentinel is for
    /// callers that walk keys the way `graph.cu` does, with the same id reused
    /// in the next launch.
    #[inline]
    pub const fn end() -> Self {
        Self {
            key: PARAM_END,
            index: 0,
            size: 1,
        }
    }
}

/// One update consumed by [`Graph::set_params`].
///
/// `bytes` is empty for a [`PARAM_END`] key. For every other key it must be
/// exactly the recorded size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamUpdate<'a> {
    /// Key recorded in a [`ParamSlot`].
    pub key: u32,
    /// Native-endian argument bytes.
    pub bytes: &'a [u8],
}

impl<'a> ParamUpdate<'a> {
    /// An update that writes `bytes` into the next site with `key`.
    #[inline]
    pub const fn new(key: u32, bytes: &'a [u8]) -> Self {
        Self { key, bytes }
    }
}

/// In-progress recording passed to the closure of [`Stream::capture`].
#[derive(Debug)]
pub struct Capture {
    ctx: Context,
    recordings: Vec<Recording>,
}

#[derive(Clone, Debug)]
struct Recording {
    function: Function,
    slots: Vec<ParamSlot>,
}

impl Capture {
    /// Context the capture runs on.
    #[inline]
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// Records the kernel launch that the caller is about to enqueue on the
    /// capturing stream. Returns the launch ordinal, starting at 0.
    ///
    /// Call this once per launch, immediately before [`Function::launch`], and
    /// in the same order the launches are enqueued. Two launches of one
    /// function are two calls. That replaces the fork's `GP_end` separator.
    /// A [`ParamSlot::end`] inside `slots` is still honored by
    /// [`Graph::set_params`].
    ///
    /// # Safety
    ///
    /// Each non-sentinel `slot.index` must be in range for `func`'s real
    /// parameter list, and `slot.size` must be that argument's size. A later
    /// [`Graph::set_params`] writes `size` bytes at `kernelParams[index]`.
    /// The next launch on the capturing stream must be `func`. Do not
    /// synchronize the capturing stream, and do not allocate or free device
    /// memory, between [`Stream::capture`]'s begin and end.
    pub unsafe fn record(&mut self, func: &Function, slots: &[ParamSlot]) -> Result<usize> {
        if !self.ctx.same(func.context()) {
            return Err(Error::ContextMismatch);
        }
        for slot in slots {
            validate_slot(*slot)?;
        }
        let launch = self.recordings.len();
        self.recordings.push(Recording {
            function: func.clone(),
            slots: slots.to_vec(),
        });
        Ok(launch)
    }
}

/// Ends a capture that the caller did not finish, so the stream is not left
/// capturing. The borrowed runtime stream does not own the driver's `CUstream`.
struct EndCapture {
    stream: Arc<RuntimeStream>,
    ctx: Context,
    active: bool,
}

impl Drop for EndCapture {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let _ = self.ctx.bind();
        if let Ok(graph) = unsafe { self.stream.end_capture() } {
            if !graph.is_null() {
                let _ = unsafe { cuda_core::sys::cuGraphDestroy(graph) };
            }
        }
    }
}

impl Stream {
    /// Captures work enqueued on this stream into a [`Graph`].
    ///
    /// The stream is synchronized first, then capture begins in thread-local
    /// mode (`CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`). This is not a device-wide
    /// synchronize: other streams are left alone. `body` must launch on
    /// `self`, must call [`Capture::record`] once before each of those
    /// launches, and must not synchronize `self`.
    ///
    /// On success the graph is instantiated and its kernel nodes are paired
    /// with the recordings. On failure, and if `body` panics, capture is ended
    /// and any graph the driver returned is destroyed so `self` can be used
    /// again.
    pub fn capture<F>(&self, body: F) -> Result<Graph>
    where
        F: FnOnce(&mut Capture) -> Result<()>,
    {
        // Thread-local capture rejects a stream that still has queued work.
        // The fork's `cudaDeviceSynchronize` also drained other streams; that
        // was a torch precaution, and this stream is the only one we record.
        self.synchronize()?;
        self.ctx.bind()?;
        let device = unsafe {
            RuntimeDevice::borrow_raw(
                self.ctx.inner.cu_ctx().cast(),
                device_as_c_int(self.ctx.inner.cu_device()),
                self.ctx.ordinal(),
            )
        };
        let runtime_stream = unsafe { RuntimeStream::borrow_raw(self.cu().cast(), &device) };
        unsafe {
            runtime_stream.begin_capture(
                cuda_core::sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        }
        .map_err(Error::driver)?;

        let mut guard = EndCapture {
            stream: runtime_stream,
            ctx: self.ctx.clone(),
            active: true,
        };
        let mut capture = Capture {
            ctx: self.ctx.clone(),
            recordings: Vec::new(),
        };
        body(&mut capture)?;

        let graph_handle = unsafe { guard.stream.end_capture() }.map_err(Error::driver)?;
        // `end_capture` succeeded, so a second end in `Drop` would run against
        // a stream that is no longer capturing. Clear the flag before anything
        // that can fail.
        guard.active = false;
        if graph_handle.is_null() {
            return Err(Error::CaptureInvalidated);
        }

        let mut graph = Graph {
            ctx: self.ctx.clone(),
            graph: graph_handle,
            exec: std::ptr::null_mut(),
            nodes: Vec::new(),
            sites: Vec::new(),
        };
        graph.finish(capture.recordings)?;
        Ok(graph)
    }
}

/// An instantiated graph.
///
/// Kernel arguments recorded during capture can be overwritten with
/// [`Self::set_params`] or [`Self::set_launch_param`] and then launched again.
/// The graph is not recaptured. Drop destroys the executable first, then the
/// graph. An in-flight launch is reclaimed by the driver when it finishes.
///
/// Not cloneable: the driver handles are owned once.
pub struct Graph {
    ctx: Context,
    graph: cuda_core::sys::CUgraph,
    exec: cuda_core::sys::CUgraphExec,
    nodes: Vec<KernelNode>,
    sites: Vec<Site>,
}

struct KernelNode {
    node: cuda_core::sys::CUgraphNode,
    params: cuda_core::sys::CUDA_KERNEL_NODE_PARAMS,
    dirty: bool,
    /// Keeps the module loaded. `params.func` is invalid after it unloads.
    _function: Function,
}

#[derive(Clone, Copy, Debug)]
struct Site {
    node: usize,
    launch: usize,
    key: u32,
    index: usize,
    size: usize,
}

// SAFETY: the driver handles are process-wide. Launch and parameter updates
// take `&mut self`, and every driver call binds the context first. `Context`
// and `Function` are already `Send`.
unsafe impl Send for Graph {}
// SAFETY: shared access only reads the context. Mutation of node storage goes
// through `&mut self`.
unsafe impl Sync for Graph {}

impl std::fmt::Debug for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Graph")
            .field("launches", &self.nodes.len())
            .field("sites", &self.sites.len())
            .finish_non_exhaustive()
    }
}

impl Graph {
    /// Context the graph was captured on.
    #[inline]
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// Recorded launches that were matched to kernel nodes.
    #[inline]
    pub fn launch_count(&self) -> usize {
        self.nodes.len()
    }

    /// Patches kernel arguments, then launches the executable on `stream`.
    ///
    /// The launch is asynchronous. `stream` must belong to this graph's
    /// context. Device memory named by the captured pointers must still be
    /// allocated.
    pub fn launch(&mut self, stream: &Stream) -> Result<()> {
        stream.same_context(&self.ctx)?;
        self.publish_dirty()?;
        self.ctx.bind()?;
        unsafe { cuda_core::sys::cuGraphLaunch(self.exec, stream.cu()) }
            .result()
            .map_err(Error::driver)
    }

    /// Writes `updates` into recorded sites.
    ///
    /// The walk is the one in `graph.cu`: for each update, skip sites until a
    /// key matches, then consume that update. [`PARAM_END`] matches and is not
    /// written. A matching site whose byte length differs from the recording
    /// is [`Error::ParamBytes`]. An update that runs out of sites is
    /// [`Error::GraphUpdate`]. Sites the updates never mention are left as
    /// captured.
    ///
    /// This does not launch. Call [`Self::launch`] after the values are in
    /// place. Bytes that already match the node storage do not mark the node
    /// dirty.
    pub fn set_params(&mut self, updates: &[ParamUpdate<'_>]) -> Result<()> {
        let writes = plan_updates(&self.sites, updates)?;
        for (site_index, update_index) in writes {
            let site = self.sites[site_index];
            write_param(
                &mut self.nodes[site.node],
                site.index,
                updates[update_index].bytes,
            )?;
        }
        Ok(())
    }

    /// Writes one argument of one recorded launch.
    ///
    /// `launch` is the ordinal [`Capture::record`] returned. `index` is the
    /// `kernelParams` index. Sentinel sites are not addressable here.
    pub fn set_launch_param(&mut self, launch: usize, index: usize, bytes: &[u8]) -> Result<()> {
        let site = lookup_launch_param(&self.sites, launch, index)?;
        if bytes.len() != site.size {
            return Err(Error::ParamBytes {
                key: site.key,
                expected: site.size,
                got: bytes.len(),
            });
        }
        write_param(&mut self.nodes[site.node], site.index, bytes)
    }

    fn finish(&mut self, recordings: Vec<Recording>) -> Result<()> {
        self.ctx.bind()?;
        let exec = unsafe {
            let mut exec = MaybeUninit::uninit();
            cuda_core::sys::cuGraphInstantiateWithFlags(exec.as_mut_ptr(), self.graph, 0)
                .result()
                .map_err(Error::driver)?;
            exec.assume_init()
        };
        self.exec = exec;

        let raw_nodes = graph_nodes(self.graph)?;
        let mut rec = 0;
        for node in raw_nodes {
            if rec == recordings.len() {
                break;
            }
            if !is_kernel_node(node)? {
                continue;
            }
            let params = read_kernel_params(node)?;
            let want = recordings[rec].function.raw_function();
            if params.func != want {
                continue;
            }
            let node_index = self.nodes.len();
            for slot in &recordings[rec].slots {
                self.sites.push(Site {
                    node: node_index,
                    launch: rec,
                    key: slot.key,
                    index: slot.index,
                    size: slot.size,
                });
            }
            self.nodes.push(KernelNode {
                node,
                params,
                dirty: false,
                _function: recordings[rec].function.clone(),
            });
            rec += 1;
        }
        if rec != recordings.len() {
            return Err(Error::GraphRecording {
                left: recordings.len() - rec,
            });
        }
        Ok(())
    }

    fn publish_dirty(&mut self) -> Result<()> {
        self.ctx.bind()?;
        for i in 0..self.nodes.len() {
            if !self.nodes[i].dirty {
                continue;
            }
            unsafe {
                cuda_core::sys::cuGraphExecKernelNodeSetParams_v2(
                    self.exec,
                    self.nodes[i].node,
                    &self.nodes[i].params,
                )
            }
            .result()
            .map_err(Error::driver)?;
            self.nodes[i].dirty = false;
        }
        Ok(())
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        let _ = self.ctx.bind();
        if !self.exec.is_null() {
            let _ = unsafe { cuda_core::sys::cuGraphExecDestroy(self.exec) };
            self.exec = std::ptr::null_mut();
        }
        if !self.graph.is_null() {
            let _ = unsafe { cuda_core::sys::cuGraphDestroy(self.graph) };
            self.graph = std::ptr::null_mut();
        }
    }
}

fn graph_nodes(graph: cuda_core::sys::CUgraph) -> Result<Vec<cuda_core::sys::CUgraphNode>> {
    let mut count: usize = 0;
    unsafe { cuda_core::sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut count) }
        .result()
        .map_err(Error::driver)?;
    let mut nodes: Vec<cuda_core::sys::CUgraphNode> = vec![std::ptr::null_mut(); count];
    unsafe { cuda_core::sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut count) }
        .result()
        .map_err(Error::driver)?;
    nodes.truncate(count);
    Ok(nodes)
}

fn is_kernel_node(node: cuda_core::sys::CUgraphNode) -> Result<bool> {
    let mut ty = MaybeUninit::uninit();
    unsafe { cuda_core::sys::cuGraphNodeGetType(node, ty.as_mut_ptr()) }
        .result()
        .map_err(Error::driver)?;
    let ty = unsafe { ty.assume_init() };
    Ok(ty == cuda_core::sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL)
}

fn read_kernel_params(
    node: cuda_core::sys::CUgraphNode,
) -> Result<cuda_core::sys::CUDA_KERNEL_NODE_PARAMS> {
    let mut params = MaybeUninit::<cuda_core::sys::CUDA_KERNEL_NODE_PARAMS>::zeroed();
    unsafe { cuda_core::sys::cuGraphKernelNodeGetParams_v2(node, params.as_mut_ptr()) }
        .result()
        .map_err(Error::driver)?;
    Ok(unsafe { params.assume_init() })
}

/// Copies `bytes` into the node-owned slot. Returns nothing; sets `dirty`
/// when the bytes differ. `index` came from a [`Capture::record`] whose caller
/// promised it was in range, and `bytes.len()` is the recorded size.
fn write_param(node: &mut KernelNode, index: usize, bytes: &[u8]) -> Result<()> {
    let storage = node.params.kernelParams;
    if storage.is_null() {
        return Err(Error::MissingParamStorage);
    }
    // SAFETY: `index` is a recorded kernel parameter. The driver allocated
    // `kernelParams` with one entry per argument and owns the slot. The graph
    // that owns the node is alive for as long as `node` is.
    let slot = unsafe { *storage.add(index) };
    if slot.is_null() {
        return Err(Error::MissingParamStorage);
    }
    let slot_bytes = slot.cast::<u8>();
    let changed = unsafe { std::slice::from_raw_parts(slot_bytes, bytes.len()) != bytes };
    if changed {
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), slot_bytes, bytes.len()) };
        node.dirty = true;
    }
    Ok(())
}

fn validate_slot(slot: ParamSlot) -> Result<()> {
    if slot.size == 0 || slot.size > MAX_PARAM_BYTES {
        return Err(Error::BadParamSize {
            key: slot.key,
            size: slot.size,
            max: MAX_PARAM_BYTES,
        });
    }
    if slot.index > MAX_PARAM_INDEX {
        return Err(Error::BadParamIndex {
            key: slot.key,
            index: slot.index,
            max: MAX_PARAM_INDEX,
        });
    }
    Ok(())
}

/// Pairs `(site_index, update_index)` that [`Graph::set_params`] writes.
/// [`PARAM_END`] matches are consumed and left out of the vector.
fn plan_updates(sites: &[Site], updates: &[ParamUpdate<'_>]) -> Result<Vec<(usize, usize)>> {
    let mut writes = Vec::new();
    let mut n = 0;
    for (p, update) in updates.iter().enumerate() {
        loop {
            if n == sites.len() {
                return Err(Error::GraphUpdate {
                    index: p,
                    key: update.key,
                });
            }
            let site = sites[n];
            n += 1;
            if site.key != update.key {
                continue;
            }
            if update.key != PARAM_END && update.bytes.len() != site.size {
                return Err(Error::ParamBytes {
                    key: site.key,
                    expected: site.size,
                    got: update.bytes.len(),
                });
            }
            if update.key != PARAM_END {
                writes.push((n - 1, p));
            }
            break;
        }
    }
    Ok(writes)
}

fn lookup_launch_param(sites: &[Site], launch: usize, index: usize) -> Result<Site> {
    sites
        .iter()
        .copied()
        .find(|site| site.launch == launch && site.index == index && site.key != PARAM_END)
        .ok_or(Error::NoSuchParam { launch, index })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn site(launch: usize, key: u32, index: usize, size: usize) -> Site {
        Site {
            node: launch,
            launch,
            key,
            index,
            size,
        }
    }

    fn mlp_sites() -> Vec<Site> {
        vec![
            site(0, 1, 0, 8),
            site(0, 2, 1, 4),
            site(0, PARAM_END, 0, 1),
            site(1, 1, 0, 8),
            site(1, 2, 1, 4),
            site(1, PARAM_END, 0, 1),
            site(2, 2, 1, 4),
        ]
    }

    #[test]
    fn param_end_is_the_fork_sentinel() {
        assert_eq!(PARAM_END, 0);
        assert_eq!(ParamSlot::end().key, PARAM_END);
        assert_eq!(ParamSlot::pointer(1, 0).size, 8);
        assert_eq!(ParamSlot::u32_value(2, 1).size, 4);
    }

    #[test]
    fn slot_limits_match_the_recorded_caps() {
        let err = validate_slot(ParamSlot {
            key: 3,
            index: 0,
            size: 0,
        })
        .unwrap_err();
        assert!(matches!(err, Error::BadParamSize { size: 0, .. }));

        let err = validate_slot(ParamSlot {
            key: 3,
            index: 0,
            size: MAX_PARAM_BYTES + 1,
        })
        .unwrap_err();
        assert!(matches!(
            err,
            Error::BadParamSize { size, .. } if size == MAX_PARAM_BYTES + 1
        ));

        let err = validate_slot(ParamSlot {
            key: 4,
            index: MAX_PARAM_INDEX + 1,
            size: 4,
        })
        .unwrap_err();
        assert!(matches!(err, Error::BadParamIndex { .. }));
        assert!(validate_slot(ParamSlot::end()).is_ok());
    }

    #[test]
    fn plan_skips_unmatched_sites_and_does_not_write_param_end() {
        let four = [0u8; 4];
        let updates = [
            ParamUpdate::new(2, &four),
            ParamUpdate::new(PARAM_END, &[]),
            ParamUpdate::new(2, &four),
        ];
        let writes = plan_updates(&mlp_sites(), &updates).unwrap();
        assert_eq!(writes, vec![(1, 0), (4, 2)]);
    }

    #[test]
    fn plan_can_reach_a_later_duplicate_key() {
        let four = [0u8; 4];
        let updates = [
            ParamUpdate::new(2, &four),
            ParamUpdate::new(PARAM_END, &[]),
            ParamUpdate::new(2, &four),
            ParamUpdate::new(PARAM_END, &[]),
            ParamUpdate::new(2, &four),
        ];
        let writes = plan_updates(&mlp_sites(), &updates).unwrap();
        assert_eq!(writes, vec![(1, 0), (4, 2), (6, 4)]);
    }

    #[test]
    fn plan_rejects_an_update_past_the_last_site() {
        let four = [0u8; 4];
        let updates = [ParamUpdate::new(9, &four)];
        let err = plan_updates(&mlp_sites(), &updates).unwrap_err();
        assert!(matches!(err, Error::GraphUpdate { key: 9, index: 0 }));
    }

    #[test]
    fn plan_rejects_a_byte_length_that_is_not_the_recorded_size() {
        let eight = [0u8; 8];
        let updates = [ParamUpdate::new(2, &eight)];
        let err = plan_updates(&mlp_sites(), &updates).unwrap_err();
        assert!(matches!(
            err,
            Error::ParamBytes {
                key: 2,
                expected: 4,
                got: 8
            }
        ));
    }

    #[test]
    fn launch_param_lookup_skips_the_sentinel() {
        let sites = mlp_sites();
        let found = lookup_launch_param(&sites, 1, 1).unwrap();
        assert_eq!(found.key, 2);
        assert_eq!(found.launch, 1);
        let err = lookup_launch_param(&sites, 1, 5).unwrap_err();
        assert!(matches!(
            err,
            Error::NoSuchParam {
                launch: 1,
                index: 5
            }
        ));
        let err = lookup_launch_param(&sites, 3, 0).unwrap_err();
        assert!(matches!(err, Error::NoSuchParam { launch: 3, .. }));
    }
}
