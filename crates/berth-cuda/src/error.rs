//! Driver failures and the checks this crate adds on top of them.
//!
//! [`Error`] is the only error type that leaves the crate. `cuda_core::DriverError`
//! is formatted into [`Error::Driver`] at the boundary.

/// Failure from the CUDA driver, or from a check this crate made before
/// calling the driver.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A driver call returned something other than `CUDA_SUCCESS`.
    ///
    /// `code` is the numeric `CUresult`. `detail` is the driver's own
    /// description of that code, when the driver could produce one.
    #[error("CUDA driver error {code}: {detail}")]
    Driver {
        /// Numeric `CUresult`.
        code: u32,
        /// Driver description, including the symbolic name when available.
        detail: String,
    },

    /// A kernel or symbol name contained an interior `NUL`.
    #[error("name contains an interior nul")]
    InteriorNul,

    /// `cuModuleLoad` takes a UTF-8 path.
    #[error("path is not valid UTF-8: {0}")]
    PathNotUtf8(String),

    /// `load_bytes` was given an empty slice.
    #[error("cubin image is empty")]
    EmptyImage,

    /// A host slice and a device buffer disagree on element count.
    #[error("host length {host} does not match device length {device}")]
    Length {
        /// Elements on the host.
        host: usize,
        /// Elements on the device.
        device: usize,
    },

    /// `len * size_of::<T>()` overflowed `usize`.
    #[error("allocation of {len} elements of {elem} bytes overflows")]
    TooLarge {
        /// Requested element count.
        len: usize,
        /// `size_of` of the element type.
        elem: usize,
    },

    /// A recorded parameter is zero bytes or larger than [`super::MAX_PARAM_BYTES`].
    #[error("parameter {key} has size {size}; expected 1..={max} bytes")]
    BadParamSize {
        /// Caller key from [`super::ParamSlot`].
        key: u32,
        /// Rejected size.
        size: usize,
        /// [`super::MAX_PARAM_BYTES`].
        max: usize,
    },

    /// A recorded parameter index is above [`super::MAX_PARAM_INDEX`].
    #[error("parameter {key} index {index} is above {max}")]
    BadParamIndex {
        /// Caller key from [`super::ParamSlot`].
        key: u32,
        /// Rejected index.
        index: usize,
        /// [`super::MAX_PARAM_INDEX`].
        max: usize,
    },

    /// Capture ended with recorded launches that matched no kernel node.
    #[error("graph recording left {left} launch(es) unmatched")]
    GraphRecording {
        /// Recorded launches that never paired with a kernel node.
        left: usize,
    },

    /// `cuStreamEndCapture` returned a null graph.
    ///
    /// The capture was invalidated. The usual cause is synchronizing the
    /// capturing stream, or another driver call the capture rules forbid.
    #[error("stream capture produced no graph")]
    CaptureInvalidated,

    /// [`super::Graph::set_params`] ran out of recorded sites before it ran
    /// out of updates.
    #[error("graph update {index} (key {key}) matched no remaining recorded site")]
    GraphUpdate {
        /// Index of the update in the slice passed to `set_params`.
        index: usize,
        /// Key that could not be matched.
        key: u32,
    },

    /// An update's byte length is not the size recorded for that site.
    #[error("graph parameter {key} is {expected} bytes, got {got}")]
    ParamBytes {
        /// Site key.
        key: u32,
        /// Size recorded at capture.
        expected: usize,
        /// Length of the update slice.
        got: usize,
    },

    /// [`super::Graph::set_launch_param`] named a launch or argument that was
    /// not recorded.
    #[error("recorded launch {launch} has no parameter at index {index}")]
    NoSuchParam {
        /// Launch ordinal from [`super::Capture::record`].
        launch: usize,
        /// Kernel parameter index.
        index: usize,
    },

    /// The captured node has no `kernelParams` array to write into.
    #[error("captured kernel node has no parameter storage")]
    MissingParamStorage,

    /// A stream, buffer, and graph were mixed across two CUDA contexts.
    #[error("CUDA objects are not on the same context")]
    ContextMismatch,

    /// Dependency edges from the capture did not produce one total order.
    #[error("captured graph dependencies are not a single order")]
    GraphOrder,

    /// [`super::Graph::set_node_param`] named a kernel ordinal this graph
    /// does not have.
    #[error("graph has no kernel node {index}")]
    NoSuchKernelNode {
        /// Index from [`super::KernelNodeInfo::index`].
        index: usize,
    },

    /// The parameter index is outside the list `cuFuncGetParamInfo` reported.
    #[error("kernel node {node} has no parameter at index {index}")]
    NoSuchKernelParam {
        /// Kernel-node ordinal.
        node: usize,
        /// Rejected `kernelParams` index.
        index: usize,
    },

    /// The byte slice is not the size `cuFuncGetParamInfo` reported.
    #[error("kernel node {node} parameter {index} is {expected} bytes, got {got}")]
    NodeParamBytes {
        /// Kernel-node ordinal.
        node: usize,
        /// `kernelParams` index.
        index: usize,
        /// Size reported by the driver.
        expected: usize,
        /// Length of the slice passed to `set_node_param`.
        got: usize,
    },

    /// The loaded driver does not export `cuFuncGetParamInfo` (CUDA 12.4).
    #[error("driver does not export cuFuncGetParamInfo (CUDA 12.4)")]
    ParamInfoUnavailable,

    /// The loaded driver does not export `cuFuncGetName` (CUDA 12.3).
    #[error("driver does not export cuFuncGetName (CUDA 12.3)")]
    FunctionNameUnavailable,

    /// A kernel node had neither a `CUfunction` nor a resolvable `CUkernel`.
    #[error("captured kernel node has no function handle")]
    MissingKernelFunction,

    /// `cuFuncGetParamInfo` did not end within [`super::MAX_PARAM_INDEX`].
    #[error("kernel parameter list is longer than {max}")]
    TooManyParams {
        /// [`super::MAX_PARAM_INDEX`].
        max: usize,
    },

    /// Dynamic shared-memory opt-in did not fit in the driver's `int` attribute.
    #[error("dynamic shared memory of {bytes} bytes does not fit in a driver attribute")]
    SharedMemory {
        /// Requested byte count.
        bytes: u32,
    },
}

impl Error {
    pub(crate) fn driver(err: cuda_core::DriverError) -> Self {
        Self::Driver {
            code: driver_code(err),
            detail: err.to_string(),
        }
    }
}

fn driver_code(err: cuda_core::DriverError) -> u32 {
    // `CUresult` is an integer enum. The cast is a no-op when bindgen already
    // emitted it as `u32`, and it is the conversion when bindgen emitted a
    // distinct enum type.
    #[allow(clippy::unnecessary_cast)]
    let code = err.0 as u32;
    code
}

/// Result of a `berth-cuda` call.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driver_error_shows_code_and_detail() {
        let err = Error::Driver {
            code: 1,
            detail: "invalid value".to_string(),
        };
        let text = err.to_string();
        assert!(text.contains('1'), "{text}");
        assert!(text.contains("invalid value"), "{text}");
    }

    #[test]
    fn length_error_names_both_sides() {
        let err = Error::Length { host: 2, device: 4 };
        let text = err.to_string();
        assert!(text.contains('2') && text.contains('4'), "{text}");
    }
}
