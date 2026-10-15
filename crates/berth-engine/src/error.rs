/// Errors from loading weights or running a forward step.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Cuda(#[from] berth_cuda::Error),
    #[error(transparent)]
    Kernel(#[from] berth_kernels::Error),
    #[error(transparent)]
    Checkpoint(#[from] berth_exl3::Error),
    #[error("reading {name}: {source}")]
    Read {
        name: String,
        #[source]
        source: std::io::Error,
    },
    /// A tensor the plan needs is missing, or has a shape or dtype this
    /// engine does not handle.
    #[error("{0}")]
    Weights(String),
    /// Bad caller input: position past the cache, token past the vocabulary.
    #[error("{0}")]
    Input(String),
}

/// Result of an engine call.
pub type Result<T> = std::result::Result<T, Error>;
