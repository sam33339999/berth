/// A launch was rejected or the CUDA runtime reported a failure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Host-side check failed, or the CUDA call returned an error.
    ///
    /// The message is the shim's, including the runtime's string when there is one.
    #[error("{0}")]
    Launch(String),
}

/// Result of a kernel launch.
pub type Result<T> = std::result::Result<T, Error>;
