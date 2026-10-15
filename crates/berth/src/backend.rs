//! What the HTTP layer needs from a model. [`berth_engine::Engine`] is the
//! real one; tests use a scripted fake so the server is testable without a GPU.

use berth_engine::{Finish, Flow, GenerateRequest, Usage};

/// One request at a time; the server serializes calls.
pub trait Backend: Send {
    fn vocab_size(&self) -> u32;
    fn generate(
        &mut self,
        request: &GenerateRequest,
        on_token: &mut dyn FnMut(u32) -> Flow,
    ) -> berth_engine::Result<(Finish, Usage)>;
}

impl Backend for berth_engine::Engine {
    fn vocab_size(&self) -> u32 {
        berth_engine::Engine::vocab_size(self)
    }
    fn generate(
        &mut self,
        request: &GenerateRequest,
        on_token: &mut dyn FnMut(u32) -> Flow,
    ) -> berth_engine::Result<(Finish, Usage)> {
        berth_engine::Engine::generate(self, request, on_token)
    }
}
