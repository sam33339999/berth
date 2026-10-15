//! `berth` — OpenAI-compatible HTTP in front of [`berth_engine::Engine`].

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use berth::{
    load_assets, serve, ServeConfig, DEFAULT_CACHE_TOKENS, DEFAULT_CHAT_MAX_TOKENS,
    DEFAULT_COMPLETION_MAX_TOKENS, DEFAULT_CUBINS, DEFAULT_HOST, DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_NAME, DEFAULT_PORT,
};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "berth", about = "OpenAI-compatible HTTP server")]
struct Args {
    /// Model directory (tokenizer, chat template, weights).
    #[arg(long, default_value = DEFAULT_MODEL_DIR)]
    model: PathBuf,

    /// Bind address. `scripts/serve-host.sh` uses 0.0.0.0.
    #[arg(long, default_value = DEFAULT_HOST)]
    host: String,

    /// Bind port. Tests use an ephemeral port and do not bind this one.
    #[arg(long, default_value_t = DEFAULT_PORT)]
    port: u16,

    /// KV cache length passed to `Engine::load` (`CACHE_TOKENS`).
    #[arg(long, default_value_t = DEFAULT_CACHE_TOKENS)]
    max_tokens: u32,

    /// Triton attention cubin directory.
    #[arg(long, default_value = DEFAULT_CUBINS)]
    cubins: PathBuf,
}

fn main() {
    let args = Args::parse();
    match port_8889_listening() {
        Ok(true) => {
            eprintln!("berth: port 8889 is in use; not loading the model");
            std::process::exit(1);
        }
        Err(err) => {
            eprintln!("berth: {err}; not loading the model");
            std::process::exit(1);
        }
        Ok(false) => {}
    }

    let engine = match berth_engine::Engine::load(&args.model, &args.cubins, args.max_tokens) {
        Ok(engine) => engine,
        Err(err) => {
            eprintln!("berth: {err}");
            std::process::exit(1);
        }
    };
    let assets = match load_assets(&args.model) {
        Ok(assets) => assets,
        Err(err) => {
            eprintln!("berth: {err}");
            std::process::exit(1);
        }
    };
    let handle = match serve(
        Arc::new(assets),
        Box::new(engine),
        ServeConfig {
            host: args.host,
            port: args.port,
            model_name: DEFAULT_MODEL_NAME.to_string(),
            chat_max_tokens: DEFAULT_CHAT_MAX_TOKENS,
            completion_max_tokens: DEFAULT_COMPLETION_MAX_TOKENS,
        },
    ) {
        Ok(handle) => handle,
        Err(err) => {
            eprintln!("berth: {err}");
            std::process::exit(1);
        }
    };
    eprintln!("berth: listening on {}", handle.addr());
    std::thread::park();
}

/// `ss -ltn | grep -q ':8889 '`. The live serve owns that port and the GPU.
fn port_8889_listening() -> std::io::Result<bool> {
    let output = Command::new("ss").args(["-ltn"]).output()?;
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "ss exited with {}",
            output.status
        )));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    Ok(text.contains(":8889 "))
}
