//! `berth` — OpenAI-compatible HTTP in front of [`berth_engine::Engine`].

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use berth::{
    load_assets, serve, ServeConfig, DEFAULT_CACHE_TOKENS, DEFAULT_CHAT_MAX_TOKENS,
    DEFAULT_COMPLETION_MAX_TOKENS, DEFAULT_CUBINS, DEFAULT_DRY_ALLOWED_LENGTH, DEFAULT_DRY_BASE,
    DEFAULT_DRY_MULTIPLIER, DEFAULT_DRY_RANGE, DEFAULT_FREQ_PENALTY, DEFAULT_FREQ_RANGE,
    DEFAULT_HOST, DEFAULT_MAX_CONCURRENT, DEFAULT_MAX_MODEL_LEN, DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_NAME, DEFAULT_PORT, DEFAULT_PRESENCE_PENALTY, DEFAULT_REP_PENALTY,
};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "berth", about = "OpenAI-compatible HTTP server")]
struct Args {
    /// Model directory (tokenizer, chat template, weights).
    #[arg(long, env = "BERTH_MODEL", default_value = DEFAULT_MODEL_DIR)]
    model: PathBuf,

    /// Model id reported by `/v1/models` and in responses.
    #[arg(long, env = "BERTH_MODEL_NAME", default_value = DEFAULT_MODEL_NAME)]
    model_name: String,

    /// Bind address.
    #[arg(long, env = "BERTH_HOST", default_value = DEFAULT_HOST)]
    host: String,

    /// Bind port.
    #[arg(long, env = "BERTH_PORT", default_value_t = DEFAULT_PORT)]
    port: u16,

    /// KV cache length passed to `Engine::load`. Must be a positive multiple of 256.
    #[arg(
        long,
        alias = "max-tokens",
        env = "CACHE_TOKENS",
        default_value_t = DEFAULT_CACHE_TOKENS
    )]
    cache_tokens: u32,

    /// Per-request cap on prompt plus generated tokens. Must be <= `--cache-tokens`.
    #[arg(long, env = "CTX", default_value_t = DEFAULT_MAX_MODEL_LEN)]
    max_model_len: u32,

    /// Chat `max_tokens` when the client omits it.
    #[arg(long, env = "MAX_TOKENS", default_value_t = DEFAULT_CHAT_MAX_TOKENS)]
    default_max_tokens: u32,

    /// `/v1/completions` `max_tokens` when the client omits it.
    #[arg(
        long,
        env = "COMPLETION_MAX_TOKENS",
        default_value_t = DEFAULT_COMPLETION_MAX_TOKENS
    )]
    default_completion_max_tokens: u32,

    /// Requests admitted at once, including the one running and those queued.
    /// Above this, new requests get HTTP 429. 0 means unlimited. The GPU still
    /// runs one job at a time.
    #[arg(long, env = "MAX_CONCURRENT", default_value_t = DEFAULT_MAX_CONCURRENT)]
    max_concurrent: u32,

    /// Triton attention cubin directory.
    #[arg(long, env = "BERTH_CUBINS", default_value = DEFAULT_CUBINS)]
    cubins: PathBuf,

    /// DRY multiplier. 0 turns DRY off.
    #[arg(long, env = "DRY_MULTIPLIER", default_value_t = DEFAULT_DRY_MULTIPLIER)]
    dry_multiplier: f32,

    /// DRY base. Longer repeats are penalized faster as this grows.
    #[arg(long, env = "DRY_BASE", default_value_t = DEFAULT_DRY_BASE)]
    dry_base: f32,

    /// Repeats at or below this length are not penalized by DRY.
    #[arg(
        long,
        env = "DRY_ALLOWED_LENGTH",
        default_value_t = DEFAULT_DRY_ALLOWED_LENGTH
    )]
    dry_allowed_length: u32,

    /// DRY lookback in tokens. 0 means the whole context.
    #[arg(long, env = "DRY_RANGE", default_value_t = DEFAULT_DRY_RANGE)]
    dry_range: u32,

    /// Repetition penalty. 1.0 is off.
    #[arg(long, env = "REP_PENALTY", default_value_t = DEFAULT_REP_PENALTY)]
    rep_penalty: f32,

    /// Frequency penalty. 0 is off.
    #[arg(long, env = "FREQ_PENALTY", default_value_t = DEFAULT_FREQ_PENALTY)]
    freq_penalty: f32,

    /// Lookback, in tokens, for repetition, frequency, and presence penalties.
    #[arg(long, env = "FREQ_RANGE", default_value_t = DEFAULT_FREQ_RANGE)]
    freq_range: u32,

    /// Presence penalty. 0 is off.
    #[arg(
        long,
        env = "PRESENCE_PENALTY",
        default_value_t = DEFAULT_PRESENCE_PENALTY
    )]
    presence_penalty: f32,

    /// MTP draft tokens. 0 leaves the current decode path.
    /// 1..=15 loads the MTP head and widens GDN history.
    #[arg(long, env = "BERTH_MTP_STEPS", default_value_t = 0)]
    mtp_steps: u32,
}

fn main() {
    let args = Args::parse();
    if let Err(message) = check_config(&args) {
        eprintln!("berth: {message}");
        std::process::exit(1);
    }
    match port_in_use(args.port) {
        Ok(true) => {
            eprintln!("berth: port {} is in use; not loading the model", args.port);
            std::process::exit(1);
        }
        Err(err) => {
            eprintln!("berth: {err}; not loading the model");
            std::process::exit(1);
        }
        Ok(false) => {}
    }
    eprintln!(
        "berth: model {} name {} cache-tokens {} max-model-len {} max-concurrent {} {}:{}",
        args.model.display(),
        args.model_name,
        args.cache_tokens,
        args.max_model_len,
        args.max_concurrent,
        args.host,
        args.port,
    );

    let engine = match berth_engine::Engine::load(
        &args.model,
        &args.cubins,
        args.cache_tokens,
        args.mtp_steps,
    ) {
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
            model_name: args.model_name,
            max_model_len: args.max_model_len,
            max_concurrent: args.max_concurrent,
            chat_max_tokens: args.default_max_tokens,
            completion_max_tokens: args.default_completion_max_tokens,
            rep_penalty: args.rep_penalty,
            presence_penalty: args.presence_penalty,
            frequency_penalty: args.freq_penalty,
            freq_range: args.freq_range,
            dry_multiplier: args.dry_multiplier,
            dry_base: args.dry_base,
            dry_allowed_length: args.dry_allowed_length,
            dry_range: args.dry_range,
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

/// `cache-tokens` is a page multiple; `max-model-len` has to fit in that cache.
fn check_config(args: &Args) -> Result<(), String> {
    if args.cache_tokens == 0 || !args.cache_tokens.is_multiple_of(256) {
        return Err(format!(
            "cache-tokens must be a positive multiple of 256 (got {})",
            args.cache_tokens
        ));
    }
    if args.max_model_len > args.cache_tokens {
        return Err(format!(
            "max-model-len ({}) is larger than cache-tokens ({})",
            args.max_model_len, args.cache_tokens
        ));
    }
    if args.mtp_steps > 15 {
        return Err(format!("mtp-steps must be 0..=15 (got {})", args.mtp_steps));
    }
    Ok(())
}

/// `ss -ltn` contains `:{port} `. The live serve owns 8889 and the GPU.
fn port_in_use(port: u16) -> std::io::Result<bool> {
    let output = Command::new("ss").args(["-ltn"]).output()?;
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "ss exited with {}",
            output.status
        )));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    Ok(text.contains(&format!(":{port} ")))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use clap::{CommandFactory, Parser};

    use super::{check_config, Args};
    use berth::{
        DEFAULT_CACHE_TOKENS, DEFAULT_CHAT_MAX_TOKENS, DEFAULT_COMPLETION_MAX_TOKENS,
        DEFAULT_CUBINS, DEFAULT_DRY_ALLOWED_LENGTH, DEFAULT_DRY_BASE, DEFAULT_DRY_MULTIPLIER,
        DEFAULT_DRY_RANGE, DEFAULT_FREQ_PENALTY, DEFAULT_FREQ_RANGE, DEFAULT_HOST,
        DEFAULT_MAX_CONCURRENT, DEFAULT_MAX_MODEL_LEN, DEFAULT_MODEL_DIR, DEFAULT_MODEL_NAME,
        DEFAULT_PORT, DEFAULT_PRESENCE_PENALTY, DEFAULT_REP_PENALTY,
    };

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    const ENV_KEYS: &[&str] = &[
        "BERTH_MODEL",
        "BERTH_MODEL_NAME",
        "BERTH_HOST",
        "BERTH_PORT",
        "CACHE_TOKENS",
        "CTX",
        "MAX_TOKENS",
        "COMPLETION_MAX_TOKENS",
        "MAX_CONCURRENT",
        "BERTH_CUBINS",
        "DRY_MULTIPLIER",
        "DRY_BASE",
        "DRY_ALLOWED_LENGTH",
        "DRY_RANGE",
        "REP_PENALTY",
        "FREQ_PENALTY",
        "FREQ_RANGE",
        "PRESENCE_PENALTY",
        "BERTH_MTP_STEPS",
    ];

    struct RestoreEnv {
        saved: Vec<(&'static str, Option<String>)>,
    }

    impl Drop for RestoreEnv {
        fn drop(&mut self) {
            for (key, value) in self.saved.drain(..) {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }

    fn with_env<T>(pairs: &[(&str, &str)], body: impl FnOnce() -> T) -> T {
        let _lock = ENV_LOCK.lock().expect("env lock");
        let saved = ENV_KEYS
            .iter()
            .copied()
            .map(|key| (key, std::env::var(key).ok()))
            .collect();
        let _restore = RestoreEnv { saved };
        for key in ENV_KEYS {
            std::env::remove_var(key);
        }
        for (key, value) in pairs {
            std::env::set_var(key, value);
        }
        body()
    }

    fn parse(args: &[&str]) -> Args {
        Args::try_parse_from(args).expect("parse")
    }

    #[test]
    fn defaults_match_serve_host_when_env_is_unset() {
        with_env(&[], || {
            let args = parse(&["berth"]);
            assert_eq!(args.model, std::path::PathBuf::from(DEFAULT_MODEL_DIR));
            assert_eq!(args.model_name, DEFAULT_MODEL_NAME);
            assert_eq!(args.host, DEFAULT_HOST);
            assert_eq!(args.port, DEFAULT_PORT);
            assert_eq!(args.cache_tokens, DEFAULT_CACHE_TOKENS);
            assert_eq!(args.max_model_len, DEFAULT_MAX_MODEL_LEN);
            assert_eq!(args.default_max_tokens, DEFAULT_CHAT_MAX_TOKENS);
            assert_eq!(
                args.default_completion_max_tokens,
                DEFAULT_COMPLETION_MAX_TOKENS
            );
            assert_eq!(args.max_concurrent, DEFAULT_MAX_CONCURRENT);
            assert_eq!(args.cubins, std::path::PathBuf::from(DEFAULT_CUBINS));
            assert_eq!(args.dry_multiplier, DEFAULT_DRY_MULTIPLIER);
            assert_eq!(args.dry_base, DEFAULT_DRY_BASE);
            assert_eq!(args.dry_allowed_length, DEFAULT_DRY_ALLOWED_LENGTH);
            assert_eq!(args.dry_range, DEFAULT_DRY_RANGE);
            assert_eq!(args.rep_penalty, DEFAULT_REP_PENALTY);
            assert_eq!(args.freq_penalty, DEFAULT_FREQ_PENALTY);
            assert_eq!(args.freq_range, DEFAULT_FREQ_RANGE);
            assert_eq!(args.presence_penalty, DEFAULT_PRESENCE_PENALTY);
            assert_eq!(args.mtp_steps, 0);
            assert!(check_config(&args).is_ok());
        });
    }

    #[test]
    fn max_tokens_alias_env_and_cli_override() {
        with_env(
            &[
                ("CACHE_TOKENS", "1024"),
                ("CTX", "768"),
                ("BERTH_MODEL_NAME", "from-env"),
                ("MAX_CONCURRENT", "4"),
                ("DRY_MULTIPLIER", "0.4"),
                ("DRY_BASE", "1.25"),
                ("DRY_ALLOWED_LENGTH", "2"),
                ("DRY_RANGE", "128"),
                ("REP_PENALTY", "1.1"),
                ("FREQ_PENALTY", "0.2"),
                ("FREQ_RANGE", "64"),
                ("PRESENCE_PENALTY", "0.3"),
                ("MAX_TOKENS", "100"),
                ("COMPLETION_MAX_TOKENS", "16"),
                ("BERTH_HOST", "127.0.0.1"),
                ("BERTH_PORT", "18889"),
                ("BERTH_CUBINS", "cubins-from-env"),
                ("BERTH_MODEL", "/models/from-env"),
            ],
            || {
                let args = parse(&["berth"]);
                assert_eq!(args.cache_tokens, 1024);
                assert_eq!(args.max_model_len, 768);
                assert_eq!(args.model_name, "from-env");
                assert_eq!(args.max_concurrent, 4);
                assert_eq!(args.dry_multiplier, 0.4);
                assert_eq!(args.dry_base, 1.25);
                assert_eq!(args.dry_allowed_length, 2);
                assert_eq!(args.dry_range, 128);
                assert_eq!(args.rep_penalty, 1.1);
                assert_eq!(args.freq_penalty, 0.2);
                assert_eq!(args.freq_range, 64);
                assert_eq!(args.presence_penalty, 0.3);
                assert_eq!(args.default_max_tokens, 100);
                assert_eq!(args.default_completion_max_tokens, 16);
                assert_eq!(args.host, "127.0.0.1");
                assert_eq!(args.port, 18889);
                assert_eq!(args.cubins, std::path::PathBuf::from("cubins-from-env"));
                assert_eq!(args.model, std::path::PathBuf::from("/models/from-env"));
                assert!(check_config(&args).is_ok());

                let alias = parse(&["berth", "--max-tokens", "2048", "--max-model-len", "2048"]);
                assert_eq!(alias.cache_tokens, 2048);
                assert_eq!(alias.max_model_len, 2048);
                assert_eq!(alias.model_name, "from-env");
                assert!(check_config(&alias).is_ok());

                let over = parse(&[
                    "berth",
                    "--cache-tokens",
                    "512",
                    "--max-model-len",
                    "256",
                    "--model-name",
                    "from-cli",
                ]);
                assert_eq!(over.cache_tokens, 512);
                assert_eq!(over.max_model_len, 256);
                assert_eq!(over.model_name, "from-cli");
                assert!(check_config(&over).is_ok());
            },
        );
    }

    #[test]
    fn rejects_a_bad_cache_length_and_a_context_past_the_cache() {
        with_env(&[], || {
            let zero = parse(&["berth", "--cache-tokens", "0"]);
            let err = check_config(&zero).unwrap_err();
            assert!(err.contains("positive multiple of 256"), "{err}");

            let odd = parse(&["berth", "--cache-tokens", "1000", "--max-model-len", "1000"]);
            let err = check_config(&odd).unwrap_err();
            assert!(err.contains("positive multiple of 256"), "{err}");
            assert!(err.contains("1000"), "{err}");

            let over = parse(&["berth", "--cache-tokens", "512", "--max-model-len", "513"]);
            let err = check_config(&over).unwrap_err();
            assert!(err.contains("513"), "{err}");
            assert!(err.contains("512"), "{err}");
            assert!(err.contains("larger than cache-tokens"), "{err}");
        });
        with_env(&[("CACHE_TOKENS", "300"), ("CTX", "256")], || {
            let err = check_config(&parse(&["berth"])).unwrap_err();
            assert!(err.contains("positive multiple of 256"), "{err}");
        });
    }

    #[test]
    fn help_lists_every_flag_with_its_env_and_default() {
        let mut command = Args::command();
        let help = command.render_help().to_string();
        let required = [
            "--model",
            "BERTH_MODEL",
            DEFAULT_MODEL_DIR,
            "--model-name",
            "BERTH_MODEL_NAME",
            DEFAULT_MODEL_NAME,
            "--host",
            "BERTH_HOST",
            DEFAULT_HOST,
            "--port",
            "BERTH_PORT",
            "--cache-tokens",
            "CACHE_TOKENS",
            "270336",
            "--max-model-len",
            "CTX",
            "262144",
            "--default-max-tokens",
            "MAX_TOKENS",
            "32768",
            "--default-completion-max-tokens",
            "COMPLETION_MAX_TOKENS",
            "--max-concurrent",
            "MAX_CONCURRENT",
            "--cubins",
            "BERTH_CUBINS",
            DEFAULT_CUBINS,
            "--dry-multiplier",
            "DRY_MULTIPLIER",
            "--dry-base",
            "DRY_BASE",
            "1.75",
            "--dry-allowed-length",
            "DRY_ALLOWED_LENGTH",
            "--dry-range",
            "DRY_RANGE",
            "4096",
            "--rep-penalty",
            "REP_PENALTY",
            "--freq-penalty",
            "FREQ_PENALTY",
            "--freq-range",
            "FREQ_RANGE",
            "--presence-penalty",
            "PRESENCE_PENALTY",
            "--mtp-steps",
            "BERTH_MTP_STEPS",
            "[env:",
            "[default:",
        ];
        for needle in required {
            assert!(help.contains(needle), "{needle} missing from:\n{help}");
        }
        assert!(
            !help.contains("--max-tokens"),
            "hidden alias leaked into help:\n{help}"
        );
    }

    #[test]
    fn mtp_steps_parses_0_4_and_15() {
        with_env(&[], || {
            let zero = parse(&["berth", "--mtp-steps", "0"]);
            let four = parse(&["berth", "--mtp-steps", "4"]);
            let fifteen = parse(&["berth", "--mtp-steps", "15"]);
            assert_eq!(zero.mtp_steps, 0);
            assert_eq!(four.mtp_steps, 4);
            assert_eq!(fifteen.mtp_steps, 15);
            assert!(check_config(&zero).is_ok());
            assert!(check_config(&four).is_ok());
            assert!(check_config(&fifteen).is_ok());
        });
        with_env(&[("BERTH_MTP_STEPS", "4")], || {
            let args = parse(&["berth"]);
            assert_eq!(args.mtp_steps, 4);
            assert!(check_config(&args).is_ok());
        });
    }

    #[test]
    fn mtp_steps_rejects_16() {
        with_env(&[], || {
            let args = parse(&["berth", "--mtp-steps", "16"]);
            let err = check_config(&args).unwrap_err();
            assert!(err.contains("0..=15"), "{err}");
            assert!(err.contains("16"), "{err}");
        });
    }
}
