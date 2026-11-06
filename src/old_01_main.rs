mod tokenizer;

use anyhow::{anyhow, bail};
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama as model;
use candle_transformers::utils::apply_repeat_penalty;
use clap::{Args, Parser, Subcommand};
use futures_util::stream::Stream;
use model::ModelWeights;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fmt::Display;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokenizer::Tok;
use tokio::sync::mpsc;

const DEFAULT_HOST: &str = "127.0.0.1:11435";
const DEFAULT_MODELS_DIR: &str = "~/.djeli/models";
const DEFAULT_MAX_TOKENS: usize = 150;
/// The OpenAI API has no small default, so its endpoint allows longer replies.
const OPENAI_DEFAULT_MAX_TOKENS: usize = 1024;
/// candle-transformers 0.4.x quantized llama supports at most 4096 positions.
const MAX_CONTEXT: usize = 4096;
/// Token strings that end generation, if the tokenizer knows them.
const STOP_TOKEN_CANDIDATES: [&str; 4] = ["</s>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>"];

// ===========================================================================
// Command line
// ===========================================================================

#[derive(Parser)]
#[command(
    name = "djeli",
    version,
    about = "Djeli: run GGUF language models locally and offline \
             (Ollama-style CLI + OpenAI-compatible API)"
)]
struct Cli {
    /// Server address. Default: "host" from the config file, else 127.0.0.1:11435
    #[arg(long, global = true, env = "DJELI_HOST")]
    host: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the inference server
    Serve {
        /// Folder with your .gguf files. Default: "models_dir" from the config file
        #[arg(long, env = "DJELI_MODELS")]
        models_dir: Option<String>,
    },
    /// Run a model: one-shot with a prompt, or an interactive chat without one
    Run(RunArgs),
    /// List the models available on the server
    #[command(visible_alias = "ls")]
    List,
    /// Unload models from the server's memory (all of them, or just one)
    Stop {
        /// Model to unload; omit it to unload every model
        model: Option<String>,
    },
    /// Show the config file (created on first use), where the models folder is set
    Config,
}

#[derive(Args, Clone)]
struct RunArgs {
    /// Model name, i.e. the file name in the models folder without ".gguf"
    model: String,

    /// Prompt. Omit it to start an interactive chat (or to read it from a pipe)
    prompt: Vec<String>,

    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Nucleus sampling threshold
    #[arg(long)]
    top_p: Option<f64>,

    /// Fixed seed for reproducible output
    #[arg(long)]
    seed: Option<u64>,

    /// Maximum number of generated tokens per reply
    #[arg(long)]
    max_tokens: Option<usize>,

    /// System prompt
    #[arg(long)]
    system: Option<String>,

    /// Chat template: zephyr (default), chatml, llama2 or raw
    #[arg(long)]
    template: Option<String>,

    /// Repeat penalty (1.0 disables it)
    #[arg(long)]
    repeat_penalty: Option<f32>,

    /// Previous exchanges remembered in chat mode (0 = every message starts fresh).
    /// Small models answer better with a short memory.
    #[arg(long, default_value_t = 4)]
    history: usize,

    /// Print timing statistics after each reply
    #[arg(long)]
    verbose: bool,
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}

fn real_main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let cfg_path = config_path();
    let cfg = load_config(&cfg_path)?;

    match cli.command {
        Commands::Serve { models_dir } => {
            if let Err(e) = ensure_config_file(&cfg_path) {
                eprintln!("warning: could not create {}: {e}", cfg_path.display());
            }
            let host = resolve_host(cli.host, &cfg);
            let models_dir = resolve_models_dir(models_dir, &cfg);
            tracing_subscriber::fmt::init();
            let runtime = tokio::runtime::Runtime::new()?;
            runtime.block_on(serve(host, models_dir, cfg_path))
        }
        Commands::Run(args) => cmd_run(&resolve_host(cli.host, &cfg), args),
        Commands::List => cmd_list(&resolve_host(cli.host, &cfg)),
        Commands::Stop { model } => cmd_stop(&resolve_host(cli.host, &cfg), model),
        Commands::Config => cmd_config(cli.host),
    }
}

// ===========================================================================
// Config file
// ===========================================================================

/// Settings read from `config.json`. Every field is optional.
#[derive(Serialize, Deserialize, Default, Clone)]
struct Config {
    /// Folder that holds your .gguf model files ("~" is expanded).
    models_dir: Option<String>,
    /// Address the server listens on and the CLI connects to.
    host: Option<String>,
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// `DJELI_CONFIG`, else `~/.config/djeli/config.json` (`%APPDATA%\djeli` on Windows).
fn config_path() -> PathBuf {
    if let Some(p) = std::env::var_os("DJELI_CONFIG") {
        return PathBuf::from(p);
    }
    let base = if cfg!(windows) {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| home_dir().map(|h| h.join(".config")))
    };
    base.unwrap_or_else(|| PathBuf::from("."))
        .join("djeli")
        .join("config.json")
}

fn load_config(path: &Path) -> anyhow::Result<Config> {
    if !path.exists() {
        return Ok(Config::default());
    }
    let text = fs::read_to_string(path)
        .map_err(|e| anyhow!("cannot read config file {}: {e}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|e| anyhow!("invalid JSON in config file {}: {e}", path.display()))
}

/// Creates the config file with default values if it does not exist yet.
/// Returns true when the file was created.
fn ensure_config_file(path: &Path) -> anyhow::Result<bool> {
    if path.exists() {
        return Ok(false);
    }
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let defaults = json!({
        "models_dir": DEFAULT_MODELS_DIR,
        "host": DEFAULT_HOST,
    });
    fs::write(path, serde_json::to_string_pretty(&defaults)? + "\n")?;
    Ok(true)
}

fn expand_tilde(p: &str) -> PathBuf {
    if p == "~" {
        return home_dir().unwrap_or_else(|| PathBuf::from(p));
    }
    if let Some(rest) = p.strip_prefix("~/").or_else(|| p.strip_prefix("~\\")) {
        if let Some(home) = home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(p)
}

/// Priority: command-line flag / environment variable, then config file, then default.
fn resolve_host(flag: Option<String>, cfg: &Config) -> String {
    flag.or_else(|| cfg.host.clone())
        .unwrap_or_else(|| DEFAULT_HOST.to_string())
}

fn resolve_models_dir(flag: Option<String>, cfg: &Config) -> PathBuf {
    let raw = flag
        .or_else(|| cfg.models_dir.clone())
        .unwrap_or_else(|| DEFAULT_MODELS_DIR.to_string());
    expand_tilde(&raw)
}

fn cmd_config(host_flag: Option<String>) -> anyhow::Result<()> {
    let path = config_path();
    let created = ensure_config_file(&path)?;
    let cfg = load_config(&path)?;
    let raw = fs::read_to_string(&path)?;

    println!(
        "Config file: {}{}",
        path.display(),
        if created { "  (just created)" } else { "" }
    );
    println!("{}", raw.trim_end());

    let host = resolve_host(host_flag, &cfg);
    let models_dir = resolve_models_dir(std::env::var("DJELI_MODELS").ok(), &cfg);
    println!("\nEffective settings:");
    println!("  host:       {host}");
    println!(
        "  models_dir: {}{}",
        models_dir.display(),
        if models_dir.is_dir() { "" } else { "  (does not exist yet)" }
    );
    println!("\nEdit the file, then restart `djeli serve` to apply the changes.");
    println!("Overrides: --host / --models-dir, DJELI_HOST / DJELI_MODELS, DJELI_CONFIG (file path).");
    Ok(())
}

// ===========================================================================
// CLI client
// ===========================================================================

#[derive(Serialize, Deserialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

fn msg(role: &str, content: impl Into<String>) -> ChatMessage {
    ChatMessage {
        role: role.to_string(),
        content: content.into(),
    }
}

fn base_url(host: &str) -> String {
    let host = host.trim_end_matches('/');
    if host.starts_with("http://") || host.starts_with("https://") {
        host.to_string()
    } else {
        format!("http://{host}")
    }
}

fn http_error(base: &str, e: ureq::Error) -> anyhow::Error {
    match e {
        ureq::Error::Status(code, resp) => {
            anyhow!("server returned HTTP {code}: {}", resp.into_string().unwrap_or_default())
        }
        ureq::Error::Transport(t) => {
            anyhow!("could not reach the djeli server at {base} ({t}). Start it with: djeli serve")
        }
    }
}

fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1000.0 && unit < UNITS.len() - 1 {
        value /= 1000.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn cmd_list(host: &str) -> anyhow::Result<()> {
    let base = base_url(host);
    let resp = ureq::get(&format!("{base}/api/tags"))
        .call()
        .map_err(|e| http_error(&base, e))?;
    let body: Value = resp.into_json()?;
    let models = body["models"].as_array().cloned().unwrap_or_default();

    if models.is_empty() {
        println!("No models found. Copy .gguf files into the server's models folder (see `djeli config`).");
        return Ok(());
    }

    let width = models
        .iter()
        .filter_map(|m| m["name"].as_str())
        .map(str::len)
        .max()
        .unwrap_or(4)
        .max(4);
    println!("{:<width$}  SIZE", "NAME");
    for m in &models {
        let name = m["name"].as_str().unwrap_or("?");
        let size = m["size"].as_u64().unwrap_or(0);
        println!("{:<width$}  {}", name, human_size(size));
    }
    Ok(())
}

fn cmd_run(host: &str, args: RunArgs) -> anyhow::Result<()> {
    let base = base_url(host);
    // Accept Ollama-style "name:latest".
    let model_name = args
        .model
        .strip_suffix(":latest")
        .unwrap_or(&args.model)
        .to_string();

    let mut prompt = args.prompt.join(" ");
    let stdin_is_pipe = !std::io::stdin().is_terminal();
    if stdin_is_pipe {
        let mut piped = String::new();
        std::io::stdin().read_to_string(&mut piped)?;
        let piped = piped.trim();
        if !piped.is_empty() {
            prompt = if prompt.is_empty() {
                piped.to_string()
            } else {
                format!("{prompt}\n\n{piped}")
            };
        }
    }

    if prompt.trim().is_empty() {
        if stdin_is_pipe {
            bail!("empty prompt");
        }
        return interactive(&base, &model_name, &args);
    }

    // One-shot
    let mut messages = Vec::new();
    if let Some(sys) = &args.system {
        messages.push(msg("system", sys.clone()));
    }
    messages.push(msg("user", prompt));

    let reply = stream_chat(&base, &model_name, &args, &messages)?;
    if args.verbose {
        print_stats(&reply.stats);
    }
    Ok(())
}

fn interactive(base: &str, model_name: &str, args: &RunArgs) -> anyhow::Result<()> {
    let mut history: Vec<ChatMessage> = Vec::new();
    if let Some(sys) = &args.system {
        history.push(msg("system", sys.clone()));
    }

    println!("Send a message (/? for help, /bye to exit)");
    if args.history == 0 {
        println!("(no memory: every message starts fresh)");
    } else {
        println!(
            "(remembering the last {} exchange(s); /clear starts fresh)",
            args.history
        );
    }
    let stdin = std::io::stdin();

    loop {
        print!(">>> ");
        std::io::stdout().flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            println!(); // Ctrl+D
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        match line {
            "/bye" | "/exit" | "/quit" => break,
            "/clear" => {
                history.retain(|m| m.role == "system");
                println!("Cleared session context");
                continue;
            }
            "/?" | "/help" => {
                println!("Available commands:");
                println!("  /clear   Forget the conversation and start fresh");
                println!("  /bye     Exit (also /exit, /quit, Ctrl+D)");
                println!("  /?       Show this help");
                continue;
            }
            _ if line.starts_with('/') => {
                println!("Unknown command '{line}'. Type /? for help.");
                continue;
            }
            _ => {}
        }

        trim_history(&mut history, args.history);
        history.push(msg("user", line));
        match stream_chat(base, model_name, args, &history) {
            Ok(reply) => {
                history.push(msg("assistant", reply.text));
                println!();
                if args.verbose {
                    print_stats(&reply.stats);
                    println!();
                }
            }
            Err(e) => {
                history.pop(); // keep the history consistent
                eprintln!("Error: {e}\n");
            }
        }
    }
    Ok(())
}

struct Reply {
    text: String,
    stats: Option<Value>,
}

/// Sends the conversation to the server, prints tokens as they stream in,
/// and returns the full reply text.
fn stream_chat(
    base: &str,
    model_name: &str,
    args: &RunArgs,
    messages: &[ChatMessage],
) -> anyhow::Result<Reply> {
    let body = json!({
        "model": model_name,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "template": args.template,
        "repeat_penalty": args.repeat_penalty,
    });

    let resp = ureq::post(&format!("{base}/api/generate"))
        .send_json(body)
        .map_err(|e| http_error(base, e))?;

    let reader = BufReader::new(resp.into_reader());
    let mut stdout = std::io::stdout();
    let mut text = String::new();
    let mut stats = None;

    for line in reader.lines() {
        let line = line?;
        // SSE lines look like "data: {...}"; keep-alive comments start with ":".
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() {
            continue;
        }

        let event: Value = serde_json::from_str(data)?;
        if let Some(err) = event.get("error").and_then(Value::as_str) {
            bail!("{err}");
        }
        if let Some(chunk) = event.get("response").and_then(Value::as_str) {
            if !chunk.is_empty() {
                print!("{chunk}");
                stdout.flush()?;
                text.push_str(chunk);
            }
        }
        if event.get("done").and_then(Value::as_bool) == Some(true) {
            stats = Some(event);
            break;
        }
    }
    println!();

    Ok(Reply { text, stats })
}

fn print_stats(stats: &Option<Value>) {
    let Some(s) = stats else { return };
    let ms = |key: &str| s[key].as_u64().unwrap_or(0) as f64 / 1000.0;
    eprintln!();
    eprintln!("total duration:       {:.2}s", ms("total_ms"));
    eprintln!("setup duration:       {:.2}s", ms("setup_ms"));
    eprintln!("prompt eval count:    {} token(s)", s["prompt_tokens"].as_u64().unwrap_or(0));
    eprintln!("prompt eval duration: {:.2}s", ms("prefill_ms"));
    eprintln!("eval count:           {} token(s)", s["generated_tokens"].as_u64().unwrap_or(0));
    eprintln!("eval rate:            {:.2} tokens/s", s["tokens_per_second"].as_f64().unwrap_or(0.0));
    eprintln!("done reason:          {}", s["done_reason"].as_str().unwrap_or("?"));
}

fn cmd_stop(host: &str, model: Option<String>) -> anyhow::Result<()> {
    let base = base_url(host);
    let resp = ureq::post(&format!("{base}/api/unload"))
        .send_json(json!({ "model": model }))
        .map_err(|e| http_error(&base, e))?;
    let body: Value = resp.into_json()?;
    let names: Vec<&str> = body["unloaded"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if names.is_empty() {
        println!("No model was loaded.");
    } else {
        println!("Unloaded: {}", names.join(", "));
    }
    Ok(())
}

/// Keeps the system prompt plus only the last `turns` exchanges.
fn trim_history(history: &mut Vec<ChatMessage>, turns: usize) {
    let non_system = history.iter().filter(|m| m.role != "system").count();
    let keep = turns * 2;
    if non_system <= keep {
        return;
    }
    let mut to_drop = non_system - keep;
    history.retain(|m| {
        if m.role == "system" {
            true
        } else if to_drop > 0 {
            to_drop -= 1;
            false
        } else {
            true
        }
    });
}



// ===========================================================================
// Server
// ===========================================================================

type SseItem = Result<Event, std::io::Error>;

struct LoadedModel {
    weights: ModelWeights,
    tokenizer: Tok,
}

struct AppState {
    /// One lock per model: different models can generate concurrently,
    /// requests for the same model are serialized (its KV cache is stateful).
    loaded_models: Mutex<HashMap<String, Arc<Mutex<LoadedModel>>>>,
    device: Device,
    models_dir: String,
}

type SharedState = Arc<AppState>;

#[derive(Deserialize)]
struct GenerationRequest {
    model: String,
    /// Single prompt. Ignored when `messages` is provided.
    #[serde(default)]
    prompt: String,
    /// Conversation history: [{"role":"user|assistant|system","content":"..."}].
    #[serde(default)]
    messages: Option<Vec<ChatMessage>>,
    #[serde(default = "default_temp")]
    temperature: f64,
    /// Nucleus sampling threshold (0.0 - 1.0). Defaults to 0.9.
    #[serde(default)]
    top_p: Option<f64>,
    /// Seed. If omitted, a random one is used for each request.
    #[serde(default)]
    seed: Option<u64>,
    /// Cap on generated tokens (default 150).
    #[serde(default)]
    max_tokens: Option<usize>,
    /// Chat template: "zephyr" (default, also TinyLlama-chat), "chatml",
    /// "llama2" or "raw".
    #[serde(default)]
    template: Option<String>,
    /// Optional system prompt (used when `messages` has no system message).
    #[serde(default)]
    system: Option<String>,
    /// Penalty for recently used tokens (default 1.1, 1.0 disables it).
    #[serde(default)]
    repeat_penalty: Option<f32>,
    /// How many recent tokens the penalty looks at (default 64).
    #[serde(default)]
    repeat_last_n: Option<usize>,
}

#[derive(Serialize)]
struct ModelTag {
    name: String,
    size: u64,
}

#[derive(Serialize)]
struct TagsResponse {
    models: Vec<ModelTag>,
}

fn default_temp() -> f64 {
    0.7
}


// ---- Generation events ----------------------------------------------------

/// What the generation thread sends back; each API formats it its own way.
enum GenEvent {
    Chunk(String),
    Done(GenStats),
}

struct GenStats {
    reason: &'static str,
    prompt_tokens: usize,
    generated_tokens: usize,
    setup: Duration,
    prefill: Duration,
    total: Duration,
}

type GenItem = Result<GenEvent, String>;

// ---- Native API: SSE event helpers ----------------------------------------

fn chunk_event(text: &str) -> SseItem {
    Ok(Event::default().data(json!({ "response": text, "done": false }).to_string()))
}

fn done_event(s: &GenStats) -> SseItem {
    // Prefill also samples the first token, so decode speed is measured on the rest.
    let decode_time = s.total.saturating_sub(s.prefill).as_secs_f64();
    let decode_tokens = s.generated_tokens.saturating_sub(1) as f64;
    let tokens_per_second = if decode_time > 0.0 {
        (decode_tokens / decode_time * 100.0).round() / 100.0
    } else {
        0.0
    };

    Ok(Event::default().data(
        json!({
            "response": "",
            "done": true,
            "done_reason": s.reason,
            "prompt_tokens": s.prompt_tokens,
            "generated_tokens": s.generated_tokens,
            "setup_ms": s.setup.as_millis() as u64,
            "prefill_ms": s.prefill.as_millis() as u64,
            "total_ms": s.total.as_millis() as u64,
            "tokens_per_second": tokens_per_second,
        })
        .to_string(),
    ))
}

fn error_event(msg: impl Display) -> SseItem {
    Ok(Event::default().data(json!({ "error": msg.to_string(), "done": true }).to_string()))
}

// ---- Small utilities ------------------------------------------------------

/// Locks a std mutex, recovering the data if a previous holder panicked.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Applies the repeat penalty to the last `last_n` tokens of `context`.
fn penalize(
    logits: &Tensor,
    penalty: f32,
    last_n: usize,
    context: &[u32],
) -> candle_core::Result<Tensor> {
    if penalty == 1.0 || context.is_empty() {
        return Ok(logits.clone());
    }
    let start = context.len().saturating_sub(last_n);
    apply_repeat_penalty(logits, penalty, &context[start..])
}

fn random_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(42)
}

fn valid_model_name(name: &str) -> bool {
    !name.is_empty() && !name.contains('/') && !name.contains('\\') && !name.contains("..")
}

// ---- Chat templates -------------------------------------------------------

/// Normalizes the request into a validated list of chat messages.
fn build_messages(req: &GenerationRequest) -> anyhow::Result<Vec<ChatMessage>> {
    let mut messages = match &req.messages {
        Some(m) if !m.is_empty() => m.clone(),
        _ => {
            if req.prompt.trim().is_empty() {
                bail!("either 'prompt' or 'messages' is required");
            }
            vec![msg("user", req.prompt.clone())]
        }
    };

    for m in &messages {
        if !["system", "user", "assistant"].contains(&m.role.as_str()) {
            bail!("unknown role '{}' (use system, user or assistant)", m.role);
        }
    }

    let system = req.system.as_deref().map(str::trim).filter(|s| !s.is_empty());
    if let Some(sys) = system {
        if !messages.iter().any(|m| m.role == "system") {
            messages.insert(0, msg("system", sys));
        }
    }

    if messages.last().map(|m| m.role.as_str()) != Some("user") {
        bail!("the last message must have role 'user'");
    }
    Ok(messages)
}

/// Builds the final prompt text for the chosen chat template.
fn format_chat(template: &str, messages: &[ChatMessage]) -> anyhow::Result<String> {
    let mut s = String::new();
    match template {
        // TinyLlama-chat / Zephyr style
        "zephyr" | "tinyllama" => {
            for m in messages {
                s.push_str(&format!("<|{}|>\n{}</s>\n", m.role, m.content));
            }
            s.push_str("<|assistant|>\n");
        }
        // Qwen and many fine-tunes
        "chatml" => {
            for m in messages {
                s.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content));
            }
            s.push_str("<|im_start|>assistant\n");
        }
        // Llama-2-chat style: the system prompt is folded into the first user turn.
        "llama2" => {
            let system = messages
                .iter()
                .find(|m| m.role == "system")
                .map(|m| m.content.as_str());
            let mut first_user = true;
            for m in messages.iter().filter(|m| m.role != "system") {
                if m.role == "user" {
                    if !first_user {
                        s.push_str("<s>");
                    }
                    s.push_str("[INST] ");
                    if first_user {
                        if let Some(sys) = system {
                            s.push_str(&format!("<<SYS>>\n{sys}\n<</SYS>>\n\n"));
                        }
                    }
                    s.push_str(&format!("{} [/INST]", m.content));
                    first_user = false;
                } else {
                    s.push_str(&format!(" {} </s>", m.content));
                }
            }
        }
        // No template: contents are sent as-is
        "raw" => {
            let parts: Vec<&str> = messages.iter().map(|m| m.content.as_str()).collect();
            s = parts.join("\n");
        }
        other => bail!("unknown template '{other}' (use zephyr, chatml, llama2 or raw)"),
    }
    Ok(s)
}


// ---- HTTP server ----------------------------------------------------------

async fn serve(host: String, models_dir: PathBuf, config_path: PathBuf) -> anyhow::Result<()> {
    fs::create_dir_all(&models_dir)?;
    let models_dir_str = models_dir.to_string_lossy().into_owned();

    let state = Arc::new(AppState {
        loaded_models: Mutex::new(HashMap::new()),
        device: Device::Cpu,
        models_dir: models_dir_str.clone(),
    });

    let app = Router::new()
        .route("/", get(|| async { "djeli is running" }))
        // Native API
        .route("/api/tags", get(list_models_handler))
        .route("/api/generate", post(generate_handler))
        .route("/api/unload", post(unload_handler))
        // OpenAI-compatible API
        .route("/v1/models", get(openai_models_handler))
        .route("/v1/chat/completions", post(openai_chat_handler))
        .with_state(state);

    let bind_address = host
        .trim_start_matches("http://")
        .trim_end_matches('/')
        .to_string();
    let listener = tokio::net::TcpListener::bind(&bind_address).await?;

    println!("🚀 djeli server listening on http://{bind_address}");
    println!("   models folder: {models_dir_str}");
    println!("   config file:   {}", config_path.display());
    if scan_models(&models_dir_str).is_empty() {
        println!("   no models yet: copy a .gguf file into the models folder.");
    }

    axum::serve(listener, app).await?;
    Ok(())
}

fn scan_models(dir: &str) -> Vec<ModelTag> {
    let mut models = vec![];
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                    models.push(ModelTag {
                        name: name.to_string(),
                        size,
                    });
                }
            }
        }
    }
    models.sort_by(|a, b| a.name.cmp(&b.name));
    models
}

async fn list_models_handler(State(state): State<SharedState>) -> Json<TagsResponse> {
    Json(TagsResponse {
        models: scan_models(&state.models_dir),
    })
}

/// Generation runs on a blocking thread so the async runtime stays free.
/// Events travel back to the HTTP handler through an mpsc channel.
fn spawn_generation(state: SharedState, req: GenerationRequest) -> mpsc::Receiver<GenItem> {
    let (tx, rx) = mpsc::channel::<GenItem>(64);
    tokio::task::spawn_blocking(move || {
        if let Err(e) = run_generation(&state, req, &tx) {
            let _ = tx.blocking_send(Err(format!("{e:#}")));
        }
    });
    rx
}

async fn generate_handler(
    State(state): State<SharedState>,
    Json(payload): Json<GenerationRequest>,
) -> Sse<impl Stream<Item = SseItem>> {
    let mut rx = spawn_generation(state, payload);

    let stream = async_stream::stream! {
        while let Some(item) = rx.recv().await {
            yield match item {
                Ok(GenEvent::Chunk(text)) => chunk_event(&text),
                Ok(GenEvent::Done(stats)) => done_event(&stats),
                Err(e) => error_event(e),
            };
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

#[derive(Deserialize)]
struct UnloadRequest {
    #[serde(default)]
    model: Option<String>,
}

/// Drops models from memory (all of them, or just the named one).
async fn unload_handler(
    State(state): State<SharedState>,
    Json(req): Json<UnloadRequest>,
) -> Json<Value> {
    let mut map = lock(&state.loaded_models);
    let names: Vec<String> = match req.model {
        Some(name) => {
            if map.remove(&name).is_some() {
                vec![name]
            } else {
                vec![]
            }
        }
        None => map.drain().map(|(name, _)| name).collect(),
    };
    Json(json!({ "unloaded": names }))
}

// ---- OpenAI-compatible API ------------------------------------------------

#[derive(Deserialize)]
struct OpenAiMessage {
    role: String,
    #[serde(default)]
    content: Value,
}

#[derive(Deserialize)]
struct OpenAiChatRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    n: Option<u32>,
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Message content is either a string or a list of `{"type":"text","text":...}` parts.
fn openai_content_text(content: &Value) -> Result<String, String> {
    match content {
        Value::String(s) => Ok(s.clone()),
        Value::Null => Ok(String::new()),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        out.push_str(part.get("text").and_then(Value::as_str).unwrap_or(""))
                    }
                    Some(other) => {
                        return Err(format!("unsupported content part '{other}' (text only)"))
                    }
                    None => return Err("unsupported content part without a type".to_string()),
                }
            }
            Ok(out)
        }
        _ => Err("unsupported message content".to_string()),
    }
}

fn openai_messages(req: &OpenAiChatRequest) -> Result<Vec<ChatMessage>, String> {
    let mut out = Vec::new();
    for m in &req.messages {
        let role = match m.role.as_str() {
            "system" | "developer" => "system",
            "user" => "user",
            "assistant" => "assistant",
            other => return Err(format!("unsupported message role '{other}'")),
        };
        out.push(msg(role, openai_content_text(&m.content)?));
    }
    Ok(out)
}

fn status_for_error(message: &str) -> (StatusCode, &'static str) {
    if message.starts_with("model '") && message.contains("not found") {
        return (StatusCode::NOT_FOUND, "invalid_request_error");
    }
    let client_errors = [
        "prompt too long",
        "invalid",
        "unknown",
        "required",
        "must have",
        "unsupported",
    ];
    if client_errors.iter().any(|k| message.contains(k)) {
        (StatusCode::BAD_REQUEST, "invalid_request_error")
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, "server_error")
    }
}

fn openai_error(message: impl Display) -> Response {
    let message = message.to_string();
    let (status, kind) = status_for_error(&message);
    (
        status,
        Json(json!({
            "error": { "message": message, "type": kind, "param": null, "code": null }
        })),
    )
        .into_response()
}

fn openai_chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: Value,
    finish_reason: Option<&str>,
) -> SseItem {
    Ok(Event::default().data(
        json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }],
        })
        .to_string(),
    ))
}

async fn openai_models_handler(State(state): State<SharedState>) -> Json<Value> {
    let created = unix_now();
    let data: Vec<Value> = scan_models(&state.models_dir)
        .into_iter()
        .map(|m| json!({ "id": m.name, "object": "model", "created": created, "owned_by": "local" }))
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

/// `POST /v1/chat/completions`, streaming and non-streaming.
async fn openai_chat_handler(
    State(state): State<SharedState>,
    Json(req): Json<OpenAiChatRequest>,
) -> Response {
    if req.n.unwrap_or(1) != 1 {
        return openai_error("unsupported value: only n = 1 is supported");
    }
    let messages = match openai_messages(&req) {
        Ok(m) => m,
        Err(e) => return openai_error(e),
    };

    let model_name = req.model.clone();
    let gen_req = GenerationRequest {
        model: req.model.clone(),
        prompt: String::new(),
        messages: Some(messages),
        temperature: req.temperature.unwrap_or_else(default_temp),
        top_p: req.top_p,
        seed: req.seed,
        max_tokens: Some(
            req.max_completion_tokens
                .or(req.max_tokens)
                .unwrap_or(OPENAI_DEFAULT_MAX_TOKENS),
        ),
        template: None,
        system: None,
        repeat_penalty: None,
        repeat_last_n: None,
    };

    let mut rx = spawn_generation(state, gen_req);
    let id = format!("chatcmpl-{:x}", random_seed());
    let created = unix_now();

    if req.stream {
        let stream = async_stream::stream! {
            yield openai_chunk(&id, created, &model_name, json!({ "role": "assistant", "content": "" }), None);
            while let Some(item) = rx.recv().await {
                match item {
                    Ok(GenEvent::Chunk(text)) => {
                        yield openai_chunk(&id, created, &model_name, json!({ "content": text }), None);
                    }
                    Ok(GenEvent::Done(stats)) => {
                        yield openai_chunk(&id, created, &model_name, json!({}), Some(stats.reason));
                    }
                    Err(e) => {
                        let (_, kind) = status_for_error(&e);
                        yield Ok(Event::default().data(
                            json!({ "error": { "message": e, "type": kind, "param": null, "code": null } }).to_string(),
                        ));
                        break;
                    }
                }
            }
            yield Ok(Event::default().data("[DONE]"));
        };
        return Sse::new(stream).keep_alive(KeepAlive::default()).into_response();
    }

    // Non-streaming: collect the whole reply.
    let mut text = String::new();
    let mut stats: Option<GenStats> = None;
    while let Some(item) = rx.recv().await {
        match item {
            Ok(GenEvent::Chunk(t)) => text.push_str(&t),
            Ok(GenEvent::Done(s)) => stats = Some(s),
            Err(e) => return openai_error(e),
        }
    }
    let (reason, prompt_tokens, completion_tokens) = match &stats {
        Some(s) => (s.reason, s.prompt_tokens, s.generated_tokens),
        None => ("stop", 0, 0),
    };
    Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": text },
            "finish_reason": reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }))
    .into_response()
}

// ---- Model loading ----------------------------------------------------------

/// Loads a GGUF model. The tokenizer comes from the GGUF metadata itself,
/// exactly like llama.cpp does, so no extra file is needed.
fn load_model(models_dir: &str, model_name: &str, device: &Device) -> anyhow::Result<LoadedModel> {
    let model_path = format!("{}/{}.gguf", models_dir, model_name);
    let mut file = File::open(&model_path)?;
    let content = gguf_file::Content::read(&mut file)?;

    // Read the tokenizer before `from_gguf` consumes the metadata.
    let tokenizer = Tok::load(models_dir, model_name, &content)?;
    let weights = ModelWeights::from_gguf(content, &mut file, device)?;

    Ok(LoadedModel { weights, tokenizer })
}

/// Returns the cached model, loading it from disk on first use.
fn get_or_load_model(state: &AppState, name: &str) -> anyhow::Result<Arc<Mutex<LoadedModel>>> {
    if let Some(existing) = lock(&state.loaded_models).get(name) {
        return Ok(existing.clone());
    }

    let model_path = format!("{}/{}.gguf", state.models_dir, name);
    if !Path::new(&model_path).exists() {
        bail!("model '{}' not found", name);
    }

    // Load without holding the map lock, so other models stay usable meanwhile.
    let loaded = Arc::new(Mutex::new(load_model(&state.models_dir, name, &state.device)?));

    let mut map = lock(&state.loaded_models);
    Ok(map.entry(name.to_string()).or_insert(loaded).clone())
}

fn run_generation(
    state: &AppState,
    req: GenerationRequest,
    tx: &mpsc::Sender<GenItem>,
) -> anyhow::Result<()> {
    let request_start = Instant::now();

    if !valid_model_name(&req.model) {
        bail!("invalid model name");
    }

    let template = req.template.as_deref().unwrap_or("zephyr");
    let messages = build_messages(&req)?;
    let formatted_prompt = format_chat(template, &messages)?;

    let model_arc = get_or_load_model(state, &req.model)?;
    let mut guard = lock(&model_arc);
    let LoadedModel { weights, tokenizer } = &mut *guard;

    // Everything before this point (waiting for the model lock, loading the
    // GGUF on first use) is setup time, not prefill time.
    let setup_time = request_start.elapsed();
    let started = Instant::now();

    // ---- Tokenize ----
    let prompt_tokens = tokenizer.encode(&formatted_prompt)?;
    let prompt_len = prompt_tokens.len();

    let room = MAX_CONTEXT.saturating_sub(prompt_len);
    if room == 0 {
        bail!(
            "prompt too long: {} tokens, context limit is {} (try /clear)",
            prompt_len,
            MAX_CONTEXT
        );
    }
    let max_tokens = req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS).clamp(1, room);

    // ---- Stop tokens ----
    let mut stop_tokens: Vec<u32> = STOP_TOKEN_CANDIDATES
        .iter()
        .filter_map(|t| tokenizer.token_id(t))
        .collect();
    if stop_tokens.is_empty() {
        stop_tokens.push(2);
    }

    // ---- Sampling setup ----
    // In candle-transformers 0.4.1, `None` temperature means greedy (argmax).
    let temperature = if req.temperature <= 0.0 {
        None
    } else {
        Some(req.temperature)
    };
    // Defaults similar to Ollama: top_p 0.9, repeat penalty 1.1 over the last 64 tokens.
    let top_p = Some(req.top_p.unwrap_or(0.9)).filter(|p| *p > 0.0 && *p < 1.0);
    let repeat_penalty = req.repeat_penalty.unwrap_or(1.1);
    let repeat_last_n = req.repeat_last_n.unwrap_or(64);
    let seed = req.seed.unwrap_or_else(random_seed);
    let mut logits_processor = LogitsProcessor::new(seed, temperature, top_p);

    // ---- Prefill: run the whole prompt, sample the first token ----
    let input = Tensor::new(prompt_tokens.as_slice(), &state.device)?.unsqueeze(0)?;
    let mut context = prompt_tokens.clone(); // prompt + generated, for the repeat penalty
    let logits = weights.forward(&input, 0)?.squeeze(0)?;
    let logits = penalize(&logits, repeat_penalty, repeat_last_n, &context)?;
    let mut next_token = logits_processor.sample(&logits)?;
    let mut index_pos = prompt_len;
    let prefill_time = started.elapsed();

    // ---- Decode loop ----
    let mut generated: Vec<u32> = Vec::new();
    let mut emitted_len = 0usize; // bytes of decoded text already sent
    let mut reason = "length";

    while generated.len() < max_tokens {
        if stop_tokens.contains(&next_token) {
            reason = "stop";
            break;
        }
        generated.push(next_token);
        context.push(next_token);

        // Decode the whole sequence so SentencePiece leading spaces are kept,
        // then send only the new suffix.
        let full = tokenizer.decode(&generated);
        // Hold back if the tail is an incomplete UTF-8 sequence.
        if !full.ends_with('\u{FFFD}') {
            if let Some(delta) = full.get(emitted_len..) {
                if !delta.is_empty() {
                    emitted_len = full.len();
                    if tx.blocking_send(Ok(GenEvent::Chunk(delta.to_string()))).is_err() {
                        // Client disconnected: stop wasting CPU.
                        return Ok(());
                    }
                }
            }
        }

        if generated.len() >= max_tokens {
            break; // no need for another forward pass
        }

        let step_input = Tensor::new(&[next_token], &state.device)?.unsqueeze(0)?;
        let logits = weights.forward(&step_input, index_pos)?.squeeze(0)?;
        let logits = penalize(&logits, repeat_penalty, repeat_last_n, &context)?;
        next_token = logits_processor.sample(&logits)?;
        index_pos += 1;
    }

    // ---- Flush any text held back by the incomplete-UTF-8 check ----
    let full = tokenizer.decode(&generated);
    if let Some(rest) = full.get(emitted_len..) {
        if !rest.is_empty() && tx.blocking_send(Ok(GenEvent::Chunk(rest.to_string()))).is_err() {
            return Ok(());
        }
    }

    let _ = tx.blocking_send(Ok(GenEvent::Done(GenStats {
        reason,
        prompt_tokens: prompt_len,
        generated_tokens: generated.len(),
        setup: setup_time,
        prefill: prefill_time,
        total: started.elapsed(),
    })));
    Ok(())
}
