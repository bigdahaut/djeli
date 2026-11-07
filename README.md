# djeli

Run GGUF language models locally. No cloud, no API key, no data leaving your machine.

`djeli` is a self-contained inference server for `.gguf` models. It reads the
tokenizer and chat template directly from the GGUF file, so supported models
work out of the box: Qwen2, Qwen2.5, Llama, TinyLlama, Mistral.

- **Offline inference.** All computation happens on your machine (CPU or GPU).
- **OpenAI-compatible API.** Drop-in replacement for `/v1/chat/completions`.
- **One binary.** No Python, no runtime dependencies.
- **Model pulling.** Download models directly from Hugging Face.

## Install

**Linux / macOS:**

```bash
curl -fsSL https://raw.githubusercontent.com/bigdahaut/djeli/main/install.sh | sh
```
**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/bigdahaut/djeli/main/install.ps1 | iex
```
The installer places djeli in your user path and automatically creates the default models folder.

## Quick start

Download a model directly from Hugging Face:

```bash
djeli pull qwen2.5-1.5b
```
(Available aliases: tinyllama, qwen2.5-0.5b, qwen2.5-1.5b, llama3.2-1b, llama3.2-3b, phi3.5-mini. You can also pass any full Hugging Face path, like user/repo/file.gguf).

Start the server:

```bash
djeli serve
```

In another terminal, run the model:

```bash
djeli run qwen2.5-1.5b
```
>>> What is the capital of Kenya?
The capital of Kenya is Nairobi.

## Customization

You can customize the model's behavior (temperature, system prompt, etc.) in three ways. Priority: CLI flags > Modelfile > Defaults.

1. Using CLI flags (One-time override)

Pass arguments directly when you run the model. This overrides any defaults:

```bash
djeli run qwen2.5-1.5b "Write a poem" --temperature 0.8 --system "You are a creative poet."
```
2. Using Modelfiles (Permanent custom defaults)

When you run djeli pull <model>, djeli automatically generates a customizable .model file right next to the .gguf file.

To customize the model's permanent behavior, simply open that .model file and uncomment/edit the lines.

(Run djeli config to see your exact models directory. By default, it is ~/.djeli/models/ on Linux/macOS and %USERPROFILE%\.djeli\models\ on Windows).

For example, if you edit qwen2.5-1.5b.model, it will look like this:

```text
SYSTEM """
Tu es SchoolStat, un assistant statistique du Ministère de l'Éducation Nationale.
Tu réponds uniquement en français, de manière concise et factuelle.
"""

PARAMETER temperature 0.0
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.3
```
Now, whenever you run djeli run qwen2.5-1.5b, it will automatically use these settings. (You can still override them with CLI flags like --temperature 0.8 if needed).

3. Using the API (Python Client)

If you are using the pip install djeli Python client, you can pass the parameters directly to the chat() method:

```python
from djeli import Djeli
client = Djeli()

answer = client.chat(
    model="qwen2.5-1.5b",
    messages=[{"role": "user", "content": "Write a poem"}],
    temperature=0.8,
    max_tokens=200
)
```
## Commands

Command	             Description
------------------------------------------------------------
djeli serve	        Start the HTTP server on 127.0.0.1:11435.

djeli pull <model>	Download a model and generate a Modelfile scaffold.

djeli run <model>	One-shot prompt or interactive chat.

djeli list	        List installed .gguf files.

djeli stop [model]	Unload a model from memory.

djeli config	    Show the config file and settings (including the models directory).


## Hardware & GPU Support

By default, djeli runs on your CPU. If you have a CUDA-enabled Nvidia GPU or an Apple Silicon Mac, you can build the engine from source with GPU acceleration:

```bash
git clone https://github.com/bigdahaut/djeli.git
cd djeli

# For Nvidia CUDA:
cargo build --release --features cuda

# For Apple Silicon (Metal):
cargo build --release --features metal
```
The server will automatically detect and use the GPU when you run djeli serve.

## API

. Native: /api/tags, /api/generate, /api/unload

. OpenAI-compatible: /v1/models, /v1/chat/completions

## License

### MIT

