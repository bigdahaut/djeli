# djeli

Run GGUF language models locally. No cloud, no API key, no data leaving your machine.

`djeli` is a self-contained inference server for `.gguf` models. It reads the
tokenizer and chat template directly from the GGUF file, so supported models
work out of the box: Qwen2, Qwen2.5, Llama, TinyLlama, Mistral.

## Install

**Linux / macOS:**

```bash
curl -fsSL https://raw.githubusercontent.com/bigdahaut/djeli/main/install.sh | sh
```

**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/bigdahaut/djeli/main/install.ps1 | iex
```

## Quick start

Download a model directly from Hugging Face:

```bash
djeli pull qwen2.5-1.5b
```

Start the server:

```bash
djeli serve
```
# In another terminal:
```bash
djeli run qwen2.5-1.5b
```

(Available aliases: tinyllama, qwen2.5-0.5b, qwen2.5-1.5b, llama3.2-1b, llama3.2-3b, phi3.5-mini)

## Commands

Command   Description
-----------------------------------------------
djeli serve	Start the HTTP server on 127.0.0.1:11435.

djeli run <model>	One-shot prompt or interactive chat.

djeli list	List installed .gguf files.

djeli stop [model]	Unload a model from memory.

djeli config	Show the config file and settings.

## Python client

```bash
pip install djeli
```

```python
from djeli import Djeli
client = Djeli()
answer = client.chat(
    model="qwen2.5-1.5b-instruct-q4_k_m",
    messages=[{"role": "user", "content": "What is the capital of Kenya?"}],
)
```

## API

. Native: /api/tags, /api/generate, /api/unload

. OpenAI-compatible: /v1/models, /v1/chat/completions

## License

### MIT

