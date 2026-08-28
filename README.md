# AI Router

A lightweight multi-provider AI routing layer that provides a unified interface for interacting with multiple AI providers and models.

The project is designed around **provider fallback, model fallback, provider/model selection, diagnostics, and OpenAI-compatible request handling**.

The current implementation is intentionally lightweight. More advanced AI Gateway capabilities will be introduced incrementally.

---

## Features

* Multi-provider AI routing
* Provider-level fallback
* Model-level fallback
* Model deduplication
* Rate-limit handling
* Authentication error handling
* Model-not-found handling
* Payment/quota error handling
* Network error handling
* Request timing and diagnostics
* Provider override
* Model override
* Primary-model testing
* Provider model discovery
* Structured OpenAI-compatible messages
* Tool/function calling support
* Google Gemini native API support
* Ollama native API support
* Ollama Cloud model support
* Local Ollama model support
* OpenAI-compatible provider support

---

## Supported Providers

| Provider              | API Type          | Status    |
| --------------------- | ----------------- | --------- |
| Google Gemini         | Gemini API        | Supported |
| Groq                  | OpenAI-compatible | Supported |
| OpenRouter            | OpenAI-compatible | Supported |
| Cloudflare Workers AI | OpenAI-compatible | Supported |
| Hugging Face          | OpenAI-compatible | Supported |
| Ollama                | Native Ollama API | Supported |

Provider priority is currently:

```text
Google
  ↓
Groq
  ↓
OpenRouter
  ↓
Cloudflare
  ↓
Hugging Face
  ↓
Ollama
```

If a provider or model fails, the router attempts the next available model/provider according to the configured fallback strategy.

---

# Ollama

Ollama is currently used as the final provider in the default routing chain.

The project supports both:

* Ollama Cloud models
* Local Ollama models

Ollama Cloud models are accessed through the local Ollama service and therefore do not require the machine to locally load the full model into RAM/VRAM.

---

## Ollama Cloud Models

The currently available Ollama Cloud models include:

```text
gemma4:31b
gpt-oss:120b
gpt-oss:20b
minimax-m3
nemotron-3-nano:30b
nemotron-3-super
nemotron-3-ultra
```

When used through the Ollama CLI/API, cloud models are referenced using the `:cloud` suffix:

```text
gemma4:31b:cloud
gpt-oss:120b:cloud
gpt-oss:20b:cloud
minimax-m3:cloud
nemotron-3-nano:30b:cloud
nemotron-3-super:cloud
nemotron-3-ultra:cloud
```

Example:

```powershell
ollama run minimax-m3:cloud
```

---

## Ollama Cloud Requirements

Install Ollama and sign in to your Ollama account.

Verify the installation:

```powershell
ollama --version
```

Check available models:

```powershell
ollama list
```

Test a cloud model:

```powershell
ollama run minimax-m3:cloud
```

Ollama should connect to the cloud model automatically.

---

# Local Ollama Models

The router can also use locally installed Ollama models.

Example:

```text
qwen2.5-coder:1.5b
qwen2.5-coder:0.5b
gemma3:1b
```

The local Ollama API normally runs at:

```text
http://localhost:11434
```

This can be changed through:

```env
OLLAMA_BASE_URL=http://localhost:11434
```

---

# Installation

## Requirements

* Python 3.10+
* Ollama
* Internet connection
* API keys for the providers you want to enable

Clone the repository:

```bash
git clone https://github.com/Dilipx07/AI-Router.git
cd AI-Router
```

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Environment Configuration

Create a `.env` file from the provided example:

### Windows

```powershell
Copy-Item .env.example .env
```

### Linux/macOS

```bash
cp .env.example .env
```

Configure the providers you want to use.

Example:

```env
HF_API_KEY=
GOOGLE_API_KEY=
GROQ_API_KEY=
OPENROUTER_API_KEY=

CLOUDFLARE_API_KEY=
CLOUDFLARE_ACCOUNT_ID=

OLLAMA_BASE_URL=http://localhost:11434
```

API keys should never be committed to Git.

The `.env` file is intentionally excluded through `.gitignore`.

---

# Usage

## Basic Request

Send a prompt to the router:

```powershell
python token_router.py "Hello, explain artificial intelligence"
```

The router will attempt providers according to the configured provider order.

---

## Check Provider Status

```powershell
python token_router.py --status
```

Example output:

```text
PROVIDER STATUS

google          READY                primary=gemini-3.1-flash-lite
groq            READY                primary=openai/gpt-oss-120b
openrouter      READY                primary=openai/gpt-oss-20b:free
cloudflare      NO API KEY           primary=@cf/qwen/qwen3.8-27b
huggingface     READY                primary=Qwen/Qwen3-Coder-Next
ollama          LOCAL                primary=minimax-m3:cloud
```

---

# View Configured Models

```powershell
python token_router.py --models
```

This displays the primary and fallback models configured for every provider.

---

# Test Primary Models

Test the primary model of every configured provider:

```powershell
python token_router.py --test-all
```

Only primary models are tested.

Fallback models are not consumed during this test.

The router validates the response against:

```text
TEST_OK
```

---

# Provider Override

Force the router to use a specific provider:

```powershell
python token_router.py \
    --provider ollama \
    "Hello"
```

Windows PowerShell:

```powershell
python token_router.py --provider ollama "Hello"
```

---

# Model Override

Force a specific model:

```powershell
python token_router.py `
    --provider ollama `
    --model minimax-m3:cloud `
    "Hello"
```

For PowerShell, this can also be written on one line:

```powershell
python token_router.py --provider ollama --model minimax-m3:cloud "Hello"
```

When `--model` is specified, the router uses only that model.

---

# Interactive Mode

Start an interactive session:

```powershell
python token_router.py --interactive
```

Optional system prompt:

```powershell
python token_router.py --interactive --system "You are an expert Python developer."
```

Force Ollama:

```powershell
python token_router.py --interactive --provider ollama
```

Exit using:

```text
exit
quit
/exit
/quit
```

---

# Model Discovery

The router can query available models from supported providers.

Example:

```powershell
python token_router.py --discover google
```

For Ollama:

```powershell
python token_router.py --discover ollama
```

This queries the local Ollama model registry:

```text
/api/tags
```

---

# Fallback Architecture

The router operates at two levels.

## Provider Fallback

If the primary provider cannot successfully process a request, the router moves to the next provider.

```text
Google
   ↓ failure
Groq
   ↓ failure
OpenRouter
   ↓ failure
Cloudflare
   ↓ failure
Hugging Face
   ↓ failure
Ollama
```

## Model Fallback

Each provider can also have multiple models.

Example:

```text
Google
│
├── gemini-3.1-flash-lite
├── gemini-3.6-flash
├── gemini-3.5-flash-lite
├── gemini-3.5-flash
├── gemini-2.5-flash
└── gemini-2.5-flash-lite
```

If the primary model fails for a recoverable reason, the router attempts the next configured model.

---

# Error Handling

The router identifies several failure conditions:

```text
rate_limit
auth_error
model_not_found
payment_required
network_error
empty
error
```

Authentication and payment/quota failures can stop attempts for the current provider rather than blindly consuming additional fallback requests.

---

# Tool Calling

The router supports structured OpenAI-compatible messages and tool calls.

Messages can contain:

```text
system
user
assistant
tool
```

Assistant tool calls and tool results are preserved when routing through compatible providers.

Google Gemini requests are converted from OpenAI-style function tools into Gemini's native function-calling format.

---

# Project Structure

```text
AI-Router/
│
├── token_router.py
│   └── Core routing logic
│
├── router_api.py
│   └── OpenAI-compatible HTTP API
│
├── requirements.txt
│   └── Python dependencies
│
├── .env.example
│   └── Environment configuration template
│
├── .gitignore
│   └── Git exclusions
│
├── LICENSE
│   └── MIT License
│
├── README.md
│   └── Project documentation
│
└── backups/
    └── Previous router versions
```

---

# Router API

The project also contains `router_api.py`, which exposes the router through an OpenAI-compatible HTTP interface.

Current endpoints:

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

The API is designed so applications that already understand the OpenAI API format can communicate with the router without directly implementing provider-specific integrations.

---

# Security

Never commit credentials to Git.

Do not commit:

```text
.env
API keys
access tokens
passwords
private keys
credentials
```

Use `.env.example` to document required environment variables.

---

# Development

The current project is intentionally lightweight.

The immediate focus is:

1. Reliable provider routing
2. Model fallback
3. Ollama Cloud integration
4. OpenAI-compatible API access
5. Diagnostics and observability

Future development may introduce additional gateway capabilities such as:

* Request authentication
* API key management
* Usage tracking
* Token accounting
* Per-model routing policies
* Cost-aware routing
* Rate-limit management
* Provider health monitoring
* Request logging
* Metrics
* Load balancing
* Streaming
* Persistent configuration
* Administrative APIs

These capabilities are not considered part of the current implementation unless explicitly implemented.

---

# License

This project is licensed under the MIT License.

See [LICENSE](LICENSE) for details.
