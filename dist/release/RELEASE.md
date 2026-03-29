# Ollama v2.0.2-turboquant Release

> **This is NOT the official Ollama.** This is the [TurboQuant fork](https://github.com/amarce/ollama) which adds automatic KV cache compression on NVIDIA GPUs.

## What's New in v2.0.2

- **TurboQuant KV cache compression**: Auto-enables on NVIDIA CUDA GPUs with Flash Attention support, compressing the KV cache by up to ~5.3x (3-bit default)
- **Tri-state TurboQuant control**: Auto (default) / On / Off — configurable via desktop app Settings dropdown or `OLLAMA_TURBOQUANT` environment variable
- **CUDA kernel optimizations**: Shared-memory residual caching eliminates ~12K volatile reads per vector in the QJL stage
- **Flash Attention safety gates**: FA promotion now properly checks both GPU capability and model support before enabling
- **CPU-only FA fix**: Restored correct semantics for CPU-only mode (no GPU = no unsupported GPU)
- **Streaming robustness**: Error JSON responses on marshal failures, progress timeout detection
- **Retry backoff**: Exponential backoff for expired runner retries (10ms–640ms cap)

## Windows amd64 Build

- **ollama.exe**: Cross-compiled with mingw for Windows x86-64
- **OllamaSetup-turboquant.exe**: Full Go-based installer that replicates official Inno Setup behavior
- **Build flags**: `-trimpath -ldflags "-s -w"` with version 2.0.2-turboquant
- **CGO**: Enabled with x86_64-w64-mingw32-gcc cross-compiler

## Installation

### One-Click Installer (recommended)

1. Download `OllamaSetup-turboquant.exe`
2. Run the installer — it will:
   - Stop any running Ollama processes
   - Install `ollama.exe` to `%LOCALAPPDATA%\Programs\Ollama`
   - Add the install directory to your user PATH
   - Register the `ollama://` URL protocol
   - Create a Start Menu shortcut
3. Can upgrade an existing official Ollama installation in-place

For unattended installs:
```powershell
OllamaSetup-turboquant.exe /VERYSILENT
```

### Manual Install

1. Download `ollama-windows-amd64-turboquant.zip`
2. Extract `ollama.exe` to `%LOCALAPPDATA%\Programs\Ollama`
3. Add that directory to your PATH

## TurboQuant Configuration

| Setting | Value | Effect |
|---------|-------|--------|
| `OLLAMA_TURBOQUANT` | _(not set)_ | Auto: enables on supported NVIDIA GPUs |
| `OLLAMA_TURBOQUANT` | `true` or `1` | Force enable |
| `OLLAMA_TURBOQUANT` | `false` or `0` | Force disable |

Default bit-width is 3-bit (~5.3x compression). Context length is automatically scaled based on measured compression ratio.

## SHA256 Checksums

See `SHA256SUMS` for verification.
