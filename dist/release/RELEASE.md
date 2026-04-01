# Ollama v2.1.0-turboquant Release

> **This is NOT the official Ollama.** This is the [TurboQuant fork](https://github.com/amarce/ollama) which adds automatic KV cache compression on NVIDIA GPUs and image generation support on Windows.

## What's New in v2.1.0

- **Image generation on Windows**: MLX CUDA 12 backend enables image generation models (FLUX, z-image-turbo, etc.) on NVIDIA GPUs
- **CUDA dequantization crash fix**: Fixed lazy tensor evaluation bug where `Shape()` returned empty slices after dequantization on CUDA backends, causing index-out-of-bounds crashes with quantized image generation models
- **Full backend bundle**: Installer now includes CPU, CUDA 12, CUDA 13, ROCm, Vulkan, and MLX CUDA 12/13 backends — everything in one package

### Carried from v2.0.x

- **TurboQuant KV cache compression**: Auto-enables on NVIDIA CUDA GPUs with Flash Attention support, compressing the KV cache by up to ~5.3x (3-bit default)
- **Tri-state TurboQuant control**: Auto (default) / On / Off — configurable via desktop app Settings dropdown or `OLLAMA_TURBOQUANT` environment variable
- **CUDA kernel optimizations**: Shared-memory residual caching eliminates ~12K volatile reads per vector in the QJL stage

## Windows amd64 Build

- **ollama.exe**: CLI binary for Windows x86-64
- **OllamaSetup-turboquant.exe**: Full Inno Setup installer with all backends
- **Backends included**:
  - CPU (ggml-cpu)
  - CUDA 12 (ggml-cuda + TurboQuant)
  - CUDA 13 (ggml-cuda)
  - MLX CUDA 12 (image generation)
  - MLX CUDA 13 (image generation)
  - ROCm 6 (AMD)
  - Vulkan

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
2. Extract to `%LOCALAPPDATA%\Programs\Ollama`
3. For image generation, also extract `ollama-windows-amd64-mlx.zip` to the same location
4. Add the directory to your PATH

## Image Generation

Supported models include FLUX and z-image-turbo variants. Example:

```
ollama run x/z-image-turbo:latest "generate an image of a mountain landscape"
```

**Requirements**: NVIDIA GPU with CUDA 12+ and cuDNN 9.18+ installed.

## TurboQuant Configuration

| Setting | Value | Effect |
|---------|-------|--------|
| `OLLAMA_TURBOQUANT` | _(not set)_ | Auto: enables on supported NVIDIA GPUs |
| `OLLAMA_TURBOQUANT` | `true` or `1` | Force enable |
| `OLLAMA_TURBOQUANT` | `false` or `0` | Force disable |

Default bit-width is 3-bit (~5.3x compression). Context length is automatically scaled based on measured compression ratio.

## SHA256 Checksums

See `SHA256SUMS` for verification.
