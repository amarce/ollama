# Ollama v0.6.0-turboquant Release

## Windows amd64 Build

- **ollama.exe**: Cross-compiled with LLVM-mingw (clang-based) for Windows x86-64
- **Build flags**: `-trimpath -ldflags "-s -w"` with version 0.6.0-turboquant
- **CGO**: Enabled with x86_64-w64-mingw32-gcc cross-compiler
- **Format**: PE32+ executable (console) x86-64, for MS Windows

## Installation

### Quick Install (PowerShell)

1. Download `ollama-windows-amd64-turboquant.zip`
2. Extract to a directory
3. Run: `.\install-ollama-turboquant.ps1`

### Manual Install

1. Download `ollama-windows-amd64-turboquant.zip`
2. Extract `ollama.exe` to `%LOCALAPPDATA%\Programs\Ollama`
3. Add that directory to your PATH
4. Set `OLLAMA_TURBOQUANT=true` to enable KV cache compression

## TurboQuant Configuration

| Environment Variable | Value | Effect |
|---------------------|-------|--------|
| `OLLAMA_TURBOQUANT` | `true` or `1` | Enable 3-bit (default, ~4x effective compression) |
| `OLLAMA_TURBOQUANT` | `4` | 4-bit (~4x effective compression) |
| `OLLAMA_TURBOQUANT` | `2` | 2-bit (~4x effective compression) |

## SHA256 Checksums

See `SHA256SUMS` for verification.

## Build from Source

To build with full CUDA backend support (includes TurboQuant CUDA kernels):

```powershell
# On Windows with CUDA Toolkit installed:
.\scripts\build_windows.ps1
```

The TurboQuant CUDA kernels (`turboquant.cu`) are automatically compiled
as part of the ggml-cuda target via CMake glob discovery.
