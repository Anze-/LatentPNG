# Do we (I) need this?
Do you need to do multiple passes through a model to edit your image, does your image degrade when doing this (becoming more and more saturated and other issues)?

When editing regular sensor-acquired photos a common approach is to use the RAW format to avoid image degradation due to multiple compressions.

When working with generated images a similar problem arises, the solution is to use the latent image.

This format is:
 * readable by any kind of common image viewer: it can open a png -> it can display this format
 * can be intuitively used in ComfyUI, by loading/saving images with the provided custom nodes.

# LatentPNG (LPNG)
A backward-compatible PNG extension for storing diffusion latents directly inside PNG metadata.

LatentPNG allows image workflows to store the model-native latent representation alongside the visible PNG image. This prevents repeated VAE encode/decode cycles when reprocessing the same image multiple times, reducing cumulative degradation (“re-baking”).

## Motivation

In most diffusion pipelines:

    pixels → VAE encode → latent
    latent → diffusion → latent
    latent → VAE decode → pixels
    (repeat)

Every additional encode/decode cycle introduces quantization noise and reconstruction bias.

LatentPNG solves this by embedding the latent tensor directly into the PNG file. If present, downstream pipelines can skip VAE encoding entirely and operate directly on the stored latent.

# Design Goals

* 100% PNG-compatible
* Backward compatible with standard image viewers
* Deterministic VAE reference validation
* Minimal file size overhead
* Extensible metadata schema
* No new container format
---

# File Format Overview


## Design Goals

- 100% PNG-compatible
- Backward compatible with standard image viewers
- Deterministic VAE reference validation
- Minimal file size overhead
- Extensible metadata schema
- No new container format


## File Format Overview

LatentPNG uses a standard PNG container and embeds a structured JSON payload inside an `iTXt` metadata chunk.

### PNG Chunk

- **Chunk type:** `iTXt`
- **Keyword:** `LPNG_LATENT`
- **Compression:** optional (recommended)

Image viewers ignore this chunk. Diffusion-aware tools consume it.


## Metadata Schema (v1.0)

```json
{
  "format_version": "1.0",

  "latent": {
    "dtype": "float16",
    "shape": [4, 128, 128],
    "compression": "zlib",
    "encoding": "base64",
    "data": "<compressed_base64_blob>",
    "checksum": "sha256_of_raw_latent_bytes"
  },

  "vae": {
    "model_id": "stabilityai/sd-vae-ft-mse",
    "sha256": "abc123deadbeef...",
    "remote_url": "https://huggingface.co/stabilityai/sd-vae-ft-mse",
    "local_path": "/models/vae/sd-vae-ft-mse.safetensors",
    "scaling_factor": 0.18215
  },

  "generator": {
    "model_id": "stable-diffusion-xl-base-1.0",
    "sha256": "def456cafebabe..."
  }
}
```



## Field Definitions

### `format_version`
String. Current version of the LatentPNG schema.



### `latent`

| Field        | Description |
|-------------|------------|
| `dtype`     | Tensor data type (`float16` recommended) |
| `shape`     | Tensor shape `[C, H, W]` |
| `compression` | Compression algorithm (`zlib` recommended) |
| `encoding`  | Encoding format (`base64`) |
| `data`      | Compressed latent tensor blob |
| `checksum`  | SHA256 of raw tensor bytes (before compression) |



### `vae`

| Field | Description |
|-------|------------|
| `model_id` | Canonical model identifier |
| `sha256` | Full SHA256 hash of model file |
| `remote_url` | Authoritative remote download location |
| `local_path` | Local filesystem reference used during creation |
| `scaling_factor` | VAE latent scaling constant |

All five fields are required.

**Hash verification is mandatory before decoding.**

If the local VAE hash does not match:
- Tooling MUST refuse automatic decoding
- Tooling MAY request explicit override



### `generator` (Optional)

Stores the diffusion backbone used to generate or modify the latent.



## Latent Encoding Specification

Recommended workflow:

1. Convert tensor to `float16`
2. Serialize raw tensor bytes (C-order)
3. Compute SHA256 of raw bytes
4. Compress using `zlib`
5. Base64 encode
6. Embed in metadata



## Loading Behavior

When loading a PNG:

### If `LPNG_LATENT` exists:
1. Parse metadata
2. Validate checksum
3. Verify VAE hash
4. If valid → skip VAE encoding
5. Feed latent directly into model

### If missing:
Fallback to standard:

```
pixels → VAE encode
```



## Advantages

- Eliminates repeated VAE re-encoding
- Reduces generational degradation
- Improves iterative img2img fidelity
- Maintains human-viewable PNG
- Minimal storage overhead (~70–120KB typical)


## Security Considerations

- Always validate SHA256 before decoding latent
- Do not auto-download models without user consent
- Treat metadata as untrusted input


## Recommended Extension

File extension may remain `.png`.

Optional clarity extension:

```
.lpng
```

Both are valid PNG files.


## Reference Implementation (Planned)

- Python encoder/decoder
- Automatic VAE hash validation
- PyTorch tensor reconstruction utilities
- CLI tool: `lpng embed` / `lpng extract`


## Versioning

Future schema changes must:
- Increment `format_version`
- Preserve backward compatibility
- Never redefine existing field semantics

---

## License

MIT
