import os
import json
import base64
import zlib
import hashlib
import struct
import torch
import numpy as np
from PIL import Image, PngImagePlugin

import folder_paths
import comfy.utils
import comfy.model_management

LPNG_KEYWORD = "LPNG_LATENT"
FORMAT_VERSION = "1.0"


# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------

def tensor_to_bytes_fp16(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu().to(torch.float16).contiguous()
    return tensor.numpy().tobytes(order="C")


def bytes_to_tensor_fp16(data: bytes, shape):
    arr = np.frombuffer(data, dtype=np.float16)
    arr = arr.reshape(shape)
    return torch.from_numpy(arr).to(torch.float32)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_vae_info(vae):
    """
    Extract VAE file path and hash.
    Works with standard ComfyUI loaded VAE objects.
    """
    if not hasattr(vae, "vae_path"):
        raise RuntimeError("VAE object does not expose vae_path attribute.")

    path = vae.vae_path
    if not os.path.exists(path):
        raise RuntimeError(f"VAE path does not exist: {path}")

    return {
        "model_id": os.path.basename(path),
        "sha256": get_file_sha256(path),
        "local_path": path,
        "remote_url": "",
        "scaling_factor": getattr(vae, "scaling_factor", 1.0)
    }


# ---------------------------------------------------------
# Save LPNG Node
# ---------------------------------------------------------

class SaveLPNG:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_prefix": ("STRING", {"default": "lpng"}),
                "latents": ("LATENT",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "LatentPNG"

    def save(self, file_prefix, latents, vae):

        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        latent_tensor = latents["samples"]

        # Serialize latent
        raw_bytes = tensor_to_bytes_fp16(latent_tensor)
        checksum = sha256_bytes(raw_bytes)
        compressed = zlib.compress(raw_bytes)
        encoded = base64.b64encode(compressed).decode("utf-8")

        vae_info = get_vae_info(vae)

        metadata = {
            "format_version": FORMAT_VERSION,
            "latent": {
                "dtype": "float16",
                "shape": list(latent_tensor.shape),
                "compression": "zlib",
                "encoding": "base64",
                "data": encoded,
                "checksum": checksum
            },
            "vae": vae_info
        }

        # Create preview image (decode once)
        with torch.no_grad():
            decoded = vae.decode(latent_tensor)
        image = comfy.utils.tensor2pil(decoded[0])

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_itxt(LPNG_KEYWORD, json.dumps(metadata))

        file_name = f"{file_prefix}_{comfy.utils.get_timestamp()}.png"
        file_path = os.path.join(output_dir, file_name)

        image.save(file_path, pnginfo=pnginfo)

        return ()


# ---------------------------------------------------------
# Load LPNG Node
# ---------------------------------------------------------

class LoadLPNG:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "choose_path": ("STRING", {"default": ""}),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latents")
    FUNCTION = "load"
    CATEGORY = "LatentPNG"

    def load(self, choose_path, vae):

        if not os.path.exists(choose_path):
            raise RuntimeError("File does not exist.")

        img = Image.open(choose_path)

        metadata_raw = img.info.get(LPNG_KEYWORD, None)

        # Always provide pixel image
        image_tensor = comfy.utils.pil2tensor(img)

        if metadata_raw is None:
            # Fallback: encode normally
            with torch.no_grad():
                latent = vae.encode(image_tensor)
            return (image_tensor, {"samples": latent})

        metadata = json.loads(metadata_raw)

        # Validate VAE match
        stored_vae = metadata["vae"]
        current_vae_info = get_vae_info(vae)

        if stored_vae["sha256"] != current_vae_info["sha256"]:
            # Fallback to encode if mismatch
            with torch.no_grad():
                latent = vae.encode(image_tensor)
            return (image_tensor, {"samples": latent})

        # Load latent directly
        latent_info = metadata["latent"]
        compressed = base64.b64decode(latent_info["data"])
        raw_bytes = zlib.decompress(compressed)

        # Verify checksum
        if sha256_bytes(raw_bytes) != latent_info["checksum"]:
            raise RuntimeError("Latent checksum mismatch.")

        latent_tensor = bytes_to_tensor_fp16(
            raw_bytes,
            latent_info["shape"]
        )

        return (image_tensor, {"samples": latent_tensor})


# ---------------------------------------------------------
# Node Registration
# ---------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "Load LPNG": LoadLPNG,
    "Save LPNG": SaveLPNG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load LPNG": "Load LPNG",
    "Save LPNG": "Save LPNG",
}
