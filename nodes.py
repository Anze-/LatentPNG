import os
import json
import base64
import zlib
import torch
import numpy as np
from PIL import Image, PngImagePlugin

import folder_paths

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

def tensor_to_pil(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()

    # Remove batch dimension if present
    if tensor.ndim == 4:
        tensor = tensor[0]

    # Clamp and convert
    tensor = tensor.clamp(0, 1)
    array = (tensor.numpy() * 255).astype(np.uint8)

    return Image.fromarray(array)
    
def pil_to_tensor(image):
    # Ensure RGB
    image = image.convert("RGB")

    array = np.array(image).astype(np.float32) / 255.0

    tensor = torch.from_numpy(array)

    # Add batch dimension → (1, H, W, C)
    tensor = tensor.unsqueeze(0)

    return tensor

from datetime import datetime

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
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
        compressed = zlib.compress(raw_bytes)
        encoded = base64.b64encode(compressed).decode("utf-8")

        metadata = {
            "format_version": FORMAT_VERSION,
            "latent": {
                "dtype": "float16",
                "shape": list(latent_tensor.shape),
                "compression": "zlib",
                "encoding": "base64",
                "data": encoded,
            }
        }

        # Decode once to create PNG preview
        with torch.no_grad():
            decoded = vae.decode(latent_tensor)

        image = tensor_to_pil(decoded[0])

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_itxt(LPNG_KEYWORD, json.dumps(metadata))

        file_name = f"{file_prefix}_{get_timestamp()}.png"
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
        image_tensor = pil_to_tensor(img)

        metadata_raw = img.info.get(LPNG_KEYWORD, None)

        # If no latent stored → fallback to encode
        if metadata_raw is None:
            with torch.no_grad():
                latent = vae.encode(image_tensor)
            return (image_tensor, {"samples": latent})

        metadata = json.loads(metadata_raw)
        latent_info = metadata["latent"]

        compressed = base64.b64decode(latent_info["data"])
        raw_bytes = zlib.decompress(compressed)

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
