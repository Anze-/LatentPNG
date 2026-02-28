import os
import json
import base64
import zlib
import torch
import numpy as np
from PIL import Image, PngImagePlugin, ImageSequence, ImageOps

import folder_paths
from .node_helpers import pillow as pil

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

    RETURN_TYPES =("IMAGE",)  # added LATENT
    ETURN_NAMES = ("image_out",)
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

        return (decoded,)


# ---------------------------------------------------------
# Load LPNG Node
# ---------------------------------------------------------

class LoadLPNG:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required":
                    {
                    "image": (sorted(files), {"image_upload": True}),
                    "vae": ("VAE",),
                    
                    },
                    
                }

    CATEGORY = "image"
    ESSENTIALS_CATEGORY = "Basics"
    SEARCH_ALIASES = [
        "load image", "load latent", "RAW", "open image", "import image", "image input",
        "upload image", "read image", "image loader"
    ]

    RETURN_TYPES = ("IMAGE", "MASK", "LATENT",'info')  # added LATENT
    FUNCTION = "load_image"

    def load_image(self, image, vae):
        image_path = folder_paths.get_annotated_filepath(image)
        img = pil(Image.open, image_path)

        output_images = []
        output_masks = []
        output_latent = 0
        w, h = None, None

        for im in ImageSequence.Iterator(img):
            i = pil(ImageOps.exif_transpose, im)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image_rgb = i.convert("RGB")

            if len(output_images) == 0:
                w, h = image_rgb.size

            if image_rgb.size[0] != w or image_rgb.size[1] != h:
                continue

            image_np = np.array(image_rgb).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]

            # Mask extraction
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
                
            # Latents extraction
            metadata_raw = i.info.get(LPNG_KEYWORD, None)
        
            # If no latent stored → fallback to encode
            if metadata_raw is None:
                info='No latent data found, using VAE! (this should happen only on your first pass)'
                import warnings
                warnings.warn("Warning: the file has empty latents! This should happen only on your first pass")
                output_latent = vae.encode(image_tensor)
                #raise RuntimeError("ERROR: the file is not a valid LPNG file, it contains no latents!")
            else:
                info='OK, latent image data was found!'
                metadata = json.loads(metadata_raw)
                latent_info = metadata["latent"]

                compressed = base64.b64decode(latent_info["data"])
                raw_bytes = zlib.decompress(compressed)

                latent = bytes_to_tensor_fp16(
                    raw_bytes,
                    latent_info["shape"]
                )
                
                output_latent = latent

            output_images.append(image_tensor)
            output_masks.append(mask.unsqueeze(0))

            if img.format == "MPO":
                break  # ignore other frames

        if len(output_images) > 1:
            raise RuntimeError("ERROR: can only handle one image at the time, a batch was received!")
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
            output_latent = {'samples':output_latent}

        return output_image, output_mask, output_latent, info

    @classmethod  
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

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
