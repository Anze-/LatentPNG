from .nodes import LPNG

# ComfyUI: Node class mappings

NODE_CLASS_MAPPINGS = {
    "Load LPNG": LoadLPNG,
    "Save LPNG": SaveLPNG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load LPNG": "Load LPNG",
    "Save LPNG": "Save LPNG",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
