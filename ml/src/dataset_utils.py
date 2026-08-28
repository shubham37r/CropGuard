import os
import hashlib
from typing import Dict, Any, Tuple, Optional
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 10 Official PlantVillage Tomato Classes
TARGET_TOMATO_CLASSES = [
    "Tomato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

def calculate_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file for exact duplicate detection."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def inspect_single_image(filepath: str) -> Dict[str, Any]:
    """
    Validates and inspects an image file.
    Returns metadata dict or error details if unreadable/corrupt.
    """
    res = {
        "filepath": filepath,
        "valid": False,
        "width": 0,
        "height": 0,
        "format": None,
        "mode": None,
        "file_size_bytes": 0,
        "hash": None,
        "error": None
    }

    try:
        if not os.path.exists(filepath):
            res["error"] = "File does not exist"
            return res

        res["file_size_bytes"] = os.path.getsize(filepath)
        if res["file_size_bytes"] == 0:
            res["error"] = "Zero byte empty file"
            return res

        res["hash"] = calculate_sha256(filepath)

        with Image.open(filepath) as img:
            img.verify()

        with Image.open(filepath) as img:
            width, height = img.size
            if width <= 0 or height <= 0:
                res["error"] = "Invalid non-positive dimensions"
                return res

            res["valid"] = True
            res["width"] = width
            res["height"] = height
            res["format"] = img.format
            res["mode"] = img.mode

    except Exception as err:
        res["valid"] = False
        res["error"] = str(err)

    return res
