import base64
from pathlib import Path
from PIL import Image
import io
from utils.logger import get_logger, CardioImageError

logger = get_logger(__name__)

SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
MAX_SIZE_MB = 10

def load_and_encode_image(image_source) -> tuple[str, str]:
    """
    Accept a file path or Streamlit UploadedFile.
    Returns (base64_encoded_string, media_type)
    """
    try:
        if hasattr(image_source, 'read'):
            # Streamlit UploadedFile
            image_bytes = image_source.getvalue() if hasattr(image_source, "getvalue") else image_source.read()
            media_type = image_source.type or "image/jpeg"
        else:
            # File path
            path = Path(image_source)
            if path.suffix.lower() not in SUPPORTED_FORMATS:
                raise CardioImageError(f"Unsupported format: {path.suffix}")
            with open(path, "rb") as f:
                image_bytes = f.read()
            media_type = f"image/{path.suffix.lower().strip('.')}"

        # Check size
        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > MAX_SIZE_MB:
            raise CardioImageError(f"Image too large: {size_mb:.1f}MB (max {MAX_SIZE_MB}MB)")

        encoded = base64.b64encode(image_bytes).decode("utf-8")
        logger.info(f"Image encoded successfully ({size_mb:.2f}MB)")
        return encoded, media_type

    except CardioImageError:
        raise
    except Exception as e:
        raise CardioImageError(f"Failed to process image: {e}")

# Vision models gain nothing past ~1568px on the long side; bigger images
# only add upload time and latency (a phone photo can be 4000px+)
MAX_VISION_EDGE = 1568

def prepare_for_vision(image_source) -> tuple[str, str]:
    """
    Load an image, shrink it to MAX_VISION_EDGE, and re-encode as JPEG.
    JPEG is accepted by every vision provider (BMP is not).
    Returns (base64_encoded_string, media_type)
    """
    encoded, _ = load_and_encode_image(image_source)
    try:
        image = Image.open(io.BytesIO(base64.b64decode(encoded)))
        image = image.convert("RGB")
        original = image.size
        image.thumbnail((MAX_VISION_EDGE, MAX_VISION_EDGE))

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        logger.info(f"Prepared image for vision: {original} -> {image.size}, "
                    f"{len(buffer.getvalue()) / 1024:.0f}KB")
        return base64.b64encode(buffer.getvalue()).decode("utf-8"), "image/jpeg"
    except Exception as e:
        raise CardioImageError(f"Failed to process image: {e}")
