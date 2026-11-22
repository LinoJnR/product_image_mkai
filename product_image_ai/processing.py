"""Memory-conscious background processing with cached model initialization."""
import io
from functools import lru_cache

from PIL import Image


@lru_cache(maxsize=1)
def _get_rembg_session():
    """Initialize the CPU model once; rembg downloads its model on first use."""
    try:
        from rembg import new_session
        return new_session("u2net")
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize the background-removal AI model. Check the internet "
            "connection for the first model download, available memory, and write access "
            "to the model cache, then restart the app. Details: " + str(exc)
        ) from exc


def make_white_background(image, remove_bg=True, transparent=False):
    """Apply a foreground mask to original pixels and composite on white."""
    original = image.convert("RGBA")

    if not remove_bg:
        foreground = original
    else:
        try:
            from rembg import remove

            session = _get_rembg_session()
            source_buffer = io.BytesIO()
            original.save(source_buffer, format="PNG")
            mask_bytes = remove(
                source_buffer.getvalue(), only_mask=True, session=session
            )

            with Image.open(io.BytesIO(mask_bytes)) as mask_image:
                mask = mask_image.convert("L")
                if mask.size != original.size:
                    mask = mask.resize(
                        original.size, Image.Resampling.LANCZOS
                    )
                foreground = original.copy()
                foreground.putalpha(mask)
        except ImportError as exc:
            raise RuntimeError(
                "Background removal is not installed. "
                "Run 'pip install -r requirements.txt'."
            ) from exc
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Background removal failed for this image: {exc}"
            ) from exc

    if transparent:
        return foreground

    white = Image.new("RGBA", original.size, (255, 255, 255, 255))
    white.alpha_composite(foreground)
    return white.convert("RGB")
