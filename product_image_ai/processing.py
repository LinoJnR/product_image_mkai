
"""Image processing for Product Image AI.

Uses U2Net through rembg to create a foreground mask.
Original image pixels are retained wherever the mask identifies
the foreground.
"""

import io
from functools import lru_cache

from PIL import Image, ImageFilter


@lru_cache(maxsize=1)
def _get_rembg_session():
    """Load and cache the U2Net model."""
    try:
        from rembg import new_session

        return new_session("u2net")

    except Exception as exc:
        raise RuntimeError(
            "Could not initialize the background-removal AI model. "
            "Check the internet connection for the first model download, "
            "available memory and model-cache permissions. "
            f"Details: {exc}"
        ) from exc


def create_foreground_mask(image):
    """Generate a foreground mask using U2Net."""

    try:
        from rembg import remove

        original = image.convert("RGBA")

        buffer = io.BytesIO()
        original.save(buffer, format="PNG")

        session = _get_rembg_session()

        mask_bytes = remove(
            buffer.getvalue(),
            only_mask=True,
            session=session,
        )

        with Image.open(io.BytesIO(mask_bytes)) as mask_image:

            mask = mask_image.convert("L")

            if mask.size != original.size:
                mask = mask.resize(
                    original.size,
                    Image.Resampling.LANCZOS,
                )

            # Slightly soften mask edges to reduce harsh cut-outs.
            mask = mask.filter(
                ImageFilter.GaussianBlur(radius=0.5)
            )

            return mask

    except ImportError as exc:
        raise RuntimeError(
            "Background removal is not installed. "
            "Run pip install -r requirements.txt."
        ) from exc

    except RuntimeError:
        raise

    except Exception as exc:
        raise RuntimeError(
            f"Foreground mask generation failed: {exc}"
        ) from exc


def make_white_background(
    image,
    remove_bg=True,
    transparent=False,
):
    """Remove the background and preserve original foreground pixels."""

    original = image.convert("RGBA")

    if remove_bg:

        mask = create_foreground_mask(original)

        foreground = original.copy()
        foreground.putalpha(mask)

    else:
        foreground = original

    if transparent:
        return foreground

    white_background = Image.new(
        "RGBA",
        original.size,
        (255, 255, 255, 255),
    )

    white_background.alpha_composite(foreground)

    return white_background.convert("RGB")


def apply_brush_edits(
    current_result,
    original_image,
    mask,
    action="restore",
):
    """Apply manual Restore or Remove brush corrections.

    Restore uses original image pixels.
    Remove paints the selected area white.
    """

    current = current_result.convert("RGBA")
    original = original_image.convert("RGBA")
    edit_mask = mask.convert("L")

    if original.size != current.size:
        original = original.resize(
            current.size,
            Image.Resampling.LANCZOS,
        )

    if edit_mask.size != current.size:
        edit_mask = edit_mask.resize(
            current.size,
            Image.Resampling.NEAREST,
        )

    if action == "restore":

        return Image.composite(
            original,
            current,
            edit_mask,
        )

    elif action == "remove":

        white = Image.new(
            "RGBA",
            current.size,
            (255, 255, 255, 255),
        )

        return Image.composite(
            white,
            current,
            edit_mask,
        )

    else:
        raise ValueError(
            "Invalid brush action. Use 'restore' or 'remove'."
        )
