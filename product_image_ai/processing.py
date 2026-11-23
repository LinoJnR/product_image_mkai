"""Memory-conscious background processing with cached model initialization."""
import io
from functools import lru_cache

from PIL import Image, ImageDraw


@lru_cache(maxsize=1)
def _get_rembg_session():
    """Initialize the CPU model once; rembg downloads its model on first use."""
    try:
        from rembg import new_session
        return new_session('u2net')
    except Exception as exc:
        raise RuntimeError(
            'Could not initialize the background-removal AI model. Check the internet '
            'connection for the first model download, available memory, and write access '
            'to the model cache, then restart the app. Details: ' + str(exc)
        ) from exc


def make_white_background(image, remove_bg=True, transparent=False):
    """Apply a mask to original RGB pixels and composite on pure white."""
    original = image.convert('RGBA')
    if not remove_bg:
        foreground = original
    else:
        try:
            from rembg import remove
            session = _get_rembg_session()
            source_buffer = io.BytesIO()
            original.save(source_buffer, format='PNG')
            mask_bytes = remove(source_buffer.getvalue(), only_mask=True, session=session)
            with Image.open(io.BytesIO(mask_bytes)) as mask_image:
                mask = mask_image.convert('L')
                if mask.size != original.size:
                    mask = mask.resize(original.size, Image.Resampling.LANCZOS)
                foreground = original.copy()
                foreground.putalpha(mask)
        except ImportError as exc:
            raise RuntimeError(
                "Background removal is not installed. Run 'pip install -r requirements.txt'."
            ) from exc
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f'Background removal failed for this image: {exc}') from exc

    if transparent:
        return foreground
    white = Image.new('RGBA', original.size, (255, 255, 255, 255))
    white.alpha_composite(foreground)
    return white.convert('RGB')


def crop_fraction(image, left=0.0, top=0.0, right=1.0, bottom=1.0):
    """Crop an image using fractional bounds (0.0-1.0) of its width/height.

    Useful for isolating one product panel out of a composite/multi-panel
    screenshot (e.g. a marketplace listing image that stitches a studio shot
    and a lifestyle shot side by side) before background removal runs.
    Fractions are clamped and ordered defensively so a bad preset/slider
    value cannot produce a zero-size or inverted crop.
    """
    width, height = image.size
    left = min(max(float(left), 0.0), 1.0)
    right = min(max(float(right), 0.0), 1.0)
    top = min(max(float(top), 0.0), 1.0)
    bottom = min(max(float(bottom), 0.0), 1.0)
    if right <= left:
        left, right = 0.0, 1.0
    if bottom <= top:
        top, bottom = 0.0, 1.0
    box = (
        round(left * width),
        round(top * height),
        max(round(right * width), round(left * width) + 1),
        max(round(bottom * height), round(top * height) + 1),
    )
    return image.crop(box)


def remove_disconnected_overlays(image, close_kernel=9, dilate_kernel=5, brightness_threshold=240):
    """Strip dimension arrows, measurement labels, and other callout overlays
    that sit in the plain background around a product (common in marketplace
    screenshots that annotate width/height/depth with arrows and text).

    These overlays are disconnected from the product in pixel space. This
    isolates the single largest connected dark shape (the product) and
    whitens every other separate shape. A product with disconnected parts in
    the photo itself (e.g. a two-piece item shot with a visible gap) will
    lose the smaller piece, so this is meant to run before background
    removal, as an optional pre-cleaning step, not a universal default.
    """
    import numpy as np
    import cv2

    rgb = image.convert('RGB')
    arr = np.array(rgb)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    mask = (gray < brightness_threshold).astype(np.uint8)

    if close_kernel > 0:
        kernel = np.ones((close_kernel, close_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return rgb  # nothing non-white was detected; leave the image untouched

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    keep = (labels == largest_label).astype(np.uint8)

    if dilate_kernel > 0:
        keep = cv2.dilate(keep, np.ones((dilate_kernel, dilate_kernel), np.uint8), iterations=1)

    out = arr.copy()
    out[keep == 0] = [255, 255, 255]
    return Image.fromarray(out)


def erase_boxes(image, boxes):
    """Paint one or more fractional rectangles pure white.

    Each box in ``boxes`` is ``(left, top, right, bottom)`` as 0.0-1.0
    fractions of width/height, same convention as crop_fraction. This is the
    manual fallback for leftover arrows, labels, or ghost residue that
    remove_disconnected_overlays could not safely separate from the product
    (e.g. an annotation whose pixels visually touch the product edge, or a
    faint anti-aliased halo left behind by a morphological close/dilate).
    Run it after crop_fraction / remove_disconnected_overlays and before
    background removal. Invalid boxes (inverted or zero-size) are skipped
    rather than raising, so a stray UI input cannot blank the whole image.
    """
    width, height = image.size
    out = image.convert('RGB').copy()
    draw = ImageDraw.Draw(out)
    for left, top, right, bottom in boxes:
        left = min(max(float(left), 0.0), 1.0)
        right = min(max(float(right), 0.0), 1.0)
        top = min(max(float(top), 0.0), 1.0)
        bottom = min(max(float(bottom), 0.0), 1.0)
        if right <= left or bottom <= top:
            continue
        box = (
            round(left * width),
            round(top * height),
            round(right * width),
            round(bottom * height),
        )
        draw.rectangle(box, fill=(255, 255, 255))
    return out
