"""Safe image ingestion helpers for Product Image AI."""
import io
import zipfile
from pathlib import PurePosixPath

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_ZIP_UNCOMPRESSED_BYTES = 250 * 1024 * 1024
MAX_ZIP_FILES = 500
MAX_IMAGE_PIXELS = 100_000_000

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


def _safe_name(name):
    return PurePosixPath(str(name).replace("\\", "/")).name or "image"


def _validate_image_bytes(data, filename):
    if not data:
        raise ValueError("The file is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"File exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB per-image limit."
        )
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(data)) as image:
            try:
                image.seek(0)
            except EOFError:
                pass
            image.load()
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"Image exceeds the {MAX_IMAGE_PIXELS:,}-pixel safety limit."
                )
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError(f"Image is too large to decode safely: {exc}") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Image exceeds"):
            raise
        raise ValueError(
            "Unsupported or corrupted image. The file could not be decoded."
        ) from exc
    return data


def _unique_name(name, seen):
    path = PurePosixPath(name)
    candidate = path.name
    stem, suffix = path.stem, path.suffix
    counter = 2
    while candidate.casefold() in seen:
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    seen.add(candidate.casefold())
    return candidate


def collect_uploaded_images(files):
    """Return [(unique_source_name, raw_bytes), ...] and individual failures."""
    accepted, failures, seen = [], [], set()
    total = 0

    for upload in files:
        name = _safe_name(getattr(upload, "name", "image"))
        try:
            data = upload.getvalue()
            total += len(data)
            if total > MAX_ZIP_UNCOMPRESSED_BYTES:
                raise ValueError(
                    "Combined uploads exceed the 250 MB batch limit."
                )
            _validate_image_bytes(data, name)
            unique = _unique_name(name, seen)
            accepted.append((unique, data))
        except Exception as exc:
            failures.append((name, str(exc)))

    return accepted, failures


def extract_zip_images(zip_upload):
    """Read images safely from ZIP; never extract paths to disk."""
    accepted, failures, seen = [], [], set()

    try:
        zip_upload.seek(0)
        with zipfile.ZipFile(zip_upload) as archive:
            entries = [item for item in archive.infolist() if not item.is_dir()]
            if len(entries) > MAX_ZIP_FILES:
                raise ValueError(
                    f"ZIP contains more than {MAX_ZIP_FILES} files."
                )

            declared_total = sum(item.file_size for item in entries)
            if declared_total > MAX_ZIP_UNCOMPRESSED_BYTES:
                raise ValueError(
                    "ZIP expands beyond the 250 MB safety limit."
                )

            for info in entries:
                raw = info.filename.replace("\\", "/")
                parts = PurePosixPath(raw).parts
                unix_mode = (info.external_attr >> 16) & 0o170000

                if raw.startswith("/") or ".." in parts or unix_mode == 0o120000:
                    failures.append(
                        (_safe_name(raw), "Unsafe ZIP path; entry skipped.")
                    )
                    continue

                if info.file_size > MAX_IMAGE_BYTES:
                    failures.append(
                        (_safe_name(raw), "File exceeds the 25 MB per-image limit.")
                    )
                    continue

                try:
                    data = archive.read(info)
                    _validate_image_bytes(data, raw)
                    name = _unique_name(_safe_name(raw), seen)
                    accepted.append((name, data))
                except Exception as exc:
                    failures.append((_safe_name(raw), str(exc)))

    except zipfile.BadZipFile as exc:
        raise ValueError(
            "The uploaded ZIP file is invalid or corrupted."
        ) from exc

    return accepted, failures


def open_image_bytes(data):
    """Decode, orient from EXIF, and convert first frame to RGBA."""
    with Image.open(io.BytesIO(data)) as image:
        try:
            image.seek(0)
        except EOFError:
            pass
        image.load()
        oriented = ImageOps.exif_transpose(image)
        return oriented.convert("RGBA")
