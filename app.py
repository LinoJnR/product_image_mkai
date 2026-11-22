
import gc
import io
import zipfile
from pathlib import Path

import streamlit as st

from product_image_ai.ingest import (
    MAX_ZIP_UNCOMPRESSED_BYTES,
    collect_uploaded_images,
    extract_zip_images,
    open_image_bytes,
)
from product_image_ai.processing import make_white_background


st.set_page_config(
    page_title="Product Image AI",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️ Product Image AI")
st.caption("Batch product-image cleanup and JPG export.")

st.info(
    "Upload individual images or a ZIP file. Images are processed one at a time, "
    "EXIF orientation is corrected, and the first frame of animated images is used."
)

with st.expander("Current AI capability", expanded=False):
    st.write(
        "The connected processing module currently performs general background "
        "removal. It does not yet reliably separate a product from a person, hand, "
        "pet, food, or another object in the foreground. Product-only cleanup will "
        "be enabled after the AI processing module is upgraded."
    )

upload_col, zip_col = st.columns(2)

with upload_col:
    image_files = st.file_uploader(
        "Upload product images",
        type=None,
        accept_multiple_files=True,
        key="bulk_images",
        help="You can select multiple image files at once.",
    )

with zip_col:
    zip_file = st.file_uploader(
        "Or upload a ZIP of product images",
        type=["zip"],
        key="image_zip",
    )

st.caption(
    "Upload limits: 25 MB per image, 250 MB combined batch, 500 files per ZIP, "
    "and 100 million pixels per image. Small images are not rejected."
)


def build_output_zip(processed_images):
    """Create the required MAIN-image folder structure in memory."""
    archive_buffer = io.BytesIO()

    with zipfile.ZipFile(
        archive_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for output_name, image_bytes, *_ in processed_images:
            archive.writestr(
                f"OUTPUT/IMAGES/MAIN/{output_name}",
                image_bytes,
            )

    archive_buffer.seek(0)
    return archive_buffer.getvalue()


def collect_all_uploads(direct_files, uploaded_zip):
    """Combine direct uploads and ZIP entries while keeping names unique."""
    uploaded, failures = collect_uploaded_images(direct_files or [])

    if uploaded_zip is not None:
        zip_images, zip_failures = extract_zip_images(uploaded_zip)
        failures.extend(zip_failures)

        used_names = {name.casefold() for name, _ in uploaded}

        for name, image_bytes in zip_images:
            candidate = name
            stem = Path(name).stem
            suffix = Path(name).suffix
            counter = 2

            while candidate.casefold() in used_names:
                candidate = f"{stem}_{counter}{suffix}"
                counter += 1

            used_names.add(candidate.casefold())
            uploaded.append((candidate, image_bytes))

    total_bytes = sum(len(image_bytes) for _, image_bytes in uploaded)

    if total_bytes > MAX_ZIP_UNCOMPRESSED_BYTES:
        raise ValueError(
            "Combined direct uploads and ZIP images exceed the 250 MB batch limit. "
            "Please reduce the batch size and try again."
        )

    return uploaded, failures


if st.button(
    "Process images",
    type="primary",
    use_container_width=True,
    disabled=not image_files and zip_file is None,
):
    try:
        uploaded, failures = collect_all_uploads(image_files, zip_file)

    except Exception as exc:
        st.error(f"Could not prepare the uploaded files: {exc}")
        st.stop()

    if not uploaded:
        st.session_state["processed_images"] = []
        st.session_state["processing_failures"] = failures
        st.warning("No valid images were found. Check the errors below.")

    else:
        st.write(f"**{len(uploaded):,} image(s) ready for processing.**")

        progress = st.progress(0)
        status = st.empty()

        processed = []
        output_names = set()

        for index, (filename, image_data) in enumerate(uploaded, start=1):

            status.write(
                f"Processing image {index:,} of {len(uploaded):,}: {filename}"
            )

            image = None
            result = None
            output_buffer = None

            try:
                image = open_image_bytes(image_data)
                original_size = image.size

                result = make_white_background(
                    image,
                    remove_bg=True,
                    transparent=False,
                )

                output_buffer = io.BytesIO()

                result.convert("RGB").save(
                    output_buffer,
                    format="JPEG",
                    quality=95,
                    optimize=True,
                )

                stem = Path(filename).stem or f"image_{index}"
                output_name = f"{stem}.MAIN.jpg"
                counter = 2

                while output_name.casefold() in output_names:
                    output_name = f"{stem}_{counter}.MAIN.jpg"
                    counter += 1

                output_names.add(output_name.casefold())

                processed.append(
                    (
                        output_name,
                        output_buffer.getvalue(),
                        filename,
                        original_size,
                        result.size,
                    )
                )

            except Exception as exc:
                failures.append((filename, str(exc)))

            finally:
                if output_buffer is not None:
                    output_buffer.close()

                del image, result, output_buffer

                progress.progress(index / len(uploaded))
                gc.collect()

        status.empty()

        st.session_state["processed_images"] = processed
        st.session_state["processing_failures"] = failures


if "processed_images" in st.session_state:

    processed = st.session_state.get("processed_images", [])
    failures = st.session_state.get("processing_failures", [])

    st.divider()
    st.subheader("Processing results")

    result_col, error_col = st.columns(2)

    result_col.metric("Images processed", len(processed))
    error_col.metric("Files needing attention", len(failures))

    if processed:

        st.success("Your processed images are ready to preview and export.")

        preview_count = min(len(processed), 8)
        preview_cols = st.columns(4)

        for index, item in enumerate(processed[:preview_count]):

            output_name, image_bytes, source_name, original_size, final_size = item

            with preview_cols[index % 4]:

                st.image(
                    image_bytes,
                    caption=output_name,
                    use_container_width=True,
                )

                st.caption(
                    f"{final_size[0]} × {final_size[1]} px"
                )

        if len(processed) > preview_count:
            st.caption(
                f"Showing {preview_count} previews. "
                f"All {len(processed):,} images are included in the ZIP."
            )

        archive_bytes = build_output_zip(processed)

        st.download_button(
            label="Download cleaned MAIN images",
            data=archive_bytes,
            file_name="Product_Image_AI_MAIN_Images.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )

    if failures:

        with st.expander(f"Files needing attention ({len(failures)})"):

            for filename, error in failures:
                st.write(f"**{filename}** — {error}")


st.divider()

st.caption(
    "Output is saved as JPG at quality 95 and retains the source image dimensions. "
    "Please inspect results, especially transparent, reflective, white, or detailed "
    "products. Product-only removal of people, pets, hands, and food requires the "
    "processing module upgrade."
)
