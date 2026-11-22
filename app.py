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

st.set_page_config(page_title="Product Image AI", page_icon="🖼️", layout="wide")
st.title("🖼️ Product Image AI")
st.caption("Bulk product-image background cleaning and JPG export.")

st.info(
    "Upload image files or a ZIP. Images are processed one at a time, EXIF orientation "
    "is corrected, and the first frame of animated images is used."
)
st.warning(
    "This version performs general foreground/background separation. It does not yet "
    "identify a specific product separately from a person, pet, food, cable, or other "
    "foreground object, and it cannot reconstruct hidden product details or empty a "
    "transparent container."
)

left, right = st.columns(2)
with left:
    image_files = st.file_uploader(
        "Upload multiple product images", type=None,
        accept_multiple_files=True, key="bulk_images"
    )
with right:
    zip_file = st.file_uploader(
        "Or upload a ZIP containing product images",
        type=["zip"], key="image_zip"
    )

st.caption(
    "Limits: 25 MB per image, 250 MB combined uncompressed batch, "
    "500 files per ZIP, and 100 million pixels per image. Small images are not rejected."
)

if st.button("Clean backgrounds", type="primary", use_container_width=True):
    uploaded, failures = collect_uploaded_images(image_files or [])

    if zip_file is not None:
        try:
            zip_images, zip_failures = extract_zip_images(zip_file)
            failures.extend(zip_failures)
            used = {name.casefold() for name, _ in uploaded}
            for name, data in zip_images:
                candidate = name
                stem, suffix = Path(name).stem, Path(name).suffix
                counter = 2
                while candidate.casefold() in used:
                    candidate = f"{stem}_{counter}{suffix}"
                    counter += 1
                used.add(candidate.casefold())
                uploaded.append((candidate, data))
        except Exception as exc:
            st.error(f"Could not read ZIP: {exc}")
            st.stop()

    combined_bytes = sum(len(data) for _, data in uploaded)
    if combined_bytes > MAX_ZIP_UNCOMPRESSED_BYTES:
        st.error(
            "Combined direct uploads and ZIP images exceed the 250 MB batch limit. "
            "Reduce the batch size and try again."
        )
        st.stop()

    if not uploaded:
        st.warning("No valid images were found. Check the individual errors below.")
        st.session_state["processed_images"] = []
        st.session_state["processing_failures"] = failures
    else:
        st.write(f"**{len(uploaded):,} images ready for processing.**")
        progress = st.progress(0)
        status = st.empty()
        processed = []
        output_names = set()

        for index, (filename, image_data) in enumerate(uploaded, start=1):
            status.write(f"Processing {index:,} of {len(uploaded):,}: {filename}")
            try:
                image = open_image_bytes(image_data)
                original_size = image.size
                result = make_white_background(
                    image, remove_bg=True, transparent=False
                )
                output = io.BytesIO()
                result.save(output, format="JPEG", quality=95, optimize=True)

                stem = Path(filename).stem or f"image_{index}"
                output_name = f"{stem}.MAIN.jpg"
                base = stem
                counter = 2
                while output_name.casefold() in output_names:
                    output_name = f"{base}_{counter}.MAIN.jpg"
                    counter += 1
                output_names.add(output_name.casefold())

                processed.append(
                    (output_name, output.getvalue(), filename, original_size, result.size)
                )
                del image, result, output
            except Exception as exc:
                failures.append((filename, str(exc)))
            finally:
                progress.progress(index / len(uploaded))
                gc.collect()

        status.empty()
        st.session_state["processed_images"] = processed
        st.session_state["processing_failures"] = failures

if "processed_images" in st.session_state:
    processed = st.session_state["processed_images"]
    failures = st.session_state.get("processing_failures", [])

    st.subheader("Processing results")
    good_col, bad_col = st.columns(2)
    good_col.metric("Images processed", len(processed))
    bad_col.metric("Needs attention", len(failures))

    if processed:
        st.success("Processed images are ready to preview and export.")
        cols = st.columns(4)
        for i, (output_name, data, source_name, original_size, final_size) in enumerate(
            processed[:8]
        ):
            with cols[i % 4]:
                st.image(data, caption=output_name, use_container_width=True)
                st.caption(f"{original_size[0]} × {original_size[1]} px")

        if len(processed) > 8:
            st.caption(
                f"Showing 8 previews. All {len(processed)} images are included in the ZIP."
            )

        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for output_name, data, source_name, original_size, final_size in processed:
                zf.writestr(f"OUTPUT/IMAGES/MAIN/{output_name}", data)
        archive.seek(0)

        st.download_button(
            "Download cleaned MAIN images",
            data=archive.getvalue(),
            file_name="Product_Image_AI_MAIN_Images.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )

    if failures:
        with st.expander(f"Files that need attention ({len(failures)})"):
            for filename, error in failures:
                st.write(f"**{filename}** — {error}")

st.divider()
st.caption(
    "The foreground mask is applied to the original image pixels. Output dimensions "
    "match the source image. Review every result, especially transparent, reflective, "
    "white, and intricate products."
)
