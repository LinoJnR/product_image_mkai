import gc
import io
import zipfile
from pathlib import Path

from PIL import Image
import streamlit as st
from product_image_ai.ingest import MAX_ZIP_UNCOMPRESSED_BYTES, collect_uploaded_images, extract_zip_images, open_image_bytes
from product_image_ai.processing import crop_fraction, make_white_background, remove_disconnected_overlays

st.set_page_config(page_title='Product Image AI', page_icon='🖼️', layout='wide')
st.title('🖼️ Product Image AI')
st.caption('Bulk product-image background cleaning. Upload images, make the background pure white, and export.')
st.info(
    'Upload image files in formats supported by the installed decoders. The app processes images one at a time, '
    'uses the first frame of animated files, corrects EXIF orientation, and exports JPG.'
)

left, right = st.columns(2)
with left:
    image_files = st.file_uploader('Upload multiple product images', type=None, accept_multiple_files=True, key='bulk_images')
with right:
    zip_file = st.file_uploader('Or upload a ZIP containing product images', type=['zip'], key='image_zip')
st.caption('Safety limits: 25 MB per image, 250 MB total uncompressed batch, and 500 files per ZIP. Small images are not rejected or discarded.')

with st.expander('Crop composite / multi-panel screenshots before cleaning (optional)'):
    st.caption(
        'Use this when a source image is a composite screenshot (e.g. a studio shot stitched next to a lifestyle '
        'scene, or a UI panel with a share icon). The same crop is applied to every image in this batch before '
        'background removal, so it works best when all uploaded images share the same layout.'
    )
    crop_enabled = st.checkbox('Crop every image before cleaning', value=False, key='crop_enabled')
    preset = st.radio(
        'Preset',
        ['Full image', 'Left half', 'Right half', 'Top half', 'Bottom half', 'Custom'],
        horizontal=True,
        key='crop_preset',
        disabled=not crop_enabled,
    )
    preset_boxes = {
        'Full image': (0.0, 0.0, 100.0, 100.0),
        'Left half': (0.0, 0.0, 50.0, 100.0),
        'Right half': (50.0, 0.0, 100.0, 100.0),
        'Top half': (0.0, 0.0, 100.0, 50.0),
        'Bottom half': (0.0, 50.0, 100.0, 100.0),
    }
    default_box = preset_boxes.get(preset, (0.0, 0.0, 100.0, 100.0))
    c1, c2, c3, c4 = st.columns(4)
    crop_left = c1.number_input('Left %', 0.0, 100.0, default_box[0], step=1.0, key='crop_left', disabled=not crop_enabled or preset != 'Custom')
    crop_top = c2.number_input('Top %', 0.0, 100.0, default_box[1], step=1.0, key='crop_top', disabled=not crop_enabled or preset != 'Custom')
    crop_right = c3.number_input('Right %', 0.0, 100.0, default_box[2], step=1.0, key='crop_right', disabled=not crop_enabled or preset != 'Custom')
    crop_bottom = c4.number_input('Bottom %', 0.0, 100.0, default_box[3], step=1.0, key='crop_bottom', disabled=not crop_enabled or preset != 'Custom')
    if preset != 'Custom':
        crop_left, crop_top, crop_right, crop_bottom = default_box
    crop_box_fraction = (crop_left / 100.0, crop_top / 100.0, crop_right / 100.0, crop_bottom / 100.0)

    preview_source = None
    if image_files:
        preview_source = image_files[0]
    if crop_enabled and preview_source is not None:
        try:
            preview_bytes = preview_source.getvalue()
            with Image.open(io.BytesIO(preview_bytes)) as preview_image:
                preview_image.load()
                preview_cropped = crop_fraction(preview_image.convert('RGB'), *crop_box_fraction)
            st.caption(f'Preview crop on "{preview_source.name}" (applies to every image in the batch):')
            st.image(preview_cropped, use_container_width=False, width=260)
        except Exception as exc:
            st.warning(f'Could not preview the crop: {exc}')
    elif crop_enabled:
        st.caption('Upload images above to see a live preview of this crop.')

with st.expander('Remove dimension arrows / callout overlays before cleaning (optional)'):
    st.caption(
        'Use this for marketplace screenshots with measurement arrows and labels (e.g. "50cm", "12cm") drawn over '
        'the plain background. It keeps only the single largest connected dark shape in the photo and whitens '
        'everything else. Known limits: an arrow/label that visually touches the product edge may survive as part '
        'of that shape (needs manual cleanup); a product shot as naturally separate pieces may lose the smaller '
        'piece. Always check the preview below before running the full batch.'
    )
    overlay_enabled = st.checkbox('Remove disconnected overlay graphics before cleaning', value=False, key='overlay_enabled')
    overlay_sensitivity = st.slider(
        'Overlay detection sensitivity (higher closes larger gaps in thin product parts)',
        min_value=3, max_value=25, value=9, step=2, key='overlay_sensitivity', disabled=not overlay_enabled,
    )
    if overlay_enabled and image_files:
        try:
            preview_bytes = image_files[0].getvalue()
            with Image.open(io.BytesIO(preview_bytes)) as preview_image:
                preview_image.load()
                preview_rgb = preview_image.convert('RGB')
                if st.session_state.get('crop_enabled'):
                    preview_rgb = crop_fraction(preview_rgb, *crop_box_fraction)
                preview_overlay = remove_disconnected_overlays(preview_rgb, close_kernel=overlay_sensitivity)
            st.caption(f'Preview on "{image_files[0].name}":')
            st.image(preview_overlay, use_container_width=False, width=260)
        except Exception as exc:
            st.warning(f'Could not preview the overlay removal: {exc}')
    elif overlay_enabled:
        st.caption('Upload images above to see a live preview.')

if st.button('Clean backgrounds', type='primary', use_container_width=True):
    uploaded, failures = collect_uploaded_images(image_files or [])
    if zip_file is not None:
        try:
            zip_images, zip_failures = extract_zip_images(zip_file)
            failures.extend(zip_failures)
            # Ensure unique source names across direct uploads and ZIP entries.
            used = {name.casefold() for name, _ in uploaded}
            for name, data in zip_images:
                candidate = name
                stem, suffix = Path(name).stem, Path(name).suffix
                counter = 2
                while candidate.casefold() in used:
                    candidate = f'{stem}_{counter}{suffix}'
                    counter += 1
                used.add(candidate.casefold())
                uploaded.append((candidate, data))
        except Exception as exc:
            st.error(f'Could not read ZIP: {exc}')
            st.stop()

    combined_bytes = sum(len(data) for _, data in uploaded)
    if combined_bytes > MAX_ZIP_UNCOMPRESSED_BYTES:
        st.error('Combined direct uploads and ZIP images exceed the 250 MB batch limit. Reduce the batch size and try again.')
        st.stop()

    if not uploaded:
        st.warning('No valid images were found. Check the individual errors below.')
        st.session_state['processed_images'] = []
        st.session_state['processing_failures'] = failures
    else:
        st.write(f'**{len(uploaded):,} images ready for processing.**')
        progress = st.progress(0)
        status = st.empty()
        processed = []
        output_names = set()
        for index, (filename, image_data) in enumerate(uploaded, start=1):
            status.write(f'Processing {index:,} of {len(uploaded):,}: {filename}')
            try:
                image = open_image_bytes(image_data)
                original_size = image.size
                if st.session_state.get('crop_enabled'):
                    image = crop_fraction(image, *crop_box_fraction)
                if st.session_state.get('overlay_enabled'):
                    image = remove_disconnected_overlays(image, close_kernel=overlay_sensitivity).convert('RGBA')
                result = make_white_background(image, remove_bg=True, transparent=False)
                output = io.BytesIO()
                result.save(output, format='JPEG', quality=95, optimize=True)
                stem = Path(filename).stem or f'image_{index}'
                output_name = f'{stem}.MAIN.jpg'
                base = stem
                counter = 2
                while output_name.casefold() in output_names:
                    output_name = f'{base}_{counter}.MAIN.jpg'
                    counter += 1
                output_names.add(output_name.casefold())
                processed.append((output_name, output.getvalue(), filename, original_size, result.size))
                del image, result, output
            except Exception as exc:
                failures.append((filename, str(exc)))
            finally:
                progress.progress(index / len(uploaded))
                gc.collect()
        status.empty()
        st.session_state['processed_images'] = processed
        st.session_state['processing_failures'] = failures

if 'processed_images' in st.session_state:
    processed = st.session_state['processed_images']
    failures = st.session_state.get('processing_failures', [])
    st.subheader('Processing results')
    good_col, bad_col = st.columns(2)
    good_col.metric('Backgrounds cleaned', len(processed))
    bad_col.metric('Needs attention', len(failures))
    if processed:
        st.success('Processed images are ready to preview and export.')
        cols = st.columns(4)
        for i, (output_name, data, source_name, original_size, final_size) in enumerate(processed[:8]):
            with cols[i % 4]:
                st.image(data, caption=output_name, use_container_width=True)
                st.caption(f'{original_size[0]} × {original_size[1]} px')
        if len(processed) > 8:
            st.caption(f'Showing 8 previews. All {len(processed)} images are included in the export.')
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zf:
            for output_name, data, source_name, original_size, final_size in processed:
                zf.writestr(f'OUTPUT/IMAGES/MAIN/{output_name}', data)
        archive.seek(0)
        st.download_button('Download cleaned MAIN images', data=archive.getvalue(), file_name='Product_Image_AI_MAIN_Images.zip', mime='application/zip', type='primary', use_container_width=True)
    if failures:
        with st.expander(f'Files that need attention ({len(failures)})'):
            for filename, error in failures:
                st.write(f'**{filename}** — {error}')

st.divider()
st.caption('Product preservation: the background-removal mask is applied to original product pixels. Output remains at the source dimensions; small images are processed without being rejected. Inspect previews, especially transparent, reflective, white, or intricate products.')
