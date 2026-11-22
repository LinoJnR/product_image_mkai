# Product Image AI

A Streamlit application for batch background removal and JPG export.

## Included
- Multiple image uploads or one ZIP upload
- Safe ZIP reading without extracting paths to disk
- U2Net foreground/background masking through `rembg`
- White-background JPG output at source dimensions
- JPG quality 95
- ZIP export to `OUTPUT/IMAGES/MAIN/`
- HEIF/HEIC decoding when the optional decoder installs successfully

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push the contents of this folder to a GitHub repository.
2. Select `app.py` as the main file.
3. Deploy. Streamlit installs packages from `requirements.txt`.

## Important capability limitation
This version performs general foreground segmentation. It does not provide product-aware
selection, person/pet removal while preserving an item held or worn, object removal from
inside transparent containers, or inpainting of hidden product details. U2Net may include
unwanted foreground objects in its mask or omit reflective/transparent product parts.
Those advanced functions require additional segmentation, user-guided mask editing, and
inpainting models; they are not implemented by the current U2Net pipeline. Always inspect
the exported images.
