import streamlit as st
import io
from streamlit_image_comparison import image_comparison
import pix
import yuv
import numpy as np
from PIL import Image

@st.cache_data
def cached_pix(image_bytes, ds_factor, target_width, target_height, operations, naive_method) -> Image:
    """Caching wrapper for Pix algorithm using pix.py main instead of naive_downsample."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    # Use target_width/target_height if provided (non-zero), otherwise scale by ds_factor
    result_img = pix.main(
        img,
        ds_factor=ds_factor,
        target_width=target_width if target_width > 0 else None,
        target_height=target_height if target_height > 0 else None,
        operations=operations,
        naive_method=naive_method
    )
    return result_img

@st.cache_data
def cache_naive(image_bytes, ds_factor, naive_method, target_width, target_height) -> Image:
    """Caching wrapper for naive downsampling."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    if target_width > 0 and target_height > 0:
        downsampled_img = img.resize((target_width, target_height), naive_method).convert("RGBA")
    else:
        downsampled_img = pix.naive_downsample(img, ds_factor, method=naive_method)
    return downsampled_img

@st.cache_data
def cached_yuv(image_bytes) -> tuple:
    """Caching wrapper for YUV algorithm."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)
    yuv_array = yuv.rgb_to_yuv(img_array)
    Y, U, V = yuv.extract_channels(yuv_array)
    Y_img = Image.fromarray(Y, mode="L")
    U_img = Image.fromarray(U, mode="L")
    V_img = Image.fromarray(V, mode="L")
    return Y_img, U_img, V_img

@st.cache_data
def cache_majority_block(image_bytes, ds_factor, target_width, target_height) -> Image:
    """Downsample using majority color block sampling directly for fair comparison."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    # Always use adaptive width/height if provided
    if target_width > 0 and target_height > 0:
        result_img = pix.majority_color_block_sampling(img, ds_factor, target_width=target_width, target_height=target_height)
        return result_img
    else:
        result_img = pix.majority_color_block_sampling(img, ds_factor)
        return result_img

@st.cache_data
def cache_refined_edge(image_bytes, ds_factor, naive_method, target_width, target_height, soft_edges=False, edge_threshold=30) -> Image:
    """Apply refined edge preserving downscale directly to the original image."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    # Always use adaptive width/height if provided
    if target_width > 0 and target_height > 0:
        result_img = pix.refined_edge_preserving_downscale(
            img, ds_factor, soft_edges, edge_threshold, naive_method,
            target_width=target_width, target_height=target_height
        )
        return result_img
    else:
        result_img = pix.refined_edge_preserving_downscale(
            img, ds_factor, soft_edges, edge_threshold, naive_method
        )
        return result_img

def get_locked_dimensions(orig_w, orig_h, width_input, height_input, ds_factor):
    """
    Returns (target_width, target_height) based on aspect ratio logic.
    If width is set, height is calculated; if height is set, width is calculated.
    If neither is set, returns (0, 0).
    """
    aspect_ratio = orig_w / orig_h
    if width_input > 0:
        target_width = width_input
        target_height = int(round(target_width / aspect_ratio))
    elif height_input > 0:
        target_height = height_input
        target_width = int(round(target_height * aspect_ratio))
    else:
        target_width = 0
        target_height = 0
    return target_width, target_height

def main():
    """Main Streamlit UI logic for Adaptive Downscale + Method Comparison App."""
    #Data Input options
    
    st.title("Adaptive Downscale + Method Comparison App")
    st.markdown(
        """
        <div style='text-align: right; font-size: 0.9em; color: #888;'>
            Developed by: 
            <a href="https://github.com/hiive" target="_blank">Hiive</a> @
            <a href="https://hiivelabs.com" target="_blank">Hiivelabs</a> &amp; 
            <a href="https://github.com/siker102" target="_blank">Siker102</a>
        </div>
        """,
        unsafe_allow_html=True
    )
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    ds_factor = st.slider("Downscale Factor (ds_factor)", min_value=2, max_value=8, value=2, step=1)
    algo = st.radio("Select algorithm:", options=["Pix", "YUV"], horizontal=True)

    # Only process if an image is uploaded
    if uploaded_image is not None:
        try:
            image_bytes = uploaded_image.getvalue()
            original_img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        except Exception as e:
            st.error("The uploaded file is not a valid image. Please upload a PNG or JPEG image.")
            return

        orig_w, orig_h = original_img.width, original_img.height
        aspect_ratio = orig_w / orig_h

        # --- Aspect-ratio-locked width/height controls ---
        st.markdown("#### Set Target Width or Height (Aspect Ratio Locked, Ignores DS Factor!)")
        col_w, col_h = st.columns(2)
        with col_w:
            width_input = st.number_input("Target Width (0 = auto)", min_value=0, value=0, step=1, key="width_input")
        with col_h:
            height_input = st.number_input("Target Height (0 = auto)", min_value=0, value=0, step=1, key="height_input")

        # Always calculate both width and height based on aspect ratio logic
        target_width, target_height = get_locked_dimensions(orig_w, orig_h, width_input, height_input, ds_factor)
        # If neither is set, fallback to ds_factor
        if target_width == 0 or target_height == 0:
            target_width = orig_w // ds_factor
            target_height = orig_h // ds_factor

        # Defensive check: Ensure minimum output size is 1x1
        if target_width < 1 or target_height < 1:
            st.error("Output size is too small. Please choose a smaller downscale factor or larger target dimensions.")
            return

        st.caption(f"Locked aspect ratio: {orig_w}:{orig_h} ({aspect_ratio:.3f})")
        st.caption(f"Output size: {target_width} x {target_height}")

        # Add naive downsampling method selection
        naive_method_label = {
            "LANCZOS": "Lanczos (high quality, default)",
            "BILINEAR": "Bilinear (smooth)",
            "NEAREST": "Nearest Neighbor (pixelated)",
            "BOX": "Box (averaging)",
            "HAMMING": "Hamming (antialiasing)",
            "BICUBIC": "Bicubic (smooth, sharp edges)"
        }
        naive_method_options = ["LANCZOS", "BILINEAR", "NEAREST", "BOX", "HAMMING", "BICUBIC"]
        naive_method = st.selectbox(
            "Naive Downsampling Method",
            options=naive_method_options,
            format_func=lambda x: naive_method_label[x]
        )

        # Map string to PIL method
        pil_methods = {
            "LANCZOS": Image.Resampling.LANCZOS,
            "BILINEAR": Image.Resampling.BILINEAR,
            "NEAREST": Image.Resampling.NEAREST,
            "BOX": Image.Resampling.BOX,
            "HAMMING": Image.Resampling.HAMMING,
            "BICUBIC": Image.Resampling.BICUBIC
        }
        selected_naive_method = pil_methods[naive_method]

        if algo == "Pix":
            operations = "1,2"  # Could be made user-configurable if desired
            st.caption(f"(operations={operations}, ds_factor={ds_factor}, target_width={target_width}, target_height={target_height})")
            with st.spinner("Running Pix algorithm..."):
                # Precompute all images, always passing target_width/target_height
                adaptive_img = cached_pix(
                    image_bytes,
                    ds_factor=ds_factor,
                    target_width=target_width,
                    target_height=target_height,
                    operations=operations,
                    naive_method=selected_naive_method
                )
                naive_img = cache_naive(image_bytes, ds_factor, selected_naive_method, target_width, target_height)
                majority_img = cache_majority_block(image_bytes, ds_factor, target_width, target_height)
                refined_img = cache_refined_edge(image_bytes, ds_factor, selected_naive_method, target_width, target_height)

                # Prepare options and mapping
                compare_options = {
                    "Original": original_img,
                    "Naive Downscaled": naive_img,
                    "Majority Downscaled": majority_img,
                    "Refined Edge Preserving": refined_img,
                    "Adaptive Downscaled": adaptive_img
                }
                compare_labels = {
                    "Original": f"Original ({original_img.width}x{original_img.height})",
                    "Naive Downscaled": f"Naive ({naive_img.width}x{naive_img.height})",
                    "Majority Downscaled": f"Majority ({majority_img.width}x{majority_img.height})",
                    "Refined Edge Preserving": f"Refined ({refined_img.width}x{refined_img.height})",
                    "Adaptive Downscaled": f"Adaptive ({adaptive_img.width}x{adaptive_img.height})"
                }

                st.subheader("Compare Any Two Methods")
                col_left, col_right = st.columns(2)
                with col_left:
                    left_choice = st.selectbox(
                        "Left Image",
                        options=list(compare_options.keys()),
                        index=0,  # Default: Original
                        key="left_image_select"
                    )
                with col_right:
                    right_choice = st.selectbox(
                        "Right Image",
                        options=list(compare_options.keys()),
                        index=4,  # Default: Adaptive Downscaled
                        key="right_image_select"
                    )

                image_comparison(
                    img1=compare_options[left_choice],
                    img2=compare_options[right_choice],
                    label1=compare_labels[left_choice],
                    label2=compare_labels[right_choice],
                    width=700,
                    show_labels=True,
                    make_responsive=True,
                    in_memory=True
                )

                # Convert images to bytes for download without overwriting variables
                result_bytes = io.BytesIO()
                adaptive_img.save(result_bytes, format="PNG")
                result_bytes.seek(0)
                
                naive_bytes = io.BytesIO()
                naive_img.save(naive_bytes, format="PNG")
                naive_bytes.seek(0)

                majority_bytes = io.BytesIO()
                majority_img.save(majority_bytes, format="PNG")
                majority_bytes.seek(0)

                refined_bytes = io.BytesIO()
                refined_img.save(refined_bytes, format="PNG")
                refined_bytes.seek(0)

                # Display download buttons
                st.markdown("---")
                st.subheader("Downloads")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.download_button(
                        label="Adaptive Downscaled",
                        data=result_bytes,
                        file_name="adaptive_downscaled.png",
                        mime="image/png",
                        key="adaptive_download",
                        type="primary"
                    )
                with col2:
                    st.download_button(
                        label="Majority Color Block",
                        data=majority_bytes,
                        file_name="majority_color_block.png",
                        mime="image/png",
                        key="majority_download"
                    )
                with col3:
                    st.download_button(
                        label="Naively Downscaled",
                        data=naive_bytes,
                        file_name="naively_downscaled.png",
                        mime="image/png",
                        key="naive_download"
                    )
                with col4:
                    st.download_button(
                        label="Refined Edge Preserving",
                        data=refined_bytes,
                        file_name="refined_edge_preserving.png",
                        mime="image/png",
                        key="refined_download"
                    )
        elif algo == "YUV":
            with st.spinner("Running YUV algorithm..."):
                Y_img, U_img, V_img = cached_yuv(image_bytes)
                st.success("YUV finished!")
                st.image(Y_img, caption="Y Channel", use_container_width=True)
                st.caption(f"Y Channel: {Y_img.width}x{Y_img.height}")
                st.image(U_img, caption="U Channel", use_container_width=True)
                st.caption(f"U Channel: {U_img.width}x{U_img.height}")
                st.image(V_img, caption="V Channel", use_container_width=True)
                st.caption(f"V Channel: {V_img.width}x{V_img.height}")

        # st.write("Selected option:", algo)  # (Optional: remove for cleaner UI)
    else:
        # Optionally, display a message or placeholder when no image is uploaded
        st.info("Please upload an image to begin.")

if __name__ == "__main__":
    main()