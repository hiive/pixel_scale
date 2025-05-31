import streamlit as st
import io
from streamlit_image_comparison import image_comparison
import pix
import yuv
import numpy as np
from PIL import Image

@st.cache_data
def cached_pix(image_bytes, ds_factor, target_width, target_height, operations) -> Image:
    # Caching wrapper for Pix algorithm using pix.py main instead of naive_downsample
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    # Use target_width/target_height if provided (non-zero), otherwise scale by ds_factor
    result_img = pix.main(
        img,
        ds_factor=ds_factor,
        target_width=target_width if target_width > 0 else None,
        target_height=target_height if target_height > 0 else None,
        operations=operations
    )
    return result_img

@st.cache_data
def cache_naive(image_bytes, ds_factor) -> Image:
    # Caching wrapper for naive downsampling
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    downsampled_img = pix.naive_downsample(img, ds_factor)
    return downsampled_img

@st.cache_data
def cached_yuv(image_bytes) -> tuple:
    # Caching wrapper for YUV algorithm
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)
    yuv_array = yuv.rgb_to_yuv(img_array)
    Y, U, V = yuv.extract_channels(yuv_array)
    Y_img = Image.fromarray(Y, mode="L")
    U_img = Image.fromarray(U, mode="L")
    V_img = Image.fromarray(V, mode="L")
    return Y_img, U_img, V_img

@st.cache_data
def cache_majority_block(image_bytes, ds_factor) -> Image:
    # Caching wrapper for majority color block sampling
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    result_img = pix.majority_color_block_sampling(img, ds_factor)
    return result_img

def main():
    #Data Input options
    st.title("Downscale + Method Comparison App")
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    ds_factor = st.radio("Downscale Factor (ds_factor)", options=[2, 4, 6, 8], index=0, horizontal=True)  # Fixed spacing
    algo = st.radio("Select algorithm:", options=["Pix", "YUV"], horizontal=True)
    
    target_width = 0
    target_height = 0

    if uploaded_image:
        try:
            image_bytes = uploaded_image.getvalue()
            original_img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        except Exception as e:
            st.error("The uploaded file is not a valid image. Please upload a PNG or JPEG image.")
            return
        if algo == "Pix":
            operations = "1,2"  # Ensure operations is always defined here
            st.caption(f"(operations=1,2, ds_factor={ds_factor})")
            # Show a progress spinner instead of st.status to avoid layout issues
            with st.spinner("Running Pix algorithm..."):
                result = cached_pix(image_bytes, ds_factor, target_width, target_height, operations)
                st.success("Pix finished!")
                st.subheader("Compare Original vs. Adaptive Downscaled")
                image_comparison(
                    img1=original_img,
                    img2=result,
                    label1="Original",
                    label2=" Adaptive Downscaled"
                )
                col1, col_spacer, col2 = st.columns([1, 3, 1])
                with col1:
                    st.caption(f"Original: {original_img.width}x{original_img.height}")
                with col2:
                    st.caption(f"Adaptive: {result.width}x{result.height}")

                st.subheader("Compare Naively Downscale vs. Adaptive Downscaled")
                naive_img = cache_naive(image_bytes, ds_factor)
                image_comparison(
                    img1=naive_img,
                    img2=result,
                    label1="Naively Downscaled",
                    label2=" Adaptive Downscaled"
                )
                col1, col_spacer, col2 = st.columns([1, 3, 1])
                with col1:
                    st.caption(f"Naive: {naive_img.width}x{naive_img.height}")
                with col2:
                    st.caption(f"Adaptive: {result.width}x{result.height}")

                st.subheader("Compare Majority Color Block vs. Adaptive Downscaled")
                majority_img = cache_majority_block(image_bytes, ds_factor)
                image_comparison(
                    img1=majority_img,
                    img2=result,
                    label1="Majority Color Block Sampled",
                    label2=" Adaptive Downscaled"
                )
                col1, col_spacer, col2 = st.columns([1, 3, 1])
                with col1:
                    st.caption(f"Majority: {majority_img.width}x{majority_img.height}")
                with col2:
                    st.caption(f"Adaptive: {result.width}x{result.height}")
                # Convert images to bytes for download without overwriting variables
                result_bytes = io.BytesIO()
                result.save(result_bytes, format="PNG")
                result_bytes.seek(0)
                
                naive_bytes = io.BytesIO()
                naive_img.save(naive_bytes, format="PNG")
                naive_bytes.seek(0)

                majority_bytes = io.BytesIO()
                majority_img.save(majority_bytes, format="PNG")
                majority_bytes.seek(0)

                # Display download buttons
                st.markdown("---")
                st.subheader("Downloads")
                col1, col2, col3 = st.columns(3)
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
    
    st.write("Selected option:", algo)

if __name__ == "__main__":
    main()