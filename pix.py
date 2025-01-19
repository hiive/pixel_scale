#!/usr/bin/env python3
# 3,1,2
import argparse
import sys
import io
import numpy as np
from PIL import Image, ImageFilter, ImageCms

from collections import deque
import skimage

DIAGNOSTICS = False

def ensure_srgb(pil_img: Image.Image) -> Image.Image:
    """
    Ensures the returned image is in the sRGB color space.
    1) If the image has an embedded ICC profile, convert it from that profile to sRGB.
    2) If there's no profile info, we either assume it's already sRGB or treat it as such.
    """
    # If there's an embedded ICC profile, we can extract it:
    icc_profile = pil_img.info.get("icc_profile", None)
    if icc_profile:
        # Convert from embedded profile to sRGB
        input_profile = io.BytesIO(icc_profile)  # the embedded profile
        srgb_profile = ImageCms.createProfile("sRGB")  # sRGB profile
        pil_img = ImageCms.profileToProfile(
            pil_img,
            input_profile,
            srgb_profile,
            outputMode="RGB"
        )
    else:
        # No embedded profile found, so we assume it's already sRGB or interpret as sRGB
        pil_img = pil_img.convert("RGB")

    return pil_img

def rgb_to_lab_bulk(img: Image.Image) -> np.ndarray:
    """
    Converts the RGB channels of a Pillow Image to Lab space in one bulk operation.
    Returns a NumPy array of shape (H, W, 3).
    """
    # Convert to RGB just to ensure consistency (strip alpha if present)
    img_rgb = ensure_srgb(img)
    rgb_arr = np.array(img_rgb, dtype=np.float32) / 255.0  # shape: (H, W, 3)

    # Vectorized bulk conversion to Lab
    lab_arr = skimage.color.rgb2lab(rgb_arr)  # shape: (H, W, 3)
    return lab_arr


def enforce_palette_lab_bulk(final_img: Image.Image, palette_img: Image.Image) -> Image.Image:
    """
    1) Convert 'final_img' to Lab in a single bulk operation (ignoring alpha for the moment).
    2) Convert each unique palette color to Lab as well (also in bulk).
    3) Find the palette entry with minimal Delta E for each non-transparent pixel.
    4) Return a new RGBA image with enforced palette, preserving any alpha=0 as is.
    """
    # Convert both to RGBA for consistent channel usage
    final_img = final_img.convert("RGBA")
    palette_img = palette_img.convert("RGBA")

    # Extract the alpha channel from final_img so we can restore it at the end
    alpha_channel = final_img.getchannel("A")

    # 1) Bulk-convert final_img's RGB to Lab (ignore alpha in color space)
    final_lab = rgb_to_lab_bulk(final_img)  # shape: (H, W, 3)

    # 2) Extract unique colors from the palette (in RGB)
    pal_rgb = np.array(palette_img)[:, :, :3]
    unique_colors = np.unique(pal_rgb.reshape(-1, 3), axis=0).astype(np.float32) / 255.0

    # 3) Convert palette colors to Lab in one shot
    unique_colors_lab = skimage.color.rgb2lab(unique_colors.reshape(-1, 1, 3))
    # shape: (N, 1, 3)
    unique_colors_lab = unique_colors_lab[:, 0, :]  # shape: (N, 3)

    # 4) For each pixel in final_img, find the nearest palette color in Lab
    #    We'll do this one pixel at a time or with a partial vector approach.
    #    A direct fully vectorized approach for large N can be more involved, but here's
    #    a middle ground: each pixel does a quick search among unique_colors.

    final_arr = np.array(final_img, dtype=np.uint8)  # shape: (H, W, 4)
    height, width, _ = final_arr.shape

    for y in range(height):
        for x in range(width):
            a = final_arr[y, x, 3]
            if a != 0:
                # Non-transparent => enforce palette
                lab_pixel = final_lab[y, x]  # shape: (3,)

                # Compare this pixel's Lab to each palette color in Lab
                # deltaE ~ Euclidean in Lab for simplicity, or more sophisticated formula
                diffs = unique_colors_lab - lab_pixel  # shape: (N, 3)
                dist_sq = np.sum(diffs * diffs, axis=1)
                idx = np.argmin(dist_sq)

                # Replace RGB with the nearest palette color (convert back to 0..255)
                chosen_color = (unique_colors[idx] * 255).astype(np.uint8)
                final_arr[y, x, :3] = chosen_color

    # 5) Construct final RGBA image, reusing the original alpha channel
    result = Image.fromarray(final_arr, mode="RGBA")
    # Replace alpha with the original
    result.putalpha(alpha_channel)
    return result

def floodfill_mask(img: Image.Image, tolerance=0) -> Image.Image:
    """
    1) Flood-fills an image from 'start' (default: (0,0)) to find all connected pixels
       that match the start color (within 'tolerance').
    2) Returns a binary mask (L-mode) where flood-filled pixels => 0 (OFF),
       and non-flood-filled pixels => 255 (ON).

    Args:
        img: Pillow Image, converted to RGB or RGBA as needed.
        start: (x, y) coordinate for the flood-fill start. Default=(0,0).
        tolerance: Allowed difference in color channels. Default=0 => exact match.
    """
    # Convert to RGBA to handle alpha if present (and consistent channel access)
    img = img.convert("RGBA")
    width, height = img.size

    img_mask = img.copy()
    img_mask = img_mask.filter(ImageFilter.BoxBlur(0.25))
    # ImageDraw.floodfill(img_mask, xy=(sx, sy), value=(255,255,255,255), thresh=tolerance)

    # return img_mask

    px = img_mask.load()  # Direct pixel access
    # Get the color at the start
    corners = [
        ((0, 0), px[0, 0]),
        ((img.width - 1, 0), px[img.width - 1, 0]),
        ((0, img.height - 1), px[0, img.height - 1]),
        ((img.width - 1, img.height - 1), px[img.width - 1, img.height - 1])
    ]

    # Find the corner with the lowest alpha among the four
    lowest_alpha_corner = min(corners, key=lambda item: item[1][3])

    # This returns a tuple ((x_coord, y_coord), (R, G, B, A))
    (sx, sy), start_color = lowest_alpha_corner

    # Visited array to mark flood-filled pixels
    visited = np.zeros((height, width), dtype=np.bool_)

    def color_matches(c1, c2, tol):
        # Check each channel difference <= tol
        return all(abs(c1[i] - c2[i]) <= tol for i in range(4))  # RGBA channels

    # BFS or DFS queue
    queue = deque()
    queue.append((sx, sy))
    visited[sy, sx] = True

    while queue:
        x, y = queue.popleft()
        current_color = px[x, y]

        # Check neighbors (4-directional)
        for nx, ny in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
            if 0 <= nx < width and 0 <= ny < height:
                if not visited[ny, nx]:
                    neighbor_color = px[nx, ny]
                    if color_matches(neighbor_color, start_color, tolerance):
                        visited[ny, nx] = True
                        queue.append((nx, ny))

    # Now build the mask:
    # if visited => OFF => 0
    # else => ON => 255
    mask_data = np.where(visited, 0, 255).astype(np.uint8)

    # Convert mask_data to a Pillow image in 'L' mode (grayscale)
    mask_img = Image.fromarray(mask_data, mode="L")

    return mask_img


def alpha_threshold_mask(img: Image.Image, threshold=0) -> Image.Image:
    """
    Returns an L-mode mask image where:
      - ON (255) if alpha > threshold
      - OFF (0) otherwise

    Example usage:
        mask_img = alpha_threshold_mask(original_img, threshold=128)
    """
    # Ensure RGBA
    img_rgba = img.convert("RGBA")

    # Extract the alpha channel as a NumPy array
    arr = np.array(img_rgba)  # shape: (height, width, 4)
    alpha_channel = arr[:, :, 3]

    # Build mask: 255 for alpha > threshold, else 0
    mask_data = np.where(alpha_channel > threshold, 255, 0).astype(np.uint8)

    # Convert to an L-mode Pillow Image
    mask_img = Image.fromarray(mask_data, mode="L")

    return mask_img


def conditional_replace(
    downsampled_original_img: Image.Image,
    second_img: Image.Image,
    mask: Image.Image,
    alpha_min: int,
) -> Image.Image:
    """
    for each pixel in 'second_img' specified by the mask:
      - If second_img's alpha < 255 AND the downsampled pixel's alpha is > 0,
        replace that pixel in second_img with the downsampled pixel.
      - Otherwise, leave second_img's pixel as-is.
      - UNLESS the pixel to place has an alpha less than alpha_min,
      - in which case, set it to fully transparent.


    Returns a new RGBA Image.
    """

    w = downsampled_original_img.width
    h = downsampled_original_img.height

    # 2) Convert second_img to RGBA for consistent pixel access
    second_img = second_img.convert("RGBA")

    # 3) Ensure second_img is the SAME SIZE as the downsampled image
    if second_img.size != (w, h):
        raise ValueError(
            "second_img must match the downsampled_original_img dimensions: "
            f"{(w,h)} but got {second_img.size}"
        )

    # 4) Convert both images to NumPy arrays
    down_arr = np.array(downsampled_original_img)  # shape: (h, w, 4)
    second_arr = np.array(second_img) # shape: (h, w, 4)
    mask_arr = np.array(mask)
    # 5) Create a final array starting as a copy of the second image
    final_arr = second_arr.copy()

    # 6) For each pixel, conditionally replace
    for y in range(h):
        for x in range(w):
            m = mask_arr[y, x]
            if m == 0:
                final_arr[y, x] = [0, 0, 0, 0]
                continue
            # If alpha < 255 in second_img and alpha in downsampled is > 0
            alpha_2 = second_arr[y, x, 3]
            alpha_d = down_arr[y, x, 3]
            if alpha_2 < 255 and alpha_d > 0:
                    if max(alpha_d, alpha_2) > alpha_min:
                        # Replace the pixel
                        final_arr[y, x, :] = down_arr[y, x, :]
                    else:
                        # set it empty
                        final_arr[y, x] = [0, 0, 0, 0]

    # 7) Convert the array back to an Image and return
    return Image.fromarray(final_arr, mode="RGBA")

def majority_color_block_sampling(img, scale_factor):
    """
    Downscale using majority-color block sampling (RGBA-aware).
    - 'scale_factor' indicates how many original pixels
      combine into one new pixel (e.g., 2 => 2x2 -> 1x1).
    - If the image is RGBA and alpha is either 0 or 255 (fully transparent or fully opaque),
      the function preserves that transparency.
    """
    # Ensure we don't lose alpha info
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    width, height = img.size
    new_width, new_height = width // scale_factor, height // scale_factor

    pixels = np.array(img)  # shape: (height, width, 4) for RGBA

    # Create an output array for RGBA
    output = np.zeros((new_height, new_width, 4), dtype=pixels.dtype)

    for y in range(new_height):
        for x in range(new_width):
            # Identify the block in the original image
            block = pixels[
                y * scale_factor : (y + 1) * scale_factor,
                x * scale_factor : (x + 1) * scale_factor
            ]
            # Flatten and handle each pixel’s RGBA
            block_2d = block.reshape(-1, 4)

            # Count transparency vs. opaque
            alpha_values = block_2d[:, 3]
            num_transparent = np.sum(alpha_values == 0)
            num_opaque = len(alpha_values) - num_transparent

            # If mostly transparent in this block => fully transparent pixel
            if num_transparent > num_opaque:
                output[y, x] = [0, 0, 0, 0]
            else:
                # Among the opaque pixels, find the most frequent RGBA
                opaque_pixels = block_2d[block_2d[:, 3] != 0]
                # If no opaque pixels exist, force transparent
                if len(opaque_pixels) == 0:
                    output[y, x] = [0, 0, 0, 0]
                else:
                    color_counts = {}
                    for row in opaque_pixels:
                        color_tuple = tuple(row)
                        color_counts[color_tuple] = color_counts.get(color_tuple, 0) + 1

                        # simple majority
                        majority_color = max(color_counts, key=color_counts.get)
                        output[y, x] = majority_color

    return Image.fromarray(output, mode="RGBA")



def refined_edge_preserving_downscale(img, scale_factor, soft_edges, edge_threshold=30):
    # Ensure RGBA so we handle transparency properly
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # 1. Detect edges in grayscale
    #    We'll do this on a copy converted to RGB just for the FIND_EDGES
    edges = img.convert("RGB").filter(ImageFilter.FIND_EDGES).convert("L")
    edges_data = np.array(edges)
    edges_data = np.where(edges_data > edge_threshold, 255, 0).astype(np.uint8)
    strong_edges = Image.fromarray(edges_data, mode="L")

    # 2. Downscale while preserving color
    downsampled = majority_color_block_sampling(img, scale_factor)  # from your code

    strong_edges = strong_edges.resize(downsampled.size, Image.Resampling.BILINEAR)

    # 3. Use a mask-based approach to set only the edge areas

    # Convert the downsampled to RGBA
    color_data = downsampled.convert("RGBA")

    # Step 3a: Add partial opacity to the mask so it doesn't produce pure black
    # We'll do this by converting edges to RGBA and adjusting alpha.
    mask_rgba = Image.new("RGBA", color_data.size)
    if soft_edges:
        mask_rgba.putdata([(0, 0, 0, v) for v in strong_edges.getdata()])
    else:
        mask_rgba.putdata([(0, 0, 0, 255 if v > 0 else 0) for v in strong_edges.getdata()])
    #
    # Now each pixel has alpha = 0 or 255, matching edge map.

    # Split out the RGBA channels
    r_c, g_c, b_c, a_c = color_data.split()  # color_data is RGBA
    r_m, g_m, b_m, a_m = mask_rgba.split()  # mask_rgba is RGBA

    # Merge them so the final image has R/G/B from color_data and A from mask_rgba
    final = Image.merge("RGBA", (r_c, g_c, b_c, a_m))

    return final


def blend_edges_second_pass(img, edge_threshold=30, alpha=0.5):
    """
    Blends edges in a separate pass after other operations.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # Convert to RGB for edge detection
    edges = img.convert("RGB").filter(ImageFilter.FIND_EDGES).convert("L")
    edges_data = np.array(edges)
    edges_data = np.where(edges_data > edge_threshold, 255, 0).astype(np.uint8)
    strong_edges = Image.fromarray(edges_data, mode="L")

    # Resize edges
    edges_resized = strong_edges.resize(img.size, Image.Resampling.BILINEAR)

    # Blend
    img_gray = img.convert("L")
    combined = Image.blend(img_gray, edges_resized, alpha)

    # Merge back, preserving original alpha
    original_alpha = img.split()[-1]
    return Image.merge("RGBA", (combined, combined, combined, original_alpha))


def main():
    parser = argparse.ArgumentParser(description="Pixel Art Downscaling with RGBA & Small-Sprite Optimizations")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("output", help="Path to output image")
    parser.add_argument("-s", "--scale_factor", type=int, default=2,
                        help="Reduction ratio (default=2 => half width & height).")
    parser.add_argument("-o", "--operations", type=str, default="1,2,3",
                        help="Comma-separated list of operations in order:"
                             " 1=Majority-Color Downscale,"
                             " 2=Refined Edge-Preserving")
    parser.add_argument("--edge_threshold", type=int, default=30,
                        help="Threshold for major edges (default=30).")
    # parser.add_argument("--blend_alpha", type=float, default=1.0,
    #                     help="Blending alpha for second-pass edges (default=0.0).")
    parser.add_argument("--palette", type=str, default=None, help="Path to PNG to extract final palette from.")
    parser.add_argument("--soft_edges", action="store_true",
                        help="Use soft edges for Refined Edge-Preserving.")
    parser.add_argument("--alpha_min", type=int, default=72,
                        help="Alpha Threshold for inclusion in output (default=72).")
    args = parser.parse_args()

    # Load original image in RGBA
    original_img = Image.open(args.input).convert("RGBA")

    # Parse operations
    ops = [op.strip() for op in args.operations.split(",")]

    downsampled_img = (original_img.resize(
        (original_img.width//args.scale_factor, original_img.height//args.scale_factor),
        Image.Resampling.LANCZOS)
                       .convert("RGBA"))

    if DIAGNOSTICS:
        downsampled_img.save('downsampled_' + args.output)

    # Intermediate result
    result_img = original_img
    for op in ops:
        if op == "1":
            result_img = majority_color_block_sampling(
                img=result_img,
                scale_factor=args.scale_factor
            )
            if DIAGNOSTICS:
                result_img.save('majority_color_block_sampling_' + args.output)
            args.scale_factor = 1
        elif op == "2":
            result_img = refined_edge_preserving_downscale(
                img=result_img,
                scale_factor=args.scale_factor,
                soft_edges=args.soft_edges,
                edge_threshold=args.edge_threshold
            )
            if DIAGNOSTICS:
                result_img.save('refined_edge_preserving_downscale_' + args.output)
            args.scale_factor = 1
        else:
            print(f"Unknown operation: {op}")
            sys.exit(1)

    mask = floodfill_mask(downsampled_img)

    if DIAGNOSTICS:
        mask.save('mask_' + args.output)
        result_img.save('pre_conditional_replace_' + args.output)

    result_img = conditional_replace(downsampled_img, result_img, mask, args.alpha_min)

    # match to palette
    if args.palette:
        palette_img = Image.open(args.palette)
        # result_img.save('pre-palette' + args.output)
        result_img = enforce_palette_lab_bulk(result_img, palette_img)
    # Save result
    result_img.save(args.output)


if __name__ == "__main__":
    main()

# for character artwork:
    # python pix.py <input>.png <output>.png -o 1 -s 2

# for tile or large artwork
    # python pix.py <input>.png <output>.png -o 1,2 -s 2