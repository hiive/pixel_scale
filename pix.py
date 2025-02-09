#!/usr/bin/env python3
# 3,1,2
import argparse
import sys
import io

import numpy as np
from PIL import Image, ImageFilter, ImageCms, ImageOps

from collections import deque, Counter
import skimage
from scipy.ndimage import binary_fill_holes

DIAGNOSTICS = False

## mask stuff
def outline_mask_by_dominant_color(
    img: Image.Image,
    tolerance: float = 10.0,
    include_internal_outline: bool = False,
) -> (Image.Image, Image.Image):
    """
    1. Opens an RGBA image from 'input_path'.
    2. Scans for candidate outline pixels:
       - Pixel alpha != 0
       - At least one 8-way neighbor with alpha=0 or out of bounds (assume out of bounds=transparent)
    3. Groups candidate pixels into color clusters (within 'tolerance' in RGB space).
    4. Identifies the cluster with the highest count => 'dominant outline color'.
    5. Builds a mask image:
       - A pixel is ON (255) if it meets the candidate condition
         and is within tolerance of the dominant color.
       - Otherwise OFF (0).
    6. Saves the resulting mask to 'output_path' as a PNG.
    """
    # Ensure we don't lose alpha info
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # --- Helper functions ---
    def color_distance_sq(c1, c2):
        # Euclidean distance squared in RGB (ignore alpha)
        return ((int(c1[0]) - int(c2[0])) ** 2 +
                (int(c1[1]) - int(c2[1])) ** 2 +
                (int(c1[2]) - int(c2[2])) ** 2)

    def find_cluster_index(clusters, color, max_dist_sq):
        """
        Looks through 'clusters' for a representative color
        whose distance to 'color' is <= max_dist_sq.
        Returns index if found, else -1.
        """
        for i, (rep_color, count) in enumerate(clusters):
            if color_distance_sq(rep_color, color) <= max_dist_sq:
                return i
        return -1

    # --- 1) Open image in RGBA and build a NumPy array ---
    arr = np.array(img)
    h, w, _ = arr.shape

    # Directions for an 8-way neighborhood
    neighbors_8 = [(-1, -1), (-1, 0), (-1, 1),
                   ( 0, -1),           ( 0, 1),
                   ( 1, -1), ( 1, 0), ( 1, 1)]

    # We consider a pixel "transparent" if alpha==0, or if it's out of bounds.

    # --- 2) Identify candidate pixels and record their colors ---
    clusters = []  # Will hold (representative_rgb, count)
    max_dist_sq = tolerance * tolerance

    for y in range(h):
        for x in range(w):
            a = arr[y, x, 3]
            if a != 0:
                # Check 8 neighbors => if any is out-of-bounds or alpha=0 => "transparent neighbor"
                transparent_neighbor = False
                for dy, dx in neighbors_8:
                    ny = y + dy
                    nx = x + dx
                    # If it's out of bounds, treat as alpha=0 => transparent
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        transparent_neighbor = True
                        break
                    else:
                        if arr[ny, nx, 3] == 0:
                            transparent_neighbor = True
                            break

                if transparent_neighbor:
                    # This pixel is a candidate outline pixel
                    c = (arr[y, x, 0], arr[y, x, 1], arr[y, x, 2])  # (R, G, B)
                    idx = find_cluster_index(clusters, c, max_dist_sq)
                    if idx >= 0:
                        rep_color, count = clusters[idx]
                        clusters[idx] = (rep_color, count + 1)
                    else:
                        clusters.append((c, 1))

    if not clusters:
        # No outline pixels found; make a blank mask
        mask_arr = np.zeros((h, w, 4), dtype=np.uint8)
        return Image.fromarray(mask_arr, mode="RGBA"), img.copy()

    # --- 3) Determine the dominant color cluster ---
    # The cluster with the largest 'count'
    dominant_cluster = max(clusters, key=lambda c: c[1])
    dominant_color = dominant_cluster[0]  # (R, G, B)

    # --- 4) Build the mask ---
    mask_arr = np.zeros((h, w, 4), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            a = arr[y, x, 3]
            if a != 0:
                # Check neighbors => if any is out-of-bounds or alpha=0
                has_transparent_neighbor = False
                for dy, dx in neighbors_8:
                    ny = y + dy
                    nx = x + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        has_transparent_neighbor = True
                        break
                    else:
                        if arr[ny, nx, 3] == 0:
                            has_transparent_neighbor = True
                            break

                if has_transparent_neighbor:
                    # See if pixel color is within tolerance of dominant_color
                    this_rgb = arr[y, x, :3]
                    if color_distance_sq(this_rgb, dominant_color) <= max_dist_sq:
                        mask_arr[y, x] = arr[y, x]  # Outline pixel

    external_mask_arr = mask_arr.copy()

    # include internal outline if necessary.
    if include_internal_outline:
        # We'll do a BFS or DFS over all outline pixels in mask_arr
        from collections import deque
        queue = deque()

        # Enqueue all existing outline pixels initially
        for y in range(h):
            for x in range(w):
                if mask_arr[y, x, 3] != 0:
                    queue.append((x, y))

        # BFS
        while queue:
            x, y = queue.popleft()
            # Explore neighbors
            for dx, dy in neighbors_8:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    # Check if not already in outline mask
                    if mask_arr[ny, nx, 3] == 0:
                        alpha = arr[ny, nx, 3]
                        if alpha != 0:
                            # Within the same 'dominant' color tolerance?
                            rgb = arr[ny, nx, :3]
                            if color_distance_sq(rgb, dominant_color) <= max_dist_sq:
                                # Mark as outline and enqueue
                                mask_arr[ny, nx] = arr[ny, nx]
                                queue.append((nx, ny))

    # --- 5) Save the resulting mask ---
    mask_img = Image.fromarray(mask_arr, mode="RGBA")
    interior_img = fill_outline_with_interior(mask_arr, img, action='fill')
    interior_img = fill_outline_with_interior(external_mask_arr, interior_img, action='blank')
    return mask_img, interior_img

def fill_outline_with_interior(
    mask_arr: np.ndarray,
    original_img: Image.Image,
    max_search_radius: int = 3,
    action: str = "fill",
) -> Image.Image:
    """
    1) Converts 'original_img' to a NumPy RGBA array.
    2) For every pixel where 'mask_arr[y, x, 3] != 0' (considered outline),
       searches an expanding neighborhood (up to 'max_search_radius')
       for interior pixels (mask_arr[y, x, 3] == 0).
         - If action='fill':
             * Gathers interior pixel colors and picks the majority color.
             * If no interior color is found, leaves it as-is (or optionally transparent).
           If action='blank':
             * Sets that pixel to fully transparent immediately (0,0,0,0).
    3) Returns a new RGBA Pillow Image with the updated pixels.
    """

    # Convert the original to RGBA and NumPy
    orig_arr = np.array(original_img.convert("RGBA"))
    h, w, _ = orig_arr.shape

    # This will hold our final result
    new_arr = orig_arr.copy()

    # 8-way directions for local neighborhood checks
    neighbors_8 = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)
    ]

    def gather_interior_colors(cx, cy, radius):
        """
        Returns a list of RGBA colors from 'orig_arr' for interior pixels within 'radius'
        of (cx, cy). 'mask_arr' alpha=0 => interior pixel, alpha!=0 => outline pixel.
        """
        gathered = []
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h:
                    # Check if neighbor is 'interior'
                    if mask_arr[ny, nx, 3] == 0 and orig_arr[ny, nx, 3] != 0:
                        gathered.append(tuple(orig_arr[ny, nx]))  # (R,G,B,A)
        return gathered

    for y in range(h):
        for x in range(w):
            # Check if this is an outline pixel
            if mask_arr[y, x, 3] != 0:
                if action == "blank":
                    # Immediately set this outline pixel to transparent
                    new_arr[y, x] = [0, 0, 0, 0]
                else:
                    # action=='fill' (default behavior)
                    found_colors = []
                    for radius in range(1, max_search_radius + 1):
                        found_colors = gather_interior_colors(x, y, radius)
                        if found_colors:
                            break

                    if found_colors:
                        # Pick the majority color among found_colors
                        color_counts = Counter(found_colors)
                        likely_color, _ = color_counts.most_common(1)[0]
                        new_arr[y, x] = likely_color
                    else:
                        # No interior neighbor found. Optionally make transparent:
                        # new_arr[y, x] = [0, 0, 0, 0]
                        pass

    return Image.fromarray(new_arr, mode="RGBA")


def downscale_outline_components(
        outline_mask: Image.Image,
        scale: int,
        fill_threshold: float = 0.5
) -> Image.Image:
    """
    Approach A: Connected-component-aware downscale of an outline mask.

    1) Convert the input outline mask to a binary array (1=on/outline, 0=off).
    2) Label connected components (8-way).
    3) For each component, extract its bounding box.
    4) Downscale that bounding box from old_size to new_size by block-sampling:
       If fraction of original 'on' pixels mapping to a new pixel >= fill_threshold,
       mark the new pixel as 'on'.
    5) Place each downscaled bounding box into a final (new_width x new_height) mask.
    6) Return an RGBA image where alpha=255 => outline, alpha=0 => background.

    Args:
        outline_mask:  Pillow Image in RGBA or L mode. Non-zero => on/outline, zero => off.
        scale: Scale factor.
        fill_threshold:
            A float in [0..1]. The fraction of "on" pixels needed in a block
            to call the downscaled pixel 'on'.
            - 0.5 => if at least half are on, new pixel becomes on.

    Returns:
        An RGBA Pillow Image of size (new_width, new_height).
        Pixel alpha=255 => outline, alpha=0 => background.
    """

    # --- 1) Convert to binary array ---
    arr = np.array(outline_mask.convert("RGBA"))
    # We'll treat alpha>0 as 'on'. If you prefer an L-mode mask, adjust accordingly.
    bin_arr = (arr[:, :, 3] > 0).astype(np.uint8)
    orig_h, orig_w = bin_arr.shape
    new_width = orig_w // scale
    new_height = orig_h // scale
    # The output array in binary form (1=on, 0=off). We'll fill it component by component.
    final_bin = np.zeros((new_height, new_width), dtype=np.uint8)

    # 8-way connectivity for BFS/DFS
    neighbors_8 = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1), (0, 1),
                   (1, -1), (1, 0), (1, 1)]

    # --- 2) Label connected components (8-way) ---
    # We'll do a manual BFS labeling approach (no external libraries).
    label_map = np.zeros_like(bin_arr, dtype=np.int32)  # 0 => unlabeled
    label_counter = 0

    for y in range(orig_h):
        for x in range(orig_w):
            if bin_arr[y, x] == 1 and label_map[y, x] == 0:
                # Found a new connected component
                label_counter += 1
                # BFS from (x, y)
                queue = deque()
                queue.append((x, y))
                label_map[y, x] = label_counter

                while queue:
                    cx, cy = queue.popleft()
                    for dx, dy in neighbors_8:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < orig_w and 0 <= ny < orig_h:
                            if bin_arr[ny, nx] == 1 and label_map[ny, nx] == 0:
                                label_map[ny, nx] = label_counter
                                queue.append((nx, ny))


    if label_counter == 0:
        # No outline pixels => return blank mask
        return Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))

    # --- 3) For each component, find bounding box ---
    # We'll store (xmin, ymin, xmax, ymax) for each label
    boxes = [(orig_w, orig_h, -1, -1)] * (label_counter + 1)  # index 0 is unused
    # Initialize: (xmin, ymin, xmax, ymax) => we'll track min max

    # Overwrite with valid min/max
    boxes = []
    for _ in range(label_counter + 1):
        boxes.append([orig_w, orig_h, -1, -1])  # [xmin, ymin, xmax, ymax]

    for y in range(orig_h):
        for x in range(orig_w):
            lab = label_map[y, x]
            if lab != 0:
                if x < boxes[lab][0]:
                    boxes[lab][0] = x
                if y < boxes[lab][1]:
                    boxes[lab][1] = y
                if x > boxes[lab][2]:
                    boxes[lab][2] = x
                if y > boxes[lab][3]:
                    boxes[lab][3] = y

    # --- 4) Downscale each bounding box and composite into final_bin ---
    # We'll define a helper function to handle block sampling for a single bounding box.

    def downscale_component_block(lab_id, xmin, ymin, xmax, ymax):
        """
        Extracts the component's bounding box from label_map==lab_id,
        downscales it to the fraction that fits new_width/new_height
        according to a global scale, or a separate bounding-box-based scale approach.

        We'll do a global scale based on:
          scale_x = new_width / orig_w
          scale_y = new_height / orig_h
        Then place it in final_bin at the appropriate location.
        Alternatively, you can choose bounding-box-based scaling separately,
        but that can complicate alignment with the interior.
        """
        # bounding box size
        w_box = (xmax - xmin + 1)
        h_box = (ymax - ymin + 1)

        # Determine global scale factors (like if you're consistently scaling entire sprite)
        scale_x = new_width / orig_w
        scale_y = new_height / orig_h

        # new bounding box sizes (round or floor)
        new_w_box = max(1, int(round(w_box * scale_x)))
        new_h_box = max(1, int(round(h_box * scale_y)))

        # We'll create a small 2D array new_box_bin to store the downscaled bounding box
        new_box_bin = np.zeros((new_h_box, new_w_box), dtype=np.uint8)

        # Each new pixel corresponds to a block of size (scale_factor_x * scale_factor_y)
        # Let's invert that logic:
        # For each new pixel (nx, ny), find which block in the original bounding box it maps to.
        # Then count how many are on. If fraction >= fill_threshold => on.

        for ny in range(new_h_box):
            for nx in range(new_w_box):
                # original block in [xmin, xmax], [ymin, ymax]
                # We'll do a float-based approach to compute the region in the original bounding box
                # that maps to (nx, ny) in new_box_bin.
                # For example: x0 = (nx / new_w_box) * w_box, x1 = ((nx+1) / new_w_box) * w_box
                # Then we offset by xmin in the global array.

                x0 = int(xmin + (nx / new_w_box) * w_box)
                x1 = int(xmin + ((nx + 1) / new_w_box) * w_box)  # not inclusive
                y0 = int(ymin + (ny / new_h_box) * h_box)
                y1 = int(ymin + ((ny + 1) / new_h_box) * h_box)

                if x1 > x0 and y1 > y0:
                    block = (label_map[y0:y1, x0:x1] == lab_id)
                    # fraction of on-pixels
                    fraction_on = block.sum() / block.size
                    if fraction_on >= fill_threshold:
                        new_box_bin[ny, nx] = 1

        # Now we place new_box_bin in the final_bin with the correct offset
        # We'll compute the scaled bounding box coords in final_bin
        scaled_xmin = int(round(xmin * scale_x))
        scaled_ymin = int(round(ymin * scale_y))

        # Then overlay new_box_bin
        for ny in range(new_h_box):
            for nx in range(new_w_box):
                if new_box_bin[ny, nx] == 1:
                    fy = scaled_ymin + ny
                    fx = scaled_xmin + nx
                    if 0 <= fx < new_width and 0 <= fy < new_height:
                        final_bin[fy, fx] = 1

    # Loop over each component
    for lab_id in range(1, label_counter + 1):
        xmin, ymin, xmax, ymax = boxes[lab_id]
        if xmin <= xmax and ymin <= ymax:
            downscale_component_block(lab_id, xmin, ymin, xmax, ymax)

    # Build RGBA from final_bin
    # alpha=255 => outline, 0 => background
    final_rgba = np.zeros((new_height, new_width, 4), dtype=np.uint8)
    final_rgba[final_bin == 1, 3] = 255

    return Image.fromarray(final_rgba, mode="RGBA")


def xor_alpha_masks(mask1: Image.Image, mask2: Image.Image) -> Image.Image:
    """
    Takes two RGBA images 'mask1' and 'mask2', each serving as a mask:
      - A pixel is considered "set" if its alpha channel > 0.
    Returns a new RGBA image 'xor_mask', where:
      - xor_mask alpha=255 if exactly one of (mask1, mask2) has alpha>0 at that pixel.
      - Otherwise alpha=0.
    All RGB channels are set to 0 in the resulting mask (only alpha is used).
    """

    # Convert both masks to RGBA arrays
    arr1 = np.array(mask1.convert("RGBA"))
    arr2 = np.array(mask2.convert("RGBA"))

    # Ensure both have the same dimensions
    if arr1.shape != arr2.shape:
        raise ValueError(f"Mask dimensions differ: {arr1.shape} vs {arr2.shape}")

    h, w, _ = arr1.shape

    # Determine which pixels are set (alpha>0)
    set1 = (arr1[..., 3] > 0)
    set2 = (arr2[..., 3] > 0)

    # Compute XOR: True if exactly one mask is set, False otherwise
    out_set = np.logical_xor(set1, set2)

    # Build an output RGBA array: everything black, alpha=255 where out_set is True
    out_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    out_rgba[out_set, 3] = 255

    # Convert back to a Pillow Image
    return Image.fromarray(out_rgba, mode="RGBA")

def postprocess_thick_outline(
        downscaled_img: Image.Image,
        internal_colors: Image.Image,
        max_search_radius: int = 3
) -> Image.Image:
    """
    Post-process a downscaled image to thin out any overly thick outlines.

    Steps:
      1) Convert 'downscaled_img' (the final, smaller sprite) to an RGBA array.
      2) For each pixel where alpha>0, check if all 8-way neighbors also have alpha>0
         => It's an "internal" outline pixel (thicker than 1px).
      3) Replace that pixel with a guessed interior color from 'original_upscaled' by
         sampling around (x*scale_factor, y*scale_factor) within max_search_radius.
         - If no interior color found, optionally leave it or make it transparent.
      4) Return a new RGBA Pillow Image.

    Args:
        downscaled_img:       The final, downscaled RGBA sprite.
        internal_colors:      The 'de-outlined' interior color image.
        max_search_radius:    Neighborhood in 'internal_colors' to look for interior colors.

    Returns:
        A Pillow Image in RGBA mode, with thick, interior outline pixels replaced by interior colors.
    """

    # Convert the final downscaled image to RGBA array
    final_arr = np.array(downscaled_img.convert("RGBA"))
    h, w, _ = final_arr.shape

    # Convert the internal color img for color referencing
    orig_arr = np.array(internal_colors.convert("RGBA"))
    H, W, _ = orig_arr.shape

    # 8-way neighbors
    neighbors_8 = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    def is_thick_outline(x, y):
        """
        True if final_arr[y, x] has alpha>0 AND
        all 8 neighbors are also alpha>0 => no transparency around => internal outline pixel.
        """
        if final_arr[y, x, 3] == 0:
            return False

        for dx, dy in neighbors_8:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if final_arr[ny, nx, 3] == 0:
                    return False
            else:
                # If out-of-bounds, treat as transparent (so the edge won't be considered thick)
                return False
        return True

    def gather_interior_colors(up_x, up_y, radius):
        """
        Gathers colors from 'orig_arr' in a (2*radius+1)x(2*radius+1) region centered at (up_x, up_y).
        We'll consider any pixel with alpha=255 as "interior."
        """
        gathered = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = up_x + dx, up_y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    if orig_arr[ny, nx, 3] == 255:  # fully opaque => likely interior
                        gathered.append(tuple(orig_arr[ny, nx]))  # (R,G,B,A)
        return gathered

    result_arr = final_arr.copy()

    for y in range(h):
        for x in range(w):
            if is_thick_outline(x, y):
                # This is an interior portion of a thick outline
                # => Attempt to fill with an interior color from original_upscaled
                found_colors = []
                for r in range(1, max_search_radius + 1):
                    found_colors = gather_interior_colors(x, y, r)
                    if found_colors:
                        break

                if found_colors:
                    color_counts = Counter(found_colors)
                    likely_color, _ = color_counts.most_common(1)[0]
                    result_arr[y, x] = likely_color
                else:
                    # If no color found, do nothing or make transparent:
                    # result_arr[y, x] = [0,0,0,0]
                    pass

    return Image.fromarray(result_arr, mode="RGBA")

def remove_l_corners(mask_img: Image.Image) -> Image.Image:
    """
    Scans a binary outline mask and removes 'L corners' in 2x2 blocks.
    Specifically, if a 2x2 block has exactly three 'on' pixels (forming an L shape),
    we remove (turn off) the corner pixel to leave a diagonal of two 'on' pixels.

    mask_img : A Pillow Image where 'on' pixels have alpha>0, or are otherwise nonzero.

    Returns : A new Pillow Image in RGBA where corner pixels are removed.
    """

    # Convert to RGBA to ensure alpha channel, then to a NumPy array
    rgba_arr = np.array(mask_img.convert("RGBA"))
    h, w, _ = rgba_arr.shape

    # Build a simple binary mask: 1=on if alpha>0, else 0
    in_mask = (rgba_arr[..., 3] > 0).astype(np.uint8)

    # We'll store results in out_mask
    out_mask = in_mask.copy()

    # Define the 2x2 block patterns and how to transform them
    # Key: (A,B,C,D) => Value: (A',B',C',D')
    # A=top-left, B=top-right, C=bottom-left, D=bottom-right
    # Each tuple is 0/1 for 'off'/'on'.
    # These 4 patterns represent the 3-on-1-off 'L shapes' in all orientations,
    # and we convert them to a 2-on diagonal.
    pattern_map = {
        (1, 1, 1, 0): (0, 1, 1, 0),  # remove top-left corner -> diagonal top-right to bottom-left
        (1, 1, 0, 1): (1, 0, 0, 1),  # remove top-right corner -> diagonal top-left to bottom-right
        (1, 0, 1, 1): (1, 0, 0, 1),  # remove bottom-left corner -> diagonal top-left to bottom-right
        (0, 1, 1, 1): (0, 1, 1, 0),  # remove bottom-right corner -> diagonal top-right to bottom-left
    }

    # Traverse every 2x2 block
    # We'll write changes to out_mask so the pass is consistent
    for y in range(h - 1):
        for x in range(w - 1):
            A = in_mask[y, x]
            B = in_mask[y, x + 1]
            C = in_mask[y + 1, x]
            D = in_mask[y + 1, x + 1]
            block = (A, B, C, D)

            if block in pattern_map:
                new_block = pattern_map[block]
                # Update out_mask
                out_mask[y, x] = new_block[0]
                out_mask[y, x + 1] = new_block[1]
                out_mask[y + 1, x] = new_block[2]
                out_mask[y + 1, x + 1] = new_block[3]
            else:
                # Otherwise, keep it as-is
                pass

    # Convert out_mask back into RGBA
    # alpha=255 if 'on', else 0, and R=G=B=0
    result_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    result_rgba[out_mask == 1, 3] = 255  # set alpha=255 for on pixels

    return Image.fromarray(result_rgba, mode="RGBA")

def set_interior_pixels(
    mask_img: Image.Image,
    fill_color: tuple = (0, 0, 0, 255)
) -> Image.Image:
    """
    Sets all pixels inside the mask outline to a specified fill color.

    Args:
        mask_img (PIL.Image.Image): The mask image in RGBA mode where the outline has alpha > 0.
        fill_color (tuple): The RGBA color to set for interior pixels. Default is (0, 0, 0, 255).
        output_img_path (str, optional): Path to save the output image. If None, the image is not saved.

    Returns:
        PIL.Image.Image: The modified image with interiors filled.
    """

    # Ensure the mask is in RGBA mode
    mask_rgba = mask_img.convert("RGBA")

    # Convert the mask image to a NumPy array
    mask_arr = np.array(mask_rgba)
    h, w, _ = mask_arr.shape

    # Create a binary mask: 1 for outline pixels (alpha > 0), 0 otherwise
    binary_mask = (mask_arr[..., 3] > 0).astype(np.uint8)

    # Use binary_fill_holes to identify interior regions
    # binary_fill_holes fills enclosed areas (interior) but leaves the outline intact
    filled_mask = binary_fill_holes(binary_mask).astype(np.uint8)

    # The interior is where filled_mask is 1 and binary_mask is 0
    interior_mask = filled_mask - binary_mask

    # Create a copy of the original mask to modify
    result_arr = mask_arr.copy()

    # Set interior pixels to the specified fill color
    result_arr[interior_mask == 1] = fill_color

    # Convert the NumPy array back to a Pillow Image
    result_img = Image.fromarray(result_arr, mode="RGBA")
    return result_img


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
    mask_arr = np.array(mask) if mask is not None else None
    # 5) Create a final array starting as a copy of the second image
    final_arr = second_arr.copy()

    # 6) For each pixel, conditionally replace
    for y in range(h):
        for x in range(w):
            m = mask_arr[y, x] if mask_arr is not None else 1
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

def paste_img(dest, src) -> Image.Image:
    new_layer = Image.new("RGBA", dest.size)
    new_layer.paste(src, (0, 0))
    return Image.alpha_composite(dest, new_layer)

def main():
    parser = argparse.ArgumentParser(description="Pixel Art Downscaling with RGBA & Small-Sprite Optimizations")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("output", help="Path to output image")
    parser.add_argument("-s", "--scale_factor", type=int, default=2,
                        help="Reduction ratio (default=2 => half width & height).")
    parser.add_argument("-o", "--operations", type=str, default="1,2",
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
    parser.add_argument("--process_outline", type=str, default=None,
                        help="Process sprite outline. (preserve/remove)")

    args = parser.parse_args()

    # Load original image in RGBA
    original_img = Image.open(args.input).convert("RGBA")

    # Parse operations
    ops = [op.strip() for op in args.operations.split(",")]

    downsampled_img = naive_downsample(original_img, args.scale_factor)

    external_outline_mask = None
    internal_outline_mask = None
    interior_img = None
    if args.process_outline is not None:
        external_outline_mask, _ = outline_mask_by_dominant_color(original_img, include_internal_outline=False, tolerance=0)
        internal_outline_mask, interior_img = outline_mask_by_dominant_color(original_img,
                                                                             include_internal_outline=True)

        # save 'de-outlined' image as original image and naive downsampled version.
        original_img = interior_img
        downsampled_img = naive_downsample(original_img, args.scale_factor,
                                           method=Image.Resampling.BILINEAR,
                                           remove_alpha=True)

        # # downscale_outline_components
        fill_threshold = args.scale_factor / 8
        external_outline_mask = downscale_outline_components(
            outline_mask=external_outline_mask,
            scale=args.scale_factor,
            fill_threshold=fill_threshold)
        # external_outline_mask = remove_l_corners(external_outline_mask)
        ff_mask = build_outline_aware_mask(external_outline_mask)

    else:
        ff_mask = floodfill_mask(downsampled_img)

    if DIAGNOSTICS:
        downsampled_img.save('downsampled_' + args.output)

        if internal_outline_mask:
            internal_outline_mask.save('internal_outline_mask' + args.output)
            internal_outline_mask = downscale_outline_components(
                outline_mask=internal_outline_mask,
                scale=args.scale_factor,
                fill_threshold=fill_threshold)
            internal_outline_mask.save('internal_outline_mask_sm_' + args.output)
        if interior_img:
            interior_img.save('interior_' + args.output)

        if external_outline_mask:
            external_outline_mask.save('external_outline_mask_sm_' + args.output)
        downsampled_img.save('downsampled_' + args.output)
        ff_mask.save('ff_mask_' + args.output)

    # perform requested operations
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

    if DIAGNOSTICS:
        result_img.save('pre_conditional_replace_' + args.output)

    result_img = conditional_replace(downsampled_img, result_img, ff_mask, args.alpha_min)
    if args.process_outline == "preserve":
        # we processed the outline separately
        result_img = paste_img(result_img, external_outline_mask)
    # result_img.save('with_outline_'+ args.output)
    # result_img = postprocess_thick_outline(result_img, interior_img, original_scale_factor)

    # match to palette
    if args.palette:
        palette_img = Image.open(args.palette)
        # result_img.save('pre-palette' + args.output)
        result_img = enforce_palette_lab_bulk(result_img, palette_img)
    # Save result
    result_img.save(args.output)


def naive_downsample(original_img, scale_factor, method=Image.Resampling.LANCZOS, remove_alpha=False):
    d_img = (original_img.resize(
        (original_img.width // scale_factor, original_img.height // scale_factor),
        method).convert("RGBA"))
    if remove_alpha:
        d_arr = np.array(d_img)
        h, w, _ = d_arr.shape
        for y in range(h):
            for x in range(w):
                a = d_arr[y, x, 3]
                if a > 0:
                    d_arr[y, x, 3] = 255
        d_img = Image.fromarray(d_arr).convert("RGBA")
    return d_img


def build_outline_aware_mask(external_outline_mask):
    ff_mask = set_interior_pixels(external_outline_mask, (0, 0, 0, 255))
    ff_arr = np.array(ff_mask.convert("RGBA"))
    h, w, _ = ff_arr.shape
    for y in range(h):
        for x in range(w):
            a = ff_arr[y, x, 3]
            if a == 0:
                ff_arr[y, x] = [255, 255, 255, 255]
    ff_mask = Image.fromarray(ff_arr, mode="RGBA").convert("L")
    ff_mask = ImageOps.invert(ff_mask)
    return ff_mask


##


if __name__ == "__main__":
    main()



# for character artwork:
    # python pix.py <input>.png <output>.png -o 1 -s 2

# for tile or large artwork
    # python pix.py <input>.png <output>.png -o 1,2 -s 2