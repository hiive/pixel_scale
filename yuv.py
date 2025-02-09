import numpy as np
from PIL import Image
import os

def rgb_to_yuv(image_array):
    """
    Converts an RGB image array to YUV color space using the BT.601 standard.

    Args:
        image_array (numpy.ndarray): Input image array in RGB format.

    Returns:
        numpy.ndarray: Image array in YUV format.
    """
    # Define the conversion matrix from RGB to YUV (BT.601)
    conversion_matrix = np.array([[0.299, 0.587, 0.114],
                                  [-0.14713, -0.28886, 0.436],
                                  [0.615, -0.51499, -0.10001]])

    # Reshape the image array to (num_pixels, 3) for matrix multiplication
    flat_image = image_array.reshape((-1, 3)).astype(np.float32)

    # Apply the conversion matrix
    yuv_flat = np.dot(flat_image, conversion_matrix.T)

    # Reshape back to the original image shape
    yuv = yuv_flat.reshape(image_array.shape)

    # Clip the values to [0, 255] and convert to uint8
    yuv = np.clip(yuv, 0, 255).astype(np.uint8)

    return yuv


def extract_channels(yuv_array):
    """
    Extracts the Y, U, and V channels from a YUV image array.

    Args:
        yuv_array (numpy.ndarray): Image array in YUV format.

    Returns:
        tuple: (Y_channel, U_channel, V_channel) as separate 2D numpy arrays.
    """
    Y = yuv_array[..., 0]
    U = yuv_array[..., 1]
    V = yuv_array[..., 2]
    return Y, U, V


def save_grayscale_image(channel_array, output_path):
    """
    Saves a single-channel image array as a grayscale image.

    Args:
        channel_array (numpy.ndarray): 2D array representing the grayscale channel.
        output_path (str): Path to save the grayscale image.
    """
    # Create a PIL Image from the array
    grayscale_image = Image.fromarray(channel_array, mode='L')

    # Save the image
    grayscale_image.save(output_path)
    print(f"Saved grayscale channel to: {output_path}")


def split_yuv_channels(input_image_path, output_directory, scale_factor=None):
    """
    Splits an input image into Y, U, and V grayscale channels and saves them.

    Args:
        input_image_path (str): Path to the input image.
        output_directory (str): Directory where the output channel images will be saved.
        scale_factor (int, optional): If provided, the image will be resized by this factor before processing.

    Returns:
        dict: Paths to the saved Y, U, and V channel images.
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Load the image and convert to RGB
    with Image.open(input_image_path) as img:
        img = img.convert('RGB')

        # Optionally resize the image
        if scale_factor is not None and scale_factor > 0:
            new_size = (img.width * scale_factor, img.height * scale_factor)
            img = img.resize(new_size, resample=Image.NEAREST)
            print(f"Resized image to: {new_size}")

        # Convert image to NumPy array
        image_array = np.array(img)

    # Convert RGB to YUV
    yuv_array = rgb_to_yuv(image_array)
    print("Converted RGB to YUV.")

    # Extract Y, U, V channels
    Y, U, V = extract_channels(yuv_array)
    print("Extracted Y, U, and V channels.")

    # Define output paths
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    y_path = os.path.join(output_directory, f"{base_name}_Y.png")
    u_path = os.path.join(output_directory, f"{base_name}_U.png")
    v_path = os.path.join(output_directory, f"{base_name}_V.png")

    # Save each channel as a grayscale image
    save_grayscale_image(Y, y_path)
    save_grayscale_image(U, u_path)
    save_grayscale_image(V, v_path)

    return {'Y': y_path, 'U': u_path, 'V': v_path}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python yuv.py <input_image> <output_directory> [scale_factor]")
        print("Example: python yuv.py input.png output_channels 2")
        sys.exit(1)

    input_image_path = sys.argv[1]
    output_directory = sys.argv[2]
    scale = int(sys.argv[3]) if len(sys.argv) > 3 else None

    channels = split_yuv_channels(input_image_path, output_directory, scale_factor=scale)
    print("Channel separation completed.")
    print(f"Y channel saved at: {channels['Y']}")
    print(f"U channel saved at: {channels['U']}")
    print(f"V channel saved at: {channels['V']}")