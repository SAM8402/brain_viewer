import os
import json
import numpy as np
from PIL import Image, ImageOps


def update_json_with_crop_values(image_path, top, left):
    """
    Update the 222_original_to_bfiw.json file by subtracting crop values from H_canvas_to_bfw.

    Args:
        image_path: Path to the image being cropped
        top: Top crop value to subtract from H_canvas_to_bfw[1][2]
        left: Left crop value to subtract from H_canvas_to_bfw[0][2]
    """
    # Extract image filename without extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Convert bfi-XXX or nissl-XXX to XXX.jpg format for JSON lookup
    if image_name.startswith('bfi-') or image_name.startswith('nissl-'):
        # Extract the numeric part
        numeric_part = image_name.split('-')[1]
        json_key = f"{numeric_part}.jpg"
    else:
        json_key = image_name

    # Path to JSON file
    json_path = "/home/projects/bfi_viewer/app/backend/brainviewer/data/222_original_to_bfiw.json"

    try:
        # Load JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Update if image exists in JSON
        if json_key in data:
            if "H_canvas_to_bfw" in data[json_key]:
                # Get original values
                orig_left_val = data[json_key]["H_canvas_to_bfw"][0][2]
                orig_top_val = data[json_key]["H_canvas_to_bfw"][1][2]

                # Subtract left from first array's last value
                data[json_key]["H_canvas_to_bfw"][0][2] -= left
                # Subtract top from second array's last value
                data[json_key]["H_canvas_to_bfw"][1][2] -= top

                print(f"Updated {image_name} (JSON key: {json_key}):")
                print(
                    f"  H_canvas_to_bfw[0][2]: {orig_left_val} - {left} = {data[json_key]['H_canvas_to_bfw'][0][2]}")
                print(
                    f"  H_canvas_to_bfw[1][2]: {orig_top_val} - {top} = {data[json_key]['H_canvas_to_bfw'][1][2]}")
        else:
            print(
                f"Warning: {image_name} (JSON key: {json_key}) not found in JSON file")

        # Save updated JSON
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)

    except Exception as e:
        print(f"Error updating JSON for {image_name}: {e}")


def crop_to_content(image_path, output_path=None, background_threshold=240, edge_detection=True):
    """
    Crop an image to its content by removing background areas.
    Uses edge detection and morphological operations for better results.

    Args:
        image_path: Path to the input image
        output_path: Path to save the cropped image (optional)
        background_threshold: Threshold for background detection (0-255)
        edge_detection: Use edge detection for better boundary detection

    Returns:
        Cropped image as PIL Image object
    """
    # Load image
    try:
        img = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Could not load image: {image_path} - {e}")

    width, height = img.size
    has_alpha = img.mode in ('RGBA', 'LA', 'PA')

    # For large images, use downsampling to find bounds efficiently
    max_size = 800  # Reduced for better performance
    if max(width, height) > max_size:
        # Calculate scale factor
        scale_factor = max_size / max(width, height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Downsample for analysis
        small_img = img.resize((new_width, new_height), Image.LANCZOS)
    else:
        # For smaller images, process directly
        small_img = img
        scale_factor = 1

    # Handle images with alpha channel differently
    if has_alpha:
        # Extract alpha channel for transparency detection
        if small_img.mode == 'RGBA':
            r, g, b, a = small_img.split()
            alpha_array = np.array(a)
        elif small_img.mode == 'LA':
            l, a = small_img.split()
            alpha_array = np.array(a)
        elif small_img.mode == 'PA':
            p, a = small_img.split()
            alpha_array = np.array(a)

        # For transparent images, content is where alpha > 0
        content_mask = alpha_array > 0

        # Also check RGB values for semi-transparent pixels
        if small_img.mode == 'RGBA':
            rgb_array = np.array(small_img.convert('RGB'))
            gray_array = np.dot(rgb_array[..., :3], [0.299, 0.587, 0.114])

            # For semi-transparent pixels, also check if they're not background color
            semi_transparent_mask = (alpha_array > 64) & (alpha_array < 255)
            color_content_mask = gray_array < background_threshold
            content_mask = content_mask | (
                semi_transparent_mask & color_content_mask)

        print(
            f"Image has alpha channel. Content pixels: {np.sum(content_mask)}")
    else:
        # Handle images without alpha - detect black or white backgrounds
        gray = ImageOps.grayscale(small_img)
        gray_array = np.array(gray)

        mean_val = np.mean(gray_array)
        min_val = np.min(gray_array)
        max_val = np.max(gray_array)

        # Check if background is black or white
        black_pixels = np.sum(gray_array <= 10)  # Nearly black
        white_pixels = np.sum(gray_array >= 245)  # Nearly white
        total_pixels = gray_array.size

        black_percentage = black_pixels / total_pixels
        white_percentage = white_pixels / total_pixels

        print(
            f"Gray stats - min: {min_val}, max: {max_val}, mean: {mean_val:.1f}")
        print(
            f"Black pixels: {black_percentage:.1%}, White pixels: {white_percentage:.1%}")

        if black_percentage > 0.5:  # Mostly black background
            # Content is anything significantly brighter than black
            threshold = max(15, min_val + 10)  # At least 15, or min + 10
            content_mask = gray_array > threshold
            print(f"Detected black background. Using threshold > {threshold}")
        elif white_percentage > 0.5:  # Mostly white background
            # Content is anything significantly darker than white
            threshold = min(240, max_val - 10)  # At most 240, or max - 10
            content_mask = gray_array < threshold
            print(f"Detected white background. Using threshold < {threshold}")
        else:
            # Mixed background - use adaptive threshold
            if mean_val > 200:  # Bright image
                adaptive_threshold = mean_val - 15
                content_mask = gray_array < adaptive_threshold
            else:
                adaptive_threshold = min(background_threshold, mean_val + 25)
                content_mask = gray_array < adaptive_threshold
            print(
                f"Mixed background. Using threshold < {adaptive_threshold:.1f}")

    # Remove small noise using morphological operations
    from scipy import ndimage
    # Remove small isolated pixels
    content_mask = ndimage.binary_opening(
        content_mask, structure=np.ones((3, 3)))
    # Fill small holes
    content_mask = ndimage.binary_closing(
        content_mask, structure=np.ones((5, 5)))

    if not np.any(content_mask):
        print(f"Warning: No content found in {image_path}")
        return img

    # Find content regions and get the largest connected component
    labeled_array, num_features = ndimage.label(content_mask)
    if num_features > 1:
        # Keep only the largest connected component
        sizes = ndimage.sum(content_mask, labeled_array,
                            range(num_features + 1))
        largest_component = np.argmax(sizes[1:]) + 1  # Skip background (0)
        content_mask = labeled_array == largest_component

    # Find bounding box
    content_coords = np.where(content_mask)
    if len(content_coords[0]) == 0:
        print(f"Warning: No content found after filtering in {image_path}")
        return img

    top_small = int(np.min(content_coords[0]))
    bottom_small = int(np.max(content_coords[0]))
    left_small = int(np.min(content_coords[1]))
    right_small = int(np.max(content_coords[1]))

    # Scale back to original image coordinates if needed
    if scale_factor != 1:
        top = int(top_small / scale_factor)
        bottom = int(bottom_small / scale_factor)
        left = int(left_small / scale_factor)
        right = int(right_small / scale_factor)
    else:
        top, bottom, left, right = top_small, bottom_small, left_small, right_small

    # Add padding (increased to prevent over-cropping)
    padding_y = max(10, int(0.03 * height))
    padding_x = max(10, int(0.03 * width))

    # Apply padding while staying within image bounds
    left = max(0, left - padding_x)
    right = min(width, right + padding_x)
    top = max(0, top - padding_y)
    bottom = min(height, bottom + padding_y)

    print(f"Image mode: {img.mode}, Has alpha: {has_alpha}")
    print(f"Crop bounds: ({left}, {top}) to ({right}, {bottom})")
    print(
        f"Original size: {width}x{height}, Cropped size: {right-left}x{bottom-top}")

    # Crop the original image
    cropped = img.crop((left, top, right, bottom))
    print("top ", top, "bottom ", bottom, "left ", left, "right ", right)

    # Update JSON file with crop values
    update_json_with_crop_values(image_path, top, left)

    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cropped.save(output_path)
        print(f"Cropped image saved to: {output_path}")

    return cropped


def crop_images_in_directory(input_dir, output_dir=None, background_threshold=240):
    """
    Crop all images in a directory to their content.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save cropped images (optional)
        background_threshold: Threshold for background detection (0-255)
    """
    if output_dir is None:
        output_dir = input_dir + "_cropped"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Supported image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

    # Process all images in the directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(image_extensions):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                crop_to_content(input_path, output_path, background_threshold)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# Example usage
if __name__ == "__main__":
    # Path to the image data directory
    # image_data_dir = "/home/projects/bfi_viewer/app/backend/brainviewer/static/images_data"

    # # Process entire directory from 222_uncrop
    # input_dir = os.path.join(image_data_dir, "222_uncrop")
    # output_dir = os.path.join(image_data_dir, "222_cropped")

    # print(f"Processing images from: {input_dir}")
    # print(f"Saving cropped images to: {output_dir}")
    # print("Updating JSON file with crop values...")

    # crop_images_in_directory(input_dir, output_dir)

    # print("Image cropping completed for all images!")

    image_data_dir = "/home/projects/bfi_viewer/app/backend/brainviewer/static/images_data"

    # Test with a single image first
    test_image = os.path.join(image_data_dir, "222_uncrop", "bfi-919.png")
    test_output = os.path.join(image_data_dir, "222_cropped", "bfi-919.png")

    print("Testing with a single image...")
    try:
        cropped = crop_to_content(test_image, test_output)
        print(
            f"Successfully cropped test image. Original size: {Image.open(test_image).size}, Cropped size: {cropped.size}")
    except Exception as e:
        print(f"Error: {e}")

    # Process entire directory - NOW ENABLED
    input_dir = os.path.join(image_data_dir, "222_uncrop")
    output_dir = os.path.join(image_data_dir, "222_cropped")
    print(f"Processing images from: {input_dir}")
    print(f"Saving cropped images to: {output_dir}")
    print("Updating JSON file with crop values...")
    crop_images_in_directory(input_dir, output_dir)

    print("Image cropping completed for all images!")