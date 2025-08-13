#!/usr/bin/env python3
import os
import numpy as np
from PIL import Image

def make_white_transparent(image_path, output_path=None, threshold=240):
    """
    Convert white background to transparent in an image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save transparent image (optional)
        threshold: RGB threshold for white detection (0-255)
    
    Returns:
        PIL Image with transparent background
    """
    try:
        # Open image
        img = Image.open(image_path)
        
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Convert to numpy array
        data = np.array(img)
        
        # Create mask for white pixels (where R, G, B are all above threshold)
        white_mask = (data[:, :, 0] >= threshold) & \
                     (data[:, :, 1] >= threshold) & \
                     (data[:, :, 2] >= threshold)
        
        # Set alpha channel to 0 for white pixels (transparent)
        data[white_mask, 3] = 0
        
        # Convert back to PIL Image
        transparent_img = Image.fromarray(data, 'RGBA')
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            transparent_img.save(output_path)
            print(f"Saved transparent image: {output_path}")
        
        return transparent_img
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_directory(input_dir, output_dir=None, threshold=240):
    """
    Process all images in a directory to make white backgrounds transparent.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save transparent images (optional)
        threshold: RGB threshold for white detection (0-255)
    """
    if output_dir is None:
        output_dir = input_dir + "_transparent"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(image_extensions)]
    
    total_files = len(image_files)
    print(f"Found {total_files} images to process...")
    
    processed = 0
    failed = 0
    
    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        
        # Always save as PNG to preserve transparency
        base_name = os.path.splitext(filename)[0]
        output_filename = base_name + '.png'
        output_path = os.path.join(output_dir, output_filename)
        
        result = make_white_transparent(input_path, output_path, threshold)
        
        if result is not None:
            processed += 1
            if processed % 100 == 0:  # Progress update every 100 images
                print(f"Processed {processed}/{total_files} images...")
        else:
            failed += 1
    
    print(f"\nCompleted processing:")
    print(f"  Successfully processed: {processed}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")

if __name__ == "__main__":
    # Directory containing the images with white backgrounds
    input_directory = "/home/projects/bfi_viewer/app/backend/brainviewer/static/images_data/244"
    
    # Output directory for transparent images
    output_directory = "/home/projects/bfi_viewer/app/backend/brainviewer/static/images_data/244_transparent"
    
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    print("Converting white backgrounds to transparent...")
    
    # Process all images
    process_directory(input_directory, output_directory, threshold=240)
    
    print("Transparency conversion completed!")