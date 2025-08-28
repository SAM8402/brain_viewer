import cv2
import numpy as np
import os
import json
from tqdm import tqdm


# --- Your provided function (unchanged) ---
def find_transformation_and_overlap(img1, img2, use_sift=True, ratio_thresh=0.75):
    """
    Finds transformation matrices between img1 and img2, and the bounding
    quadrilateral of their overlapping region.

    Returns:
      - H_1_to_2: 3x3 homography matrix from img1 to img2
      - H_2_to_1: 3x3 homography matrix from img2 to img1
      - src_corners_overlap: 4x2 ndarray of corner points of the overlap in img1
      - dst_corners_overlap: 4x2 ndarray of corresponding corner points in img2
    """
    # Ensure images are grayscale for feature detection if they are not
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1.copy()
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2.copy()

    # 1. Detect and describe features
    if use_sift and hasattr(cv2, 'SIFT_create'):
        detector = cv2.SIFT_create()
        # print("Using SIFT detector.")
    elif use_sift and hasattr(cv2, 'xfeatures2d') and hasattr(cv2.xfeatures2d, 'SIFT_create'):
        detector = cv2.xfeatures2d.SIFT_create() # For older OpenCV contrib
        # print("Using SIFT detector (from xfeatures2d).")
    else:
        if use_sift:
            print("SIFT not available for the current image pair, falling back to ORB.")
        # else:
            # print("Using ORB detector.")
        detector = cv2.ORB_create(nfeatures=5000) # Increased nfeatures for ORB

    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        # Try to be more specific if possible
        if des1 is None and des2 is None:
            msg = "Could not compute descriptors for both images."
        elif des1 is None:
            msg = "Could not compute descriptors for image 1 (thumbnail)."
        else: # des2 is None
            msg = "Could not compute descriptors for image 2 (stack)."
        raise ValueError(msg)

    if len(kp1) == 0 or len(kp2) == 0:
        if len(kp1) == 0 and len(kp2) == 0:
            msg = "No keypoints detected in both images."
        elif len(kp1) == 0:
            msg = "No keypoints detected in image 1 (thumbnail)."
        else: # len(kp2) == 0
            msg = "No keypoints detected in image 2 (stack)."
        raise ValueError(msg)

    # 2. Match descriptors
    matcher_type_used = "FLANN (SIFT)"
    if use_sift and (isinstance(detector, cv2.SIFT) or (hasattr(cv2, 'xfeatures2d') and hasattr(cv2.xfeatures2d, 'SIFT_create') and isinstance(detector, cv2.xfeatures2d.SIFT_create()))):
        # FLANN parameters for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else: # ORB
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) # crossCheck=False for knnMatch
        matcher_type_used = "BFMatcher (ORB)"

    # knnMatch for ratio test
    try:
        # Ensure descriptors are float32 for FLANN/SIFT
        if matcher_type_used.startswith("FLANN") and des1.dtype != np.float32:
            des1 = np.float32(des1)
        if matcher_type_used.startswith("FLANN") and des2.dtype != np.float32:
            des2 = np.float32(des2)
            
        raw_matches = matcher.knnMatch(des1, des2, k=2)
    except cv2.error as e:
        raise RuntimeError(f"Error during knnMatch with {matcher_type_used}: {e}. Des1 shape: {des1.shape}, dtype: {des1.dtype}. Des2 shape: {des2.shape}, dtype: {des2.dtype}.")


    # Lowe's ratio test
    good_matches = []
    if raw_matches is None: # Should not happen if knnMatch succeeded, but good to check
        raise RuntimeError("knnMatch returned None for raw_matches.")

    for m_n_pair in raw_matches:
        if m_n_pair is not None and len(m_n_pair) == 2: # Ensure we have two neighbors
            m, n = m_n_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        elif m_n_pair is not None and len(m_n_pair) == 1: # Only one match found
             good_matches.append(m_n_pair[0])


    MIN_MATCH_COUNT = 10 # Minimum matches for robust homography
    if len(good_matches) < MIN_MATCH_COUNT:
        raise ValueError(f"Not enough good matches found - {len(good_matches)}/{MIN_MATCH_COUNT} using {matcher_type_used}")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    # 3. Compute homography using RANSAC
    H_1_to_2, mask_ransac = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H_1_to_2 is None:
        raise RuntimeError("Homography estimation failed (H_1_to_2 is None)")

    try:
        H_2_to_1 = np.linalg.inv(H_1_to_2)
    except np.linalg.LinAlgError:
        raise RuntimeError("Homography matrix H_1_to_2 is singular, cannot compute inverse H_2_to_1.")

    # 4. Determine the bounding quadrilateral of the overlapping region
    h1, w1 = gray1.shape[:2]
    h2, w2 = gray2.shape[:2]

    # Create a mask for the entire img2
    mask_img2_full = np.full((h2, w2), 255, dtype=np.uint8)

    # Warp this mask to img1's perspective to see where img2 projects onto img1
    warped_mask_img2_in_img1_plane = cv2.warpPerspective(mask_img2_full, H_2_to_1, (w1, h1))

    # Find contours of this warped mask (this is the overlap region in img1's frame)
    contours, _ = cv2.findContours(warped_mask_img2_in_img1_plane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Try warping the other way to see if there's any projection at all
        mask_img1_full = np.full((h1, w1), 255, dtype=np.uint8)
        warped_mask_img1_in_img2_plane = cv2.warpPerspective(mask_img1_full, H_1_to_2, (w2, h2))
        contours_alt, _ = cv2.findContours(warped_mask_img1_in_img2_plane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_alt:
            raise ValueError("No overlapping region found after warping mask (checked both directions). Images might not overlap based on computed homography.")
        else: # Fallback: use the largest contour from the alternative warping
            # This case means the overlap is found when projecting img1 onto img2,
            # implying the previous `warped_mask_img2_in_img1_plane` might have been empty due to
            # img2 being entirely outside img1's FOV *after* transformation, or a degenerate homography.
            # We'll try to recover using the alternative projection.
            # This part is a bit heuristic and might need adjustment based on specific failure modes.
            print(f"Warning: No overlap contour from img2->img1 warp. Using img1->img2 warp to define overlap.")
            all_contour_points_alt = np.concatenate(contours_alt)
            if len(all_contour_points_alt) < 3:
                 raise ValueError("Alternative overlap region is too small (less than 3 points).")
            rect_dst_overlap_alt = cv2.minAreaRect(all_contour_points_alt)
            dst_corners_overlap = np.array(cv2.boxPoints(rect_dst_overlap_alt), dtype=np.float32)
            src_corners_overlap_transformed_alt = cv2.perspectiveTransform(dst_corners_overlap.reshape(-1,1,2), H_2_to_1)
            if src_corners_overlap_transformed_alt is None:
                raise RuntimeError("Perspective transform for src_corners_overlap (alternative) failed.")
            src_corners_overlap = src_corners_overlap_transformed_alt.reshape(4,2)
            return H_1_to_2, H_2_to_1, src_corners_overlap, dst_corners_overlap


    # Combine all contour points if multiple contours are found (usually one main one)
    all_contour_points = np.concatenate(contours)
    if len(all_contour_points) < 3: # minAreaRect needs at least 3 points
         raise ValueError("Overlap region is too small (less than 3 points).")


    # Get the minimum area rectangle enclosing these points in img1's coordinate system
    rect_src_overlap = cv2.minAreaRect(all_contour_points)
    src_corners_overlap_unordered = cv2.boxPoints(rect_src_overlap) # 4x2 ndarray
    src_corners_overlap = np.array(src_corners_overlap_unordered, dtype=np.float32)

    # Transform these src_corners_overlap to img2's coordinate system
    dst_corners_overlap_transformed = cv2.perspectiveTransform(src_corners_overlap.reshape(-1, 1, 2), H_1_to_2)
    if dst_corners_overlap_transformed is None:
        raise RuntimeError("Perspective transform for dst_corners_overlap failed.")
    dst_corners_overlap = dst_corners_overlap_transformed.reshape(4, 2)

    return H_1_to_2, H_2_to_1, src_corners_overlap, dst_corners_overlap

# --- Main processing script ---
def process_image_folders(thumbnail_folder, stack_folder, output_json_file, use_sift_by_default=True):
    """
    Processes image pairs from thumbnail and stack folders, calculates transformations,
    and saves results to a JSON file.
    """
    # --- Configuration ---
    THUMBNAIL_DIR = "/home/projects/bfi_viewer/app/backend/brainviewer/static/images_data/142"  # BFI images
    STACK_DIR = "/home/projects/bfi_viewer/app/backend/registertion/142"  # Nissl images
    OUTPUT_JSON_PATH = "./142_t_to_s.json"
    USE_SIFT = use_sift_by_default # True to try SIFT first, False to use ORB directly
    # ---------------------

    if not os.path.isdir(THUMBNAIL_DIR):
        print(f"Error: Thumbnail directory not found: {THUMBNAIL_DIR}")
        return
    if not os.path.isdir(STACK_DIR):
        print(f"Error: Stack directory not found: {STACK_DIR}")
        return

    all_results = {}
    valid_image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')

    # Get BFI files (thumbnail)
    bfi_files = [f for f in os.listdir(THUMBNAIL_DIR)
                 if os.path.isfile(os.path.join(THUMBNAIL_DIR, f)) and
                 f.startswith('bfi-') and f.lower().endswith(valid_image_extensions)]

    print(f"Found {len(bfi_files)} BFI images in {THUMBNAIL_DIR}.")
    if not bfi_files:
        print("No BFI image files found in the directory.")
        return

    for bfi_filename in tqdm(bfi_files, desc="Processing BFI to Nissl pairs"):
        print(bfi_filename)
        bfi_img_path = os.path.join(THUMBNAIL_DIR, bfi_filename)
        
        # Extract slice number from BFI filename (bfi-123.png -> 123)
        slice_number = bfi_filename.replace('bfi-', '').replace('.png', '').replace('.jpg', '')
        
        # Find corresponding Nissl image
        nissl_filename = f"nissl-{slice_number}.jpg"
        nissl_img_path = os.path.join(STACK_DIR, nissl_filename)

        current_pair_result = {
            "thumbnail_path": bfi_img_path,
            "stack_path": nissl_img_path,
            "status": "pending",
            "error_message": None,
            "H_thumbnail_to_stack": None,
            "H_stack_to_thumbnail": None,
            "corners_thumbnail_overlap": None,
            "corners_stack_overlap": None
        }

        if not os.path.isfile(nissl_img_path):
            print(f"Skipping {bfi_filename}: Corresponding Nissl file not found: {nissl_filename}")
            current_pair_result["status"] = "error"
            current_pair_result["error_message"] = "Corresponding Nissl image not found."
            all_results[bfi_filename] = current_pair_result
            continue

        # Load images
        img_bfi = cv2.imread(bfi_img_path)
        img_nissl = cv2.imread(nissl_img_path)

        if img_bfi is None:
            print(f"Skipping {bfi_filename}: Could not load BFI image from {bfi_img_path}.")
            current_pair_result["status"] = "error"
            current_pair_result["error_message"] = f"Could not load BFI image."
            all_results[bfi_filename] = current_pair_result
            continue
        if img_nissl is None:
            print(f"Skipping {bfi_filename}: Could not load Nissl image from {nissl_img_path}.")
            current_pair_result["status"] = "error"
            current_pair_result["error_message"] = f"Could not load Nissl image."
            all_results[bfi_filename] = current_pair_result
            continue

        try:
            # img1 is BFI (thumbnail), img2 is Nissl (stack)
            H_bfi_to_nissl, H_nissl_to_bfi, corners_bfi, corners_nissl = \
                find_transformation_and_overlap(img_bfi, img_nissl, use_sift=USE_SIFT)

            current_pair_result["status"] = "success"
            current_pair_result["H_thumbnail_to_stack"] = H_bfi_to_nissl.tolist() if H_bfi_to_nissl is not None else None
            current_pair_result["H_stack_to_thumbnail"] = H_nissl_to_bfi.tolist() if H_nissl_to_bfi is not None else None
            current_pair_result["corners_thumbnail_overlap"] = corners_bfi.tolist() if corners_bfi is not None else None
            current_pair_result["corners_stack_overlap"] = corners_nissl.tolist() if corners_nissl is not None else None

        except Exception as e:
            # print(f"Error processing {bfi_filename}: {e}") # tqdm might interfere with this print
            current_pair_result["status"] = "error"
            current_pair_result["error_message"] = str(e)
        
        all_results[bfi_filename] = current_pair_result

    # Save results to JSON
    try:
        with open(OUTPUT_JSON_PATH, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\nProcessing complete. Results saved to {OUTPUT_JSON_PATH}")
    except IOError:
        print(f"Error: Could not write JSON to {OUTPUT_JSON_PATH}. Check permissions or path.")
    except TypeError as e:
        print(f"Error serializing data to JSON: {e}. This might happen if some results are not JSON serializable.")


if __name__ == '__main__':
    # --- IMPORTANT: SET YOUR FOLDER PATHS AND OUTPUT FILE NAME HERE ---
    thumbnail_folder_path = "/home/projects/bfi_viewer/app/backend/brainviewer/static/images_data/142"  # BFI images
    stack_folder_path = "/home/projects/bfi_viewer/app/backend/registertion/142"  # Nissl images
    json_output_filename = "./142_t_to_s.json"
    # --- END OF CONFIGURATION ---

    # Call the main processing function
    # Set use_sift_by_default to True if you have OpenCV Contrib with SIFT,
    # otherwise set to False to use ORB.
    # The find_transformation_and_overlap function will automatically fall back
    # to ORB if SIFT is requested but not available.
    process_image_folders(thumbnail_folder_path, stack_folder_path, json_output_filename, use_sift_by_default=True)
