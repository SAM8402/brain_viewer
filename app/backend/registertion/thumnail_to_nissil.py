import cv2
import numpy as np
import os
import json
from tqdm import tqdm # For a progress bar, install with: pip install tqdm

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


    MIN_MATCH_COUNT = 10 # Increased minimum matches for robust homography
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
            print(f"Warning: No overlap contour from img2->img1 warp. Using img1->img2 warp to define overlap for {os.path.basename(img1_path) if 'img1_path' in locals() else 'img1'}")
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
    THUMBNAIL_DIR = "/home/projects/medimg/supriti/brain-registration/142/142_nisl_ser"
    STACK_DIR = "/home/projects/medimg/supriti/brain-registration/142/142_nissl_stack"
    OUTPUT_JSON_PATH = "/home/projects/medimg/supriti/brain-registration/Final_run_scripts/142_t_to_s.json"
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

    thumbnail_files = [f for f in os.listdir(THUMBNAIL_DIR)
                       if os.path.isfile(os.path.join(THUMBNAIL_DIR, f)) and
                       f.lower().endswith(valid_image_extensions)]

    print(f"Found {len(thumbnail_files)} potential images in {THUMBNAIL_DIR}.")
    if not thumbnail_files:
        print("No image files found in the thumbnail directory.")
        return

    for filename in tqdm(thumbnail_files, desc="Processing image pairs"):
        print(filename)
        thumb_img_path = os.path.join(THUMBNAIL_DIR, filename)
        stack_img_path = os.path.join(STACK_DIR, filename)

        current_pair_result = {
            "thumbnail_path": thumb_img_path,
            "stack_path": stack_img_path,
            "status": "pending",
            "error_message": None,
            "H_thumbnail_to_stack": None,
            "H_stack_to_thumbnail": None,
            "corners_thumbnail_overlap": None,
            "corners_stack_overlap": None
        }

        if not os.path.isfile(stack_img_path):
            print(f"Skipping {filename}: Corresponding file not found in stack folder.")
            current_pair_result["status"] = "error"
            current_pair_result["error_message"] = "Corresponding stack image not found."
            all_results[filename] = current_pair_result
            continue

        # Load images
        img_thumb = cv2.imread(thumb_img_path)
        img_stack = cv2.imread(stack_img_path)

        if img_thumb is None:
            print(f"Skipping {filename}: Could not load thumbnail image from {thumb_img_path}.")
            current_pair_result["status"] = "error"
            current_pair_result["error_message"] = f"Could not load thumbnail image."
            all_results[filename] = current_pair_result
            continue
        if img_stack is None:
            print(f"Skipping {filename}: Could not load stack image from {stack_img_path}.")
            current_pair_result["status"] = "error"
            current_pair_result["error_message"] = f"Could not load stack image."
            all_results[filename] = current_pair_result
            continue

        try:
            # img1 is thumbnail, img2 is stack
            H_thumb_to_stack, H_stack_to_thumb, corners_thumb, corners_stack = \
                find_transformation_and_overlap(img_thumb, img_stack, use_sift=USE_SIFT)

            current_pair_result["status"] = "success"
            current_pair_result["H_thumbnail_to_stack"] = H_thumb_to_stack.tolist() if H_thumb_to_stack is not None else None
            current_pair_result["H_stack_to_thumbnail"] = H_stack_to_thumb.tolist() if H_stack_to_thumb is not None else None
            current_pair_result["corners_thumbnail_overlap"] = corners_thumb.tolist() if corners_thumb is not None else None
            current_pair_result["corners_stack_overlap"] = corners_stack.tolist() if corners_stack is not None else None

        except Exception as e:
            # print(f"Error processing {filename}: {e}") # tqdm might interfere with this print
            current_pair_result["status"] = "error"
            current_pair_result["error_message"] = str(e)
        
        all_results[filename] = current_pair_result

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
    thumbnail_folder_path = "/home/projects/medimg/supriti/brain-registration/142/142_nisl_ser"
    stack_folder_path = "/home/projects/medimg/supriti/brain-registration/142/142_nissl_stack"
    json_output_filename = "/home/project/medimg/supriti/brain-registration/Final_run_scripts/142_t_to_s.json"
    # --- END OF CONFIGURATION ---
 
    # Example: Create dummy folders and images for testing
    # You should replace this with your actual paths
    if thumbnail_folder_path == "/home/projects/medimg/supriti/brain-registration/149/149_nissl_ser":
        print("INFO: Using dummy folders and images for demonstration.")
        print("Please update 'thumbnail_folder_path', 'stack_folder_path', and 'json_output_filename'.")

        # Create dummy directories
        base_test_dir = "test_image_processing"
        thumbnail_folder_path = os.path.join(base_test_dir, "thumbnails")
        stack_folder_path = os.path.join(base_test_dir, "stacks")
        json_output_filename = os.path.join(base_test_dir, "test_output.json")

        os.makedirs(thumbnail_folder_path, exist_ok=True)
        os.makedirs(stack_folder_path, exist_ok=True)

        # Create dummy images (similar to your original test but simpler for this)
        # Image 1 (thumbnail)
        img1_dummy = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(img1_dummy, (30, 30), (130, 130), (0, 255, 0), -1)
        cv2.putText(img1_dummy, "T1", (40,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
        cv2.imwrite(os.path.join(thumbnail_folder_path, "1.jpg"), img1_dummy)
        cv2.imwrite(os.path.join(thumbnail_folder_path, "2.png"), img1_dummy) # another one

        # Image 2 (stack - transformed version of thumbnail's content)
        img2_dummy = np.zeros((250, 300, 3), dtype=np.uint8)
        pts1_dummy = np.float32([[30, 30], [130, 30], [30, 130], [130, 130]])
        pts2_dummy = np.float32([[50, 40], [180, 50], [40, 170], [170, 180]])
        H_dummy_true = cv2.getPerspectiveTransform(pts1_dummy, pts2_dummy)
        
        # Content to warp from img1_dummy
        content_to_warp = np.zeros_like(img1_dummy)
        cv2.rectangle(content_to_warp, (30, 30), (130, 130), (0, 255, 0), -1)
        cv2.putText(content_to_warp, "T1", (40,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)

        warped_content_dummy = cv2.warpPerspective(content_to_warp, H_dummy_true, (img2_dummy.shape[1], img2_dummy.shape[0]))
        
        mask_warped_dummy = cv2.cvtColor(warped_content_dummy, cv2.COLOR_BGR2GRAY)
        _, mask_warped_dummy = cv2.threshold(mask_warped_dummy, 1, 255, cv2.THRESH_BINARY)
        img2_dummy_bg = img2_dummy.copy()
        img2_dummy = cv2.bitwise_and(img2_dummy_bg, img2_dummy_bg, mask=cv2.bitwise_not(mask_warped_dummy))
        img2_dummy = cv2.add(img2_dummy, warped_content_dummy)
        cv2.putText(img2_dummy, "S1", (200,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)


        cv2.imwrite(os.path.join(stack_folder_path, "1.jpg"), img2_dummy)
        # For 2.png, let's make it slightly different to test robustness or failure
        img2_dummy_diff = np.zeros((250, 300, 3), dtype=np.uint8)
        cv2.circle(img2_dummy_diff, (150,125), 50, (0,0,255), -1)
        cv2.imwrite(os.path.join(stack_folder_path, "2.png"), img2_dummy_diff) # No overlap here

        # A file only in thumbnails
        cv2.imwrite(os.path.join(thumbnail_folder_path, "only_thumb.jpg"), img1_dummy)
        
        print(f"Dummy files created in {base_test_dir}")
        print(f"Thumbnail folder: {thumbnail_folder_path}")
        print(f"Stack folder: {stack_folder_path}")
        print(f"Output JSON will be: {json_output_filename}")
        print("-" * 30)


    # Call the main processing function
    # Set use_sift_by_default to True if you have OpenCV Contrib with SIFT,
    # otherwise set to False to use ORB.
    # The find_transformation_and_overlap function will automatically fall back
    # to ORB if SIFT is requested but not available.
    process_image_folders(thumbnail_folder_path, stack_folder_path, json_output_filename, use_sift_by_default=True)

    # To clean up dummy files after testing (optional):
    # import shutil
    # if os.path.exists(base_test_dir) and thumbnail_folder_path == "path/to/your/thumbnail_folder": # safety check
    #     shutil.rmtree(base_test_dir)
    #     print(f"\nCleaned up dummy directory: {base_test_dir}")


