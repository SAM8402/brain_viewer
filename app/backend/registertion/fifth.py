import json
import numpy as np
import argparse
import os # For extracting filename without extension

def chain_canvas_to_bfw_transformations(
    canvas_to_stack_json_path,
    stack_to_bfw_json_path,
    output_json_path,
    sample_id="142", # The key in stack_to_bfw_json that contains the slice data
    has_cv2=False
):
    """
    Chains transformations: Canvas -> Stack -> BFW.

    Args:
        canvas_to_stack_json_path (str): Path to JSON with Canvas to Stack transforms
                                         (e.g., "222_c_to_s.json").
        stack_to_bfw_json_path (str): Path to JSON with Stack (Nissl) to BFW transforms
                                      (e.g., "all_transformation_data.json").
        output_json_path (str): Path for the output JSON
                                (e.g., "222_original_to_bfiw.json").
        sample_id (str): The key in stack_to_bfw_json (e.g., "222") under which
                         slice transformation data is stored.
    """
    try:
        with open(canvas_to_stack_json_path, 'r') as f:
            c_to_s_data = json.load(f)
        print(f"Successfully loaded Canvas-to-Stack data from: {canvas_to_stack_json_path}")
    except FileNotFoundError:
        print(f"Error: Canvas-to-Stack JSON file not found at '{canvas_to_stack_json_path}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{canvas_to_stack_json_path}'.")
        return

    try:
        with open(stack_to_bfw_json_path, 'r') as f:
            s_to_b_data_full = json.load(f)
        print(f"Successfully loaded Stack-to-BFW data from: {stack_to_bfw_json_path}")
    except FileNotFoundError:
        print(f"Error: Stack-to-BFW JSON file not found at '{stack_to_bfw_json_path}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{stack_to_bfw_json_path}'.")
        return

    if sample_id not in s_to_b_data_full:
        print(f"Error: Sample ID '{sample_id}' not found in '{stack_to_bfw_json_path}'. Please verify the sample ID or the contents of the JSON file.")
        return

    s_to_b_slice_data = s_to_b_data_full[sample_id].get("slices", [])
    if not s_to_b_slice_data:
        print(f"Warning: No 'slices' data found for sample ID '{sample_id}' in '{stack_to_bfw_json_path}'.")
        # Proceeding will result in no matches, but let's handle it gracefully.

    # Create a lookup for stack-to-bfw transforms by slice number for efficiency
    s_to_b_lookup = {}
    for slice_info in s_to_b_slice_data:
        if "slice_num" in slice_info and "H_nissl_to_bfw" in slice_info and "H_bfw_to_nissl" in slice_info:
            s_to_b_lookup[str(slice_info["slice_num"])] = {
                "H_s_b": np.array(slice_info["H_nissl_to_bfw"], dtype=np.float64),
                "H_b_s": np.array(slice_info["H_bfw_to_nissl"], dtype=np.float64),
                # You can include other s_to_b metadata if needed
                "bfw_dims_original": s_to_b_data_full[sample_id].get("bfw_dims_original"),
                "interpolated_bfw_css_rotation_deg": slice_info.get("interpolated_bfw_css_rotation_deg")
            }
        else:
            print(f"Warning: Incomplete slice data in Stack-to-BFW JSON for slice_num around {slice_info.get('slice_num', 'Unknown')}. Skipping this slice.")


    if not s_to_b_lookup:
         print(f"No valid Stack-to-BFW transformations could be loaded from '{stack_to_bfw_json_path}' for sample ID '{sample_id}'. Output will be empty or incomplete.")
         # Consider exiting if this is critical, or continue to process c_to_s_data and mark missing.

    canvas_to_bfw_results = {}
    processed_count = 0
    missing_c_s_data_count = 0
    missing_s_b_data_count = 0
    error_count = 0

    print(f"\nProcessing {len(c_to_s_data)} entries from Canvas-to-Stack data...")

    for image_filename_with_ext, c_s_entry in c_to_s_data.items():
        if not isinstance(c_s_entry, dict):
            print(f"Warning: Entry for '{image_filename_with_ext}' in C-S data is not a dictionary. Skipping.")
            continue

        # Extract slice number from image_filename_with_ext (e.g., "bfi-920.png" -> "920")
        filename_base = os.path.splitext(image_filename_with_ext)[0]
        if filename_base.startswith("bfi-"):
            slice_num_str = filename_base.replace("bfi-", "")
        else:
            slice_num_str = filename_base

        # Retrieve necessary data from c_s_entry
        H_c_s_raw = c_s_entry.get("H_canvas_to_stack")
        H_s_c_raw = c_s_entry.get("H_stack_to_canvas")
        corners_canvas_raw = c_s_entry.get("corners_canvas_overlap")
        corners_stack_raw = c_s_entry.get("corners_stack_overlap") # These are corners *of the C-S overlap* in S coords

        new_entry = {
            "canvas_path": c_s_entry.get("canvas_path"),
            "stack_path_intermediate": c_s_entry.get("stack_path"), # Keep for reference
            # "bfw_path": None, # Placeholder, if you have BFW image paths
            "status": "pending",
            "error_message": None,
            "H_canvas_to_bfw": None,
            "H_bfw_to_canvas": None,
            "corners_canvas_overlap": corners_canvas_raw, # This remains the primary overlap in canvas coords
            "corners_bfw_overlap": None, # This will be the C-S overlap transformed to BFW coords
        }
        # Copy other relevant metadata from c_s_entry if needed
        if "status" in c_s_entry and c_s_entry["status"] == "error":
             new_entry["status"] = "error"
             new_entry["error_message"] = f"Original C-S processing failed: {c_s_entry.get('error_message', 'Unknown C-S error')}"
             missing_c_s_data_count +=1


        if H_c_s_raw is None or H_s_c_raw is None or corners_canvas_raw is None or corners_stack_raw is None:
            if new_entry["status"] != "error": # Don't overwrite previous error
                new_entry["status"] = "error"
                new_entry["error_message"] = "Missing C-S transformation data."
            missing_c_s_data_count += 1
            # Create clean key: remove "bfi-" prefix and change extension to .jpg
            if filename_base.startswith("bfi-"):
                clean_key = slice_num_str + ".jpg"
            else:
                clean_key = slice_num_str + ".jpg"
            canvas_to_bfw_results[clean_key] = new_entry
            continue # Cannot proceed without C-S data

        # Check if corresponding S-B data exists
        if slice_num_str not in s_to_b_lookup:
            new_entry["status"] = "error"
            new_entry["error_message"] = f"No Stack-to-BFW transformation data found for slice {slice_num_str}."
            missing_s_b_data_count += 1
            # Create clean key: remove "bfi-" prefix and change extension to .jpg
            if filename_base.startswith("bfi-"):
                clean_key = slice_num_str + ".jpg"
            else:
                clean_key = slice_num_str + ".jpg"
            canvas_to_bfw_results[clean_key] = new_entry
            continue # Cannot proceed without S-B data

        s_b_transforms = s_to_b_lookup[slice_num_str]
        H_s_b = s_b_transforms["H_s_b"]
        H_b_s = s_b_transforms["H_b_s"]
        
        # Add BFW metadata to the entry
        new_entry["bfw_dims_original"] = s_b_transforms.get("bfw_dims_original")
        new_entry["interpolated_bfw_css_rotation_deg"] = s_b_transforms.get("interpolated_bfw_css_rotation_deg")


        try:
            H_c_s = np.array(H_c_s_raw, dtype=np.float64)
            H_s_c = np.array(H_s_c_raw, dtype=np.float64)
            corners_stack_for_transform = np.array(corners_stack_raw, dtype=np.float64)

            # 1. Calculate H_canvas_to_bfw
            H_c_b = H_s_b @ H_c_s
            new_entry["H_canvas_to_bfw"] = H_c_b.tolist()

            # 2. Calculate H_bfw_to_canvas
            H_b_c = H_s_c @ H_b_s
            new_entry["H_bfw_to_canvas"] = H_b_c.tolist()

            # 3. Transform corners_stack_overlap (from C-S context) to BFW coordinates
            if has_cv2 and corners_stack_for_transform.ndim == 2 and corners_stack_for_transform.shape[0] > 0:
                 # Reshape for perspectiveTransform: (N, 1, 2)
                corners_stack_reshaped = corners_stack_for_transform.reshape(-1, 1, 2)
                corners_bfw_transformed = cv2.perspectiveTransform(corners_stack_reshaped, H_s_b)
                if corners_bfw_transformed is not None:
                    new_entry["corners_bfw_overlap"] = corners_bfw_transformed.reshape(-1, 2).tolist()
                else:
                    print(f"Warning: cv2.perspectiveTransform returned None for BFW corners for {image_filename_with_ext}")
                    new_entry["corners_bfw_overlap"] = None # Or keep as empty list/error
            else:
                # Skip corner transformation without OpenCV or use original corners
                new_entry["corners_bfw_overlap"] = corners_canvas_raw  # Use canvas corners as fallback


            if new_entry["status"] != "error": # Don't overwrite if C-S was already an error
                new_entry["status"] = "success"
            processed_count += 1

        except Exception as e:
            new_entry["status"] = "error"
            new_entry["error_message"] = f"Error during matrix multiplication or corner transformation: {e}"
            error_count += 1
            print(f"Error processing {image_filename_with_ext}: {e}")


        # Create clean key: remove "bfi-" prefix and change extension to .jpg
        if filename_base.startswith("bfi-"):
            clean_key = slice_num_str + ".jpg"
        else:
            clean_key = slice_num_str + ".jpg"
        canvas_to_bfw_results[clean_key] = new_entry

    print(f"\nChaining complete. Summary:")
    print(f"  Successfully processed entries: {processed_count}")
    print(f"  Entries with missing/failed C-S data (not processed for C-B): {missing_c_s_data_count}")
    print(f"  Entries missing corresponding S-B data: {missing_s_b_data_count}")
    print(f"  Entries with errors during C-B calculation: {error_count}")


    try:
        with open(output_json_path, 'w') as f:
            json.dump(canvas_to_bfw_results, f, indent=4)
        print(f"\nSuccessfully saved Canvas-to-BFW transformations to '{output_json_path}'")
    except IOError:
        print(f"Error: Could not write the output JSON to '{output_json_path}'. Check permissions or path.")
    except TypeError as e:
        print(f"Error serializing data to JSON: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chains Canvas-to-Stack and Stack-to-BFW transformations."
    )
    parser.add_argument(
        "canvas_to_stack_json",
        help="Path to the JSON file with Canvas-to-Stack transformations (e.g., 222_c_to_s.json)."
    )
    parser.add_argument(
        "stack_to_bfw_json",
        help="Path to the JSON file with Stack(Nissl)-to-BFW transformations (e.g., all_transformation_data.json)."
    )
    parser.add_argument(
        "output_json", 
        help="Path for the output JSON file (e.g., 222_original_to_bfiw.json)."
    )
    parser.add_argument(
        "--sample_id",
        default="142",
        help="The sample ID key in the stack_to_bfw_json (default: '222')."
    )

    args = parser.parse_args()

    # We need OpenCV for perspectiveTransform if we transform corners
    try:
        import cv2
        HAS_CV2 = True
    except ImportError:
        print("Warning: OpenCV not available. Corner transformations will be skipped.")
        HAS_CV2 = False


    chain_canvas_to_bfw_transformations(
        args.canvas_to_stack_json,
        args.stack_to_bfw_json,
        args.output_json,
        sample_id=args.sample_id,
        has_cv2=HAS_CV2
    )

    # --- Example Dummy File Generation (for testing) ---
    # To test, create dummy files:
    # 1. dummy_c_s.json (Canvas to Stack)
    # {
    #   "1.jpg": {
    #     "canvas_path": "canvas/1.jpg", "stack_path": "stack/1.jpg", "status": "success",
    #     "H_canvas_to_stack": [[1,0,10],[0,1,20],[0,0,1]],
    #     "H_stack_to_canvas": [[1,0,-10],[0,1,-20],[0,0,1]],
    #     "corners_canvas_overlap": [[0,0],[100,0],[100,100],[0,100]],
    #     "corners_stack_overlap": [[10,20],[110,20],[110,120],[10,120]]
    #   },
    #   "2.jpg": {
    #     "canvas_path": "canvas/2.jpg", "stack_path": "stack/2.jpg", "status": "success",
    #     "H_canvas_to_stack": [[1,0,5],[0,1,15],[0,0,1]],
    #     "H_stack_to_canvas": [[1,0,-5],[0,1,-15],[0,0,1]],
    #     "corners_canvas_overlap": [[0,0],[50,0],[50,50],[0,50]],
    #     "corners_stack_overlap": [[5,15],[55,15],[55,65],[5,65]]
    #   },
    #   "3.jpg": { /* No S-B data for this one */
    #     "canvas_path": "canvas/3.jpg", "stack_path": "stack/3.jpg", "status": "success",
    #     "H_canvas_to_stack": [[1,0,1],[0,1,1],[0,0,1]], "H_stack_to_canvas": [[1,0,-1],[0,1,-1],[0,0,1]],
    #     "corners_canvas_overlap": [[0,0],[10,0],[10,10],[0,10]], "corners_stack_overlap": [[1,1],[11,1],[11,11],[1,11]]
    #   }
    # }
    #
    # 2. dummy_s_b.json (Stack to BFW)
    # {
    #   "222": {
    #     "nissl_dims_original": [1000,1000], "bfw_dims_original": [2000,2000],
    #     "slices": [
    #       { "slice_num": 1, "H_nissl_to_bfw": [[2,0,50],[0,2,60],[0,0,1]], "H_bfw_to_nissl": [[0.5,0,-25],[0,0.5,-30],[0,0,1]] },
    #       { "slice_num": 2, "H_nissl_to_bfw": [[2,0,5],[0,2,6],[0,0,1]], "H_bfw_to_nissl": [[0.5,0,-2.5],[0,0.5,-3],[0,0,1]] }
    #       /* Slice 10 from your example isn't in the dummy c_s data above */
    #     ]
    #   }
    # }
    #
    # Command:
    # python your_script_name.py dummy_c_s.json dummy_s_b.json dummy_c_b_output.json --sample_id 222
    #
    # Expected for "1.jpg" in dummy_c_b_output.json:
    # H_c_s = [[1,0,10],[0,1,20],[0,0,1]]
    # H_s_b = [[2,0,50],[0,2,60],[0,0,1]]
    # H_c_b = H_s_b @ H_c_s = [[2,0,50],[0,2,60],[0,0,1]] @ [[1,0,10],[0,1,20],[0,0,1]]
    #       = [[2*1+0*0+50*0, 2*0+0*1+50*0, 2*10+0*20+50*1],
    #          [0*1+2*0+60*0, 0*0+2*1+60*0, 0*10+2*20+60*1],
    #          [0*1+0*0+1 *0, 0*0+0*1+1 *0, 0*10+0*20+1 *1]]
    #       = [[2, 0, 20+50], [0, 2, 40+60], [0, 0, 1]]
    #       = [[2, 0, 70],    [0, 2, 100],   [0, 0, 1]]
    # corners_stack_overlap for 1.jpg = [[10,20],[110,20],[110,120],[10,120]]
    # corners_bfw_overlap = transform these with H_s_b:
    #   [10,20] -> [2*10+50, 2*20+60] = [70, 100]
    #   [110,20]-> [2*110+50,2*20+60] = [270,100]
    #   etc.
