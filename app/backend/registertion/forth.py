import json
import numpy as np
import argparse

def convert_thumbnail_to_canvas_json(
    thumbnail_json_path, canvas_json_path, downsample_factor=64.0
):
    """
    Converts a JSON file with thumbnail-to-stack transformations
    to a new JSON file with original_canvas-to-stack transformations.

    Args:
        thumbnail_json_path (str): Path to the input JSON (e.g., "222_t_to_s.json").
        canvas_json_path (str): Path for the output JSON (e.g., "222_c_to_s.json").
        downsample_factor (float): The factor by which the original canvas images
                                   were downsampled to get the thumbnails (e.g., 64).
    """
    try:
        with open(thumbnail_json_path, 'r') as f:
            thumb_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input thumbnail JSON file not found at '{thumbnail_json_path}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{thumbnail_json_path}'. File might be corrupted.")
        return

    if not isinstance(thumb_data, dict):
        print(f"Error: Expected a dictionary at the top level of '{thumbnail_json_path}'.")
        return

    canvas_data = {}
    s = float(downsample_factor) # Ensure it's a float for division

    # Scaling matrix from Canvas to Thumbnail (points in canvas are divided by s)
    M_c_t = np.array([[1 / s, 0, 0],
                      [0, 1 / s, 0],
                      [0, 0, 1]], dtype=np.float64)

    # Scaling matrix from Thumbnail to Canvas (points in thumbnail are multiplied by s)
    M_t_c = np.array([[s, 0, 0],
                      [0, s, 0],
                      [0, 0, 1]], dtype=np.float64)

    print(f"Processing {len(thumb_data)} entries from '{thumbnail_json_path}'...")
    processed_count = 0
    skipped_count = 0

    for image_key, data_entry in thumb_data.items():
        if not isinstance(data_entry, dict):
            print(f"Warning: Entry for '{image_key}' is not a dictionary. Skipping.")
            skipped_count += 1
            continue

        new_entry = data_entry.copy() # Start with a copy

        # Check if essential transformation keys are present and not None
        h_t_s_raw = data_entry.get("H_thumbnail_to_stack")
        h_s_t_raw = data_entry.get("H_stack_to_thumbnail")
        corners_thumb_raw = data_entry.get("corners_thumbnail_overlap")
        # corners_stack_overlap remains the same, so just check if it exists
        corners_stack_raw = data_entry.get("corners_stack_overlap")


        if h_t_s_raw is None or h_s_t_raw is None or \
           corners_thumb_raw is None or corners_stack_raw is None:
            # If any crucial part is missing, we can't convert.
            # We'll keep the entry as is (with nulls) or you could choose to skip it.
            # For now, let's keep it, but mark the new fields as None too.
            new_entry["H_canvas_to_stack"] = None
            new_entry["H_stack_to_canvas"] = None
            new_entry["corners_canvas_overlap"] = None
            # corners_stack_overlap is already in new_entry from the copy
            print(f"Warning: Missing transformation data for '{image_key}'. Canvas transforms will be null.")
            skipped_count +=1
        else:
            try:
                H_t_s = np.array(h_t_s_raw, dtype=np.float64)
                H_s_t = np.array(h_s_t_raw, dtype=np.float64)
                corners_thumb = np.array(corners_thumb_raw, dtype=np.float64)

                # 1. Calculate H_canvas_to_stack
                H_c_s = H_t_s @ M_c_t
                new_entry["H_canvas_to_stack"] = H_c_s.tolist()

                # 2. Calculate H_stack_to_canvas
                H_s_c = M_t_c @ H_s_t
                new_entry["H_stack_to_canvas"] = H_s_c.tolist()

                # 3. Adjust corners_thumbnail_overlap to corners_canvas_overlap
                # corners_thumb are (x,y) points. Multiply by s.
                corners_canvas = corners_thumb * s
                new_entry["corners_canvas_overlap"] = corners_canvas.tolist()
                
                processed_count += 1

            except Exception as e:
                print(f"Error processing entry '{image_key}': {e}. Setting canvas transforms to null.")
                new_entry["H_canvas_to_stack"] = None
                new_entry["H_stack_to_canvas"] = None
                new_entry["corners_canvas_overlap"] = None
                skipped_count +=1


        # Rename/remove old keys for clarity in the new file
        new_entry.pop("H_thumbnail_to_stack", None)
        new_entry.pop("H_stack_to_thumbnail", None)
        new_entry.pop("corners_thumbnail_overlap", None)

        # Update paths if they contain "thumbnail" - this is optional and heuristic
        if "thumbnail_path" in new_entry and new_entry["thumbnail_path"]:
            new_entry["canvas_path"] = new_entry["thumbnail_path"].replace("thumbnail", "canvas", 1) # Heuristic
            new_entry.pop("thumbnail_path")
        else:
            new_entry["canvas_path"] = None # Or some placeholder


        canvas_data[image_key] = new_entry

    print(f"\nConversion summary:")
    print(f"  Total entries processed: {processed_count}")
    print(f"  Entries with missing data or errors (transforms set to null): {skipped_count}")


    try:
        with open(canvas_json_path, 'w') as f:
            json.dump(canvas_data, f, indent=4)
        print(f"\nSuccessfully converted and saved data to '{canvas_json_path}'")
    except IOError:
        print(f"Error: Could not write the canvas JSON to '{canvas_json_path}'. Check permissions or path.")
    except TypeError as e:
        print(f"Error serializing data to JSON: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts thumbnail-to-stack transformation JSON to canvas-to-stack JSON."
    )
    parser.add_argument(
        "thumbnail_json",
        help="Path to the input JSON file with thumbnail-to-stack transformations (e.g., 222_t_to_s.json)."
    )
    parser.add_argument(
        "canvas_json",
        help="Path for the output JSON file with canvas-to-stack transformations (e.g., 222_c_to_s.json)."
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=64.0,
        help="The downsampling factor used to create thumbnails from canvas images (default: 64.0)."
    )

    args = parser.parse_args()

    convert_thumbnail_to_canvas_json(
        args.thumbnail_json,
        args.canvas_json,
        downsample_factor=args.factor
    )

    # To test this script, you'd first need a "222_t_to_s.json"-like file.
    # Example: create a dummy `dummy_t_s.json`:
    """
    {
        "imageA.jpg": {
            "thumbnail_path": "data/thumbnails/imageA.jpg",
            "stack_path": "data/stacks/imageA.jpg",
            "status": "success",
            "error_message": null,
            "H_thumbnail_to_stack": [
                [1.0, 0.0, 10.0],
                [0.0, 1.0, 5.0],
                [0.0, 0.0, 1.0]
            ],
            "H_stack_to_thumbnail": [
                [1.0, 0.0, -10.0],
                [0.0, 1.0, -5.0],
                [0.0, 0.0, 1.0]
            ],
            "corners_thumbnail_overlap": [
                [1.0, 1.0], [10.0, 1.0], [10.0, 10.0], [1.0, 10.0]
            ],
            "corners_stack_overlap": [
                [11.0, 6.0], [20.0, 6.0], [20.0, 15.0], [11.0, 15.0]
            ]
        },
        "imageB.jpg": {
            "thumbnail_path": "data/thumbnails/imageB.jpg",
            "stack_path": "data/stacks/imageB.jpg",
            "status": "error",
            "error_message": "Not enough matches",
            "H_thumbnail_to_stack": null,
            "H_stack_to_thumbnail": null,
            "corners_thumbnail_overlap": null,
            "corners_stack_overlap": null
        }
    }
    """
    # Then run:
    # python your_script_name.py dummy_t_s.json dummy_c_s.json --factor 64
    #
    # Check dummy_c_s.json. For imageA.jpg:
    # H_canvas_to_stack[0][2] should be 10.0 / 64.0
    # H_canvas_to_stack[1][2] should be 5.0 / 64.0
    # H_stack_to_canvas[0][2] should be -10.0 * 64.0
    # H_stack_to_canvas[1][2] should be -5.0 * 64.0
    # corners_canvas_overlap should be corners_thumbnail_overlap * 64.0
