import json
import argparse

def clean_json_of_incomplete_transforms(json_filepath, output_filepath=None, create_backup=True):
    """
    Reads a JSON file, removes entries that do not have complete transformation data
    (H_thumbnail_to_stack, H_stack_to_thumbnail, corners_thumbnail_overlap, corners_stack_overlap),
    and saves the cleaned data.

    Args:
        json_filepath (str): Path to the input JSON file.
        output_filepath (str, optional): Path to save the cleaned JSON.
                                         If None, overwrites the input file. Defaults to None.
        create_backup (bool): If True and overwriting, creates a backup of the original file.
    """
    try:
        with open(json_filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at {json_filepath}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_filepath}. File might be corrupted or not valid JSON.")
        return

    if not isinstance(data, dict):
        print(f"Error: Expected a dictionary at the top level of the JSON file {json_filepath}, but found {type(data)}.")
        return

    keys_to_remove = []
    required_transform_keys = [
        "H_thumbnail_to_stack",
        "H_stack_to_thumbnail",
        "corners_thumbnail_overlap",
        "corners_stack_overlap"
    ]

    print(f"Scanning '{json_filepath}' for entries with missing transformation data...")
    for key, value_dict in data.items():
        if isinstance(value_dict, dict):
            # Check if any of the required transformation keys are missing or have a None/null value
            is_incomplete = False
            for transform_key in required_transform_keys:
                if transform_key not in value_dict or value_dict[transform_key] is None:
                    is_incomplete = True
                    break # No need to check further keys for this entry

            if is_incomplete:
                keys_to_remove.append(key)
                # Optional: print which entry is being marked for removal
                # print(f"  - Marking '{key}' for removal due to incomplete transformation data.")
        else:
            print(f"Warning: Entry for key '{key}' in {json_filepath} is not a dictionary. Skipping this entry for cleaning.")

    if not keys_to_remove:
        print(f"No entries with incomplete transformation data found in '{json_filepath}'. File remains unchanged.")
        return

    print(f"\nFound {len(keys_to_remove)} entries to remove:")
    if len(keys_to_remove) <= 20:
        for k_to_remove in keys_to_remove:
            print(f"  - {k_to_remove}")
    else:
        print(f"  (List of {len(keys_to_remove)} keys is too long to display fully)")


    for key in keys_to_remove:
        del data[key]

    if output_filepath is None:
        output_filepath = json_filepath # Overwrite the original
        if create_backup:
            backup_filepath = json_filepath + ".bak"
            try:
                import shutil
                shutil.copy2(json_filepath, backup_filepath)
                print(f"Backup of original file created at: {backup_filepath}")
            except Exception as e:
                print(f"Warning: Could not create backup file. {e}")

    try:
        with open(output_filepath, 'w') as f:
            json.dump(data, f, indent=4)
        if output_filepath == json_filepath:
            print(f"Successfully removed {len(keys_to_remove)} entries and updated '{output_filepath}'.")
        else:
            print(f"Successfully removed {len(keys_to_remove)} entries. Cleaned data saved to '{output_filepath}'.")

    except IOError:
        print(f"Error: Could not write updated JSON to '{output_filepath}'. Check permissions or path.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Cleans a JSON file by removing entries that lack complete transformation data. "
                    "An entry is considered incomplete if 'H_thumbnail_to_stack', "
                    "'H_stack_to_thumbnail', 'corners_thumbnail_overlap', or "
                    "'corners_stack_overlap' is missing or null."
    )
    parser.add_argument("json_file", help="Path to the JSON file to clean (e.g., 222_t_to_s.json).")
    parser.add_argument(
        "--output_file",
        help="Optional: Path to save the cleaned JSON. If not provided, the input file will be overwritten."
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Do not create a backup file when overwriting the input file."
    )

    args = parser.parse_args()

    should_create_backup = not args.no_backup if args.output_file is None else False

    clean_json_of_incomplete_transforms(
        args.json_file,
        output_filepath=args.output_file,
        create_backup=should_create_backup
    )

    # Example of creating a dummy JSON for testing this script:
    # To test, you could manually create a file named 'test_cleaner.json' with content like:
    """
    {
        "image1.jpg": {
            "thumbnail_path": "thumb/image1.jpg",
            "stack_path": "stack/image1.jpg",
            "status": "success",
            "error_message": null,
            "H_thumbnail_to_stack": [[1,0,0],[0,1,0],[0,0,1]],
            "H_stack_to_thumbnail": [[1,0,0],[0,1,0],[0,0,1]],
            "corners_thumbnail_overlap": [[0,0],[1,0],[1,1],[0,1]],
            "corners_stack_overlap": [[0,0],[1,0],[1,1],[0,1]]
        },
        "image2.jpg": {
            "thumbnail_path": "thumb/image2.jpg",
            "stack_path": "stack/image2.jpg",
            "status": "error",
            "error_message": "Not enough good matches",
            "H_thumbnail_to_stack": null,
            "H_stack_to_thumbnail": null,
            "corners_thumbnail_overlap": null,
            "corners_stack_overlap": null
        },
        "image3.jpg": {
            "thumbnail_path": "thumb/image3.jpg",
            "stack_path": "stack/image3.jpg",
            "status": "success",
            "error_message": null,
            "H_thumbnail_to_stack": [[2,0,0],[0,2,0],[0,0,1]],
            "H_stack_to_thumbnail": null,
            "corners_thumbnail_overlap": [[0,0],[1,0],[1,1],[0,1]],
            "corners_stack_overlap": [[0,0],[1,0],[1,1],[0,1]]
        },
        "image4.jpg": {
            "thumbnail_path": "thumb/image4.jpg",
            "stack_path": "stack/image4.jpg",
            "status": "error",
            "error_message": "Corresponding stack image not found.",
            "H_thumbnail_to_stack": null,
            "H_stack_to_thumbnail": null,
            "corners_thumbnail_overlap": null,
            "corners_stack_overlap": null
        }
    }
    """
    # Then run: python your_script_name.py test_cleaner.json
    # Or: python your_script_name.py test_cleaner.json --output_file cleaned_test.json
    #
    # Expected output (if overwriting test_cleaner.json): image2.jpg, image3.jpg, image4.jpg removed. Only image1.jpg remains.
