#!/usr/bin/env python3
"""
Script to update interpolated_bfw_css_rotation_deg values in 222_original_to_bfiw.json
with the rotation value from jp2_meta['rotation'] for biosample_id 222
"""

import json
import sys


def update_rotation_values(json_file_path, new_rotation_value, biosample_id):
    """
    Update all interpolated_bfw_css_rotation_deg values in the JSON file
    
    Args:
        json_file_path: Path to the JSON file to update
        new_rotation_value: The new rotation value to set
        biosample_id: The biosample ID (for logging purposes)
    """
    try:
        # Read the JSON file
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Count updates for verification
        update_count = 0
        
        # Update each entry
        for key, value in data.items():
            if isinstance(value, dict) and 'interpolated_bfw_css_rotation_deg' in value:
                old_value = value['interpolated_bfw_css_rotation_deg']
                value['interpolated_bfw_css_rotation_deg'] = new_rotation_value
                update_count += 1
                print(f"Updated {key}: {old_value} -> {new_rotation_value}")
        
        # Write the updated data back to the file
        with open(json_file_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"\nSuccessfully updated {update_count} entries in {json_file_path}")
        print(f"All interpolated_bfw_css_rotation_deg values set to: {new_rotation_value}")
        
    except FileNotFoundError:
        print(f"Error: File not found - {json_file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file - {json_file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


def get_rotation_from_views():
    """
    This function would normally get the rotation value from views.py
    For this script, we'll accept it as a command line argument
    """
    # In a real implementation, you would import the views module
    # and extract jp2_meta["rotation"] for biosample_id 222
    # For now, this is a placeholder
    pass


if __name__ == "__main__":
    # Configuration
    biosample_id = "222"
    json_file_path = "/home/projects/bfi_viewer/app/backend/brainviewer/data/222_original_to_bfiw.json"
    
    # Check if rotation value is provided as command line argument
    if len(sys.argv) < 2:
        print("Usage: python update_rotation.py <rotation_value>")
        print("Example: python update_rotation.py 90")
        print("\nNote: In production, this script would extract jp2_meta['rotation'] directly from views.py")
        sys.exit(1)
    
    try:
        new_rotation = float(sys.argv[1])
    except ValueError:
        print("Error: Rotation value must be a number")
        sys.exit(1)
    
    print(f"Updating rotation values for biosample_id {biosample_id}")
    print(f"New rotation value: {new_rotation}")
    print(f"Target file: {json_file_path}")
    print("-" * 50)
    
    # Update the JSON file
    update_rotation_values(json_file_path, new_rotation, biosample_id)