#!/usr/bin/env python3
"""
Script to extract rotation value from brain viewer API and update 222_original_to_bfiw.json
This mimics the logic in views.py to get jp2_meta["rotation"] for biosample_id 222
"""

import json
import requests
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_brain_viewer_details(biosample_id):
    """Fetch brain viewer details from API - same as in views.py"""
    bfidic = {
        "222": 100,
        "244": 116,
        "142": 65,
    }
    bfi_value = bfidic.get(str(biosample_id))
    if not bfi_value:
        logger.error(f"BFI value not found for biosample_id: {biosample_id}")
        return None
    
    url = f"http://dev2adi.humanbrain.in:8000/GW/getBrainViewerDetails/IIT/V1/SS-{bfi_value}:-1:-1"
    logger.info(f"Sending request to: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        logger.info(f"Successfully fetched data from {url}")
        return response.json()
    except requests.exceptions.Timeout:
        logger.error(f"Request timeout while fetching data from {url}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from {url}: {e}")
        return None


def get_jp2_rotation(biosample_id, section_number):
    """Get rotation value for a specific section - same logic as get_jp2_metadata in views.py"""
    logger.info(f"Fetching rotation for biosample: {biosample_id}, section: {section_number}")
    
    # Fetch JSON data
    json_data = fetch_brain_viewer_details(biosample_id)
    if not json_data:
        logger.error(f"Failed to fetch JSON data for biosample {biosample_id}")
        return None
    
    # Extract NISSL data from the JSON response
    jp2_data = json_data.get('thumbNail', {}).get('NISSL', [])
    if not jp2_data:
        logger.error(f"No 'NISSL' data found in the JSON for biosample {biosample_id}")
        return None
    
    # Try exact match first
    logger.info(f"Searching for exact match for section {section_number} in NISSL data.")
    exact_match = next(
        (item for item in jp2_data if str(item.get('position_index')) == str(section_number)),
        None
    )
    
    match_item = exact_match
    
    # If exact match not found, find the nearest match by numeric distance
    if not match_item:
        logger.info(f"No exact match found for section {section_number}. Searching for the nearest match.")
        try:
            section_number_int = int(section_number)
        except ValueError:
            logger.error(f"Invalid section_number: {section_number}. It must be an integer.")
            return None
        
        valid_items = [item for item in jp2_data if isinstance(item.get('position_index'), int)]
        if not valid_items:
            logger.error(f"No valid position_index found in NISSL data for biosample {biosample_id}.")
            return None
        
        nearest_item = min(
            valid_items,
            key=lambda item: abs(item['position_index'] - section_number_int)
        )
        match_item = nearest_item
        logger.info(f"Nearest match found for section {section_number}: position_index {match_item['position_index']}.")
    
    # If no match is found, log the failure and return None
    if not match_item:
        logger.error(f"Could not find any matching data for section {section_number} in NISSL.")
        return None
    
    # Extract rotation value
    rotation = match_item.get('rigidrotation', 0)
    logger.info(f"Found rotation value: {rotation} for section {section_number}")
    
    return rotation


def update_rotation_values(json_file_path, biosample_id):
    """Update all rotation values in the JSON file with values from API"""
    try:
        # Read the JSON file
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Get all section numbers from the JSON file
        section_numbers = []
        for key in data.keys():
            if key.endswith('.jpg'):
                section_num = key.replace('.jpg', '')
                section_numbers.append(section_num)
        
        logger.info(f"Found {len(section_numbers)} sections to update")
        
        # Update each entry with the rotation from API
        update_count = 0
        for section_num in section_numbers:
            section_key = f"{section_num}.jpg"
            
            # Get rotation value from API
            rotation = get_jp2_rotation(biosample_id, section_num)
            
            if rotation is not None and section_key in data:
                if isinstance(data[section_key], dict) and 'interpolated_bfw_css_rotation_deg' in data[section_key]:
                    old_value = data[section_key]['interpolated_bfw_css_rotation_deg']
                    data[section_key]['interpolated_bfw_css_rotation_deg'] = rotation
                    update_count += 1
                    logger.info(f"Updated {section_key}: {old_value} -> {rotation}")
        
        # Write the updated data back to the file
        with open(json_file_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        logger.info(f"\nSuccessfully updated {update_count} entries in {json_file_path}")
        
    except FileNotFoundError:
        logger.error(f"Error: File not found - {json_file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Error: Invalid JSON in file - {json_file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


def update_with_single_rotation(json_file_path, rotation_value):
    """Update all rotation values with a single value"""
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
                value['interpolated_bfw_css_rotation_deg'] = rotation_value
                update_count += 1
                logger.info(f"Updated {key}: {old_value} -> {rotation_value}")
        
        # Write the updated data back to the file
        with open(json_file_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        logger.info(f"\nSuccessfully updated {update_count} entries with rotation value: {rotation_value}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Configuration
    biosample_id = "244"
    json_file_path = "/home/projects/bfi_viewer/app/backend/brainviewer/data/244_original_to_bfiw.json"
    
    print(f"Rotation Update Script for biosample_id {biosample_id}")
    print("-" * 60)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # If a rotation value is provided, use it for all entries
        try:
            rotation_value = float(sys.argv[1])
            print(f"Using provided rotation value: {rotation_value} for all entries")
            update_with_single_rotation(json_file_path, rotation_value)
        except ValueError:
            print("Error: Rotation value must be a number")
            sys.exit(1)
    else:
        # Otherwise, fetch rotation values from API for each section
        print("Fetching rotation values from API for each section...")
        print(f"Target file: {json_file_path}")
        print("-" * 60)
        
        # Update the JSON file with API values
        update_rotation_values(json_file_path, biosample_id)