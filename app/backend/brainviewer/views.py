from django.shortcuts import render, redirect
from django.http import HttpResponse, Http404
from django.conf import settings

import requests
import json
import os
import re
import logging

logger = logging.getLogger(__name__)

def fetch_brain_viewer_details(biosample_id):
    bfidic = {
        "222": 100,
        "244": 116,
        "142": 65,
    }
    bfi_value = bfidic.get(str(biosample_id))
    if not bfi_value:
        logger.error(f"BFI value not found for biosample_id: {biosample_id}")
        return None
    urls = [
        f"http://dev2adi.humanbrain.in:8000/GW/getBrainViewerDetails/IIT/V1/SS-{bfi_value}:-1:-1",
        f"http://dev2mani.humanbrain.in:8000/GW/getBrainViewerDetails/IIT/V1/SS-{bfi_value}:-1:-1",
        f"http://dev2kamal.humanbrain.in:8000/GW/getBrainViewerDetails/IIT/V1/SS-{bfi_value}:-1:-1"
        ]
    for url in urls:
        logger.info(f"Trying URL: {url}")
        logger.info(f"Sending request to: {url}")
        try:
            response = requests.get(url, timeout=5)  # Add 5 second timeout
            response.raise_for_status()  # Raise an error if the response code isn't 200
            logger.info(f"Successfully fetched data from {url}")
            return response.json()  # Or response.text based on the expected data format
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout while fetching data from {url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from {url}: {e}")
    logger.error("All URLs failed.")
    return None
def get_metadata_by_type(biosample_id, section_number, data_type, type_name):
    logger.info(
        f"Fetching {type_name} metadata for biosample: {biosample_id}, section: {section_number}")

    # Fetch JSON data
    json_data = fetch_brain_viewer_details(biosample_id)
    if not json_data:
        logger.error(
            f"Failed to fetch JSON data for biosample {biosample_id}, section {section_number}")
        return None

    # Extract data from the JSON response
    metadata_data = json_data.get('thumbNail', {}).get(data_type, [])
    if not metadata_data:
        logger.error(
            f"No '{data_type}' data found in the JSON for biosample {biosample_id}, section {section_number}")
        return None

    # Try exact match first
    logger.info(
        f"Searching for exact match for section {section_number} in {data_type} data.")
    exact_match = next(
        (item for item in metadata_data if str(
            item.get('position_index')) == str(section_number)),
        None
    )

    match_item = exact_match

    # If exact match not found, find the nearest match by numeric distance
    if not match_item:
        logger.info(
            f"No exact match found for section {section_number}. Searching for the nearest match.")
        try:
            section_number_int = int(section_number)
        except ValueError:
            logger.error(
                f"Invalid section_number: {section_number}. It must be an integer.")
            return None

        valid_items = [item for item in metadata_data if isinstance(
            item.get('position_index'), int)]
        if not valid_items:
            logger.error(
                f"No valid position_index found in {data_type} data for biosample {biosample_id}.")
            return None

        nearest_item = min(
            valid_items,
            key=lambda item: abs(item['position_index'] - section_number_int)
        )
        match_item = nearest_item
        logger.info(
            f"Nearest match found for section {section_number}: position_index {match_item['position_index']}.")

    # If no match is found, log the failure and return None
    if not match_item:
        logger.error(
            f"Could not find any matching data for section {section_number} in {data_type}.")
        return None

    # Extract only required fields and log the values
    filtered = {
        'height': match_item['height'],
        'width': match_item['width'],
        'rotation': match_item.get('rigidrotation', 0),
        'jp2_path_fragment': match_item['jp2Path'],
        'position_index': match_item['position_index']
    }

    logger.info(
        f"Successfully retrieved {type_name} metadata for biosample {biosample_id}, section {section_number}: {filtered}")

    return filtered

def get_jp2_metadata(biosample_id, section_number):
    return get_metadata_by_type(biosample_id, section_number, 'NISSL', 'JP2')

def get_bfiw_metadata(biosample_id, section_number):
    return get_metadata_by_type(biosample_id, section_number, 'BFI White', 'BFI White')


def get_haematoxylin_and_eosin_metadata(biosample_id, section_number):
    return get_metadata_by_type(biosample_id, section_number, 'Haematoxylin and Eosin', 'Haematoxylin and Eosin')


def get_mri_metadata(biosample_id, section_number):
    return get_metadata_by_type(biosample_id, section_number, 'MRI', 'MRI')


def get_transformation_data(biosample_id, section_number_str):
    json_filename = os.path.join(
        settings.BASE_DIR, 'brainviewer', 'data', f"{biosample_id}_original_to_bfiw.json")
    logger.info(
        f"Fetching transformation data from: {json_filename} for section key: {section_number_str}.jpg")
    try:
        json_filepath = json_filename

        if not os.path.exists(json_filepath):
            logger.error(
                f"JSON file not found at path: {os.path.abspath(json_filepath)}")
            return None

        with open(json_filepath, 'r') as f:
            all_transform_data = json.load(f)

        section_key = f"{section_number_str}.jpg"
        if section_key in all_transform_data:
            section_data = all_transform_data[section_key]
            logger.debug(
                f"Raw section data from JSON for {section_key}: {section_data}")
            if section_data.get("status") == "success":
                h_matrix = section_data.get("H_canvas_to_bfw")
                bfi_dims = section_data.get("bfw_dims_original")
                bfi_rotation = section_data.get("bfw_rotation", 0)

                if not h_matrix:
                    logger.warning(
                        f"H_canvas_to_bfw (H_jp2_to_bfi) is missing in JSON for {section_key}")
                if not bfi_dims or not isinstance(bfi_dims, list) or len(bfi_dims) < 2:
                    logger.warning(
                        f"bfw_dims_original (bfi_natural_dims) is missing or invalid (not a list of 2+) in JSON for {section_key}")

                transform_details = {
                    "H_jp2_to_bfi": h_matrix,
                    "bfi_natural_dims": bfi_dims,
                    "bfi_rotation": section_data.get("interpolated_bfw_css_rotation_deg", 0),
                }
                logger.info(
                    f"Processed transformation data for {section_key}: {transform_details}")
                return transform_details
            else:
                logger.warning(
                    f"Transformation status not 'success' ({section_data.get('status')}) for {section_key} in {json_filename}")
        else:
            logger.warning(
                f"Section key {section_key} not found in {json_filename}")
    except FileNotFoundError:
        logger.error(f"JSON file not found: {json_filename}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {json_filename}")
    except Exception as e:
        logger.error(
            f"Error reading transformation data from {json_filename}: {e}")
    return None


def get_available_biosample_ids():
    return ["222", "244", "142"]

def generate_error_html(error_type, identifier, title):
    """Generate standardized error HTML for missing biosample or slice IDs"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                text-align: center;
                padding: 40px;
            }}
            .error-box {{
                background-color: #fff;
                border: 1px solid #ccc;
                padding: 30px;
                display: inline-block;
                border-radius: 10px;
                box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
            }}
            a {{
                display: block;
                margin-top: 20px;
                text-decoration: none;
                color: #007BFF;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="error-box">
            <h2>🚫 {error_type} Not Found</h2>
            <p>The ID <strong>{identifier}</strong> does not exist in our records.</p>
        </div>
    </body>
    </html>
    """

def validate_inputs_and_get_metadata(biosample_id, slice_number_str, get_all_metadata=False):
    """Validate inputs and fetch metadata. Returns tuple of (validation_response, metadata_dict)
    If validation_response is not None, it's an HttpResponse that should be returned immediately."""
    
    try:
        slice_number_int = int(slice_number_str)
    except ValueError:
        logger.error(f"'slice_number' in URL must be an integer, got: {slice_number_str}")
        raise Http404(f"'slice_number' ({slice_number_str}) must be an integer.")
    
    if biosample_id not in get_available_biosample_ids():
        logger.warning(f"Invalid biosample_id: {biosample_id}")
        html_content = generate_error_html("Biosample ID", biosample_id, "No Brain ID Found")
        return HttpResponse(html_content), None
        
    if slice_number_str not in get_available_slice_numbers_for_biosample(biosample_id):
        logger.warning(f"Invalid slice_id: {slice_number_str}")
        html_content = generate_error_html("Slice ID", slice_number_str, "No Slice ID Found")
        return HttpResponse(html_content), None
    
    # Fetch metadata
    metadata = {}
    jp2_meta = get_jp2_metadata(biosample_id, slice_number_int)
    transform_data = get_transformation_data(biosample_id, slice_number_str)
    
    if get_all_metadata:
        bfi_meta = get_bfiw_metadata(biosample_id, slice_number_int)
        hae_meta = get_haematoxylin_and_eosin_metadata(biosample_id, slice_number_int)
        mri_meta = get_mri_metadata(biosample_id, slice_number_int)
        metadata.update({'bfi_meta': bfi_meta, 'hae_meta': hae_meta, 'mri_meta': mri_meta})
    
    # Validate required data
    if not jp2_meta:
        logger.error(f"Failed to get JP2 metadata for biosample {biosample_id}, slice {slice_number_str}.")
        raise Http404(f"JP2 Data not found for biosample {biosample_id}, slice {slice_number_str}.")
        
    if not transform_data:
        logger.error(f"Failed to get transformation data for biosample {biosample_id}, slice {slice_number_str}.")
        raise Http404(f"Transformation Data not found for biosample {biosample_id}, slice {slice_number_str}.")
    
    if get_all_metadata:
        if not metadata['hae_meta']:
            logger.error(f"Failed to get Haematoxylin and Eosin metadata for biosample {biosample_id}, slice {slice_number_str}.")
            raise Http404(f"Haematoxylin and Eosin Data not found for biosample {biosample_id}, slice {slice_number_str}.")
        if not metadata['mri_meta']:
            logger.error(f"Failed to get MRI metadata for biosample {biosample_id}, slice {slice_number_str}.")
            raise Http404(f"MRI Data not found for biosample {biosample_id}, slice {slice_number_str}.")
    
    metadata.update({
        'jp2_meta': jp2_meta,
        'transform_data': transform_data,
        'slice_number_int': slice_number_int
    })
    
    return None, metadata


def get_available_slice_numbers_for_biosample(biosample_id):
    json_filename = os.path.join(
        settings.BASE_DIR, 'brainviewer', 'data', f"{biosample_id}_original_to_bfiw.json")
    slice_numbers = []
    try:
        json_filepath = json_filename
        if not os.path.exists(json_filepath):
            logger.warning(
                f"Slice number source file not found: {json_filepath} for biosample_id {biosample_id}")
            return []

        with open(json_filepath, 'r') as f:
            all_transform_data = json.load(f)

        for key in all_transform_data.keys():
            match = re.match(r"(\d+)\.jpg$", key)
            if match:
                slice_num_str = match.group(1)
                slice_numbers.append(slice_num_str)
            else:
                logger.debug(
                    f"Key '{key}' in {json_filename} does not match expected slice format (e.g., '123.jpg').")
        slice_numbers.sort(key=int)
        logger.info(
            f"Found available slice numbers for biosample {biosample_id}: {slice_numbers}")
    except FileNotFoundError:
        logger.error(
            f"JSON file for slice numbers not found: {json_filename}")
    except json.JSONDecodeError:
        logger.error(
            f"Error decoding JSON for slice numbers from {json_filename}")
    except Exception as e:
        logger.error(
            f"Error getting available slice numbers for {biosample_id}: {e}")
    return slice_numbers


def home_view(request):
    available_ids = get_available_biosample_ids()
    default_biosample_id = available_ids[0] if available_ids else None
    if default_biosample_id:
        default_slices = get_available_slice_numbers_for_biosample(default_biosample_id)
        default_slice_number = default_slices[0] if default_slices else None
        if default_slice_number:
            return redirect('brainviewer:viewer', biosample_id=default_biosample_id, slice_number_str=default_slice_number)

    logger.warning("Could not determine a default slice to redirect to. Serving simple message.")
    return HttpResponse("Welcome to the Brain Slice Viewer. No default slice configured or available.")


def viewer_view(request, biosample_id, slice_number_str, port_no=10803):
    logger.info(
        f"Received request for /viewer/{biosample_id}/{slice_number_str}")

    try:
        slice_number_int = int(slice_number_str)
    except ValueError:
        logger.error(
            f"'slice_number' in URL must be an integer, got: {slice_number_str}")
        raise Http404(
            f"'slice_number' ({slice_number_str}) must be an integer.")

    jp2_meta = get_jp2_metadata(biosample_id, slice_number_int)
    transform_data = get_transformation_data(biosample_id, slice_number_str)
    if not transform_data:
        raise Http404(
            f"Transformation Data not found for biosample {biosample_id}, slice {slice_number_str}.")

    if not jp2_meta:
        logger.error(
            f"Failed to get JP2 metadata for biosample {biosample_id}, slice {slice_number_str}.")
        raise Http404(
            f"JP2 Data not found for biosample {biosample_id}, slice {slice_number_str}.")
    if not transform_data:
        logger.error(
            f"Failed to get transformation data for biosample {biosample_id}, slice {slice_number_str}.")
        raise Http404(
            f"Transformation Data not found for biosample {biosample_id}, slice {slice_number_str}.")

    h_jp2_to_bfi = transform_data.get("H_jp2_to_bfi")
    bfi_natural_dims = transform_data.get("bfi_natural_dims")

    if not h_jp2_to_bfi:
        logger.error(
            f"H_jp2_to_bfi matrix is missing from transformation data for biosample {biosample_id}, slice {slice_number_str}.")
        raise Http404(
            f"Critical transformation matrix H_jp2_to_bfi is missing for slice {slice_number_str}.")
    if not bfi_natural_dims or not isinstance(bfi_natural_dims, list) or len(bfi_natural_dims) < 2:
        logger.error(
            f"BFI natural dimensions are missing or invalid from transformation data for biosample {biosample_id}, slice {slice_number_str}.")
        raise Http404(
            f"Critical BFI natural dimensions are missing or invalid for slice {slice_number_str}.")

    jp2_base_url = "https://apollo2.humanbrain.in/iipsrv/fcgi-bin/iipsrv.fcgi?FIF=/"
    jp2_path = jp2_meta['jp2_path_fragment']
    if jp2_path.startswith('/'):
        jp2_path = jp2_path[1:]

    jp2_suffix = "&WID=1024&GAM=1.4&MINMAX=1:0,255&MINMAX=2:0,255&MINMAX=3:0,255&JTL={z},{tileIndex}"
    full_jp2_url = f"{jp2_base_url}{jp2_path}{jp2_suffix}"

    # bfi_image_url = f"/static/images_data/{biosample_id}/bfi-{slice_number_str}.png"
    # bfi_image_url = f"images_data/{biosample_id}/bfi-{slice_number_str}.png"   # this is not working
    # bfi_image_url = f"http://dgx3.humanbrain.in:10803/images/222/bfi-763.png"
    # bfi_image_url = f"http://dgx3.humanbrain.in:{port_no}/images/{biosample_id}/bfi-{slice_number_str}.png"
    bfi_image_url = f"https://apollo2.humanbrain.in/bfiViewerServer/images/{biosample_id}/bfi-{slice_number_str}.png"
    logger.info(f"Generated BFI Image URL: {bfi_image_url}")

    available_b_ids = get_available_biosample_ids()
    available_s_nums = get_available_slice_numbers_for_biosample(biosample_id)

    if slice_number_str not in available_s_nums and available_s_nums:
        logger.warning(
            f"Current slice {slice_number_str} for biosample {biosample_id} is not in the dynamically generated available_slice_numbers list: {available_s_nums}. The dropdown might not pre-select it correctly if it's missing from the JSON keys.")

    bfi_w = bfi_natural_dims[0] if bfi_natural_dims and len(
        bfi_natural_dims) > 0 else 0
    bfi_h = bfi_natural_dims[1] if bfi_natural_dims and len(
        bfi_natural_dims) > 1 else 0
    print("BFI image url:", json.dumps(bfi_image_url))
    template_data = {
        "jp2_map_url": json.dumps(full_jp2_url),
        "jp2_full_size": json.dumps([jp2_meta["width"], jp2_meta["height"]]),
        "jp2_initial_view_rotation_deg": json.dumps(jp2_meta["rotation"]),
        "bfi_image_url": json.dumps(bfi_image_url),
        "bfi_css_rotation_deg": json.dumps(jp2_meta["rotation"]),
        "h_jp2_to_bfi": json.dumps(h_jp2_to_bfi),
        "bfi_natural_width": json.dumps(bfi_w),
        "bfi_natural_height": json.dumps(bfi_h),
        "test_points_jp2": [],
        "title": f"Slice Viewer - BSID {biosample_id}, Slice {slice_number_str}",
        "current_biosample_id": json.dumps(biosample_id),
        "current_slice_number": json.dumps(slice_number_str),
        "available_biosample_ids": json.dumps(available_b_ids),
        "available_slice_numbers": json.dumps(available_s_nums),
        # CORRECTED PARAMETER NAME HERE
        "viewer_url_template": json.dumps(f"/viewer/__BID__/__SID__/")
    }
    logger.debug(
        f"Data being passed to template for /viewer/{biosample_id}/{slice_number_str}: {template_data}")

    return render(request, 'brainviewer/viewer.html', template_data)


def viewer_view_split(request, biosample_id, slice_number_str, port_no=8000):
    return _viewer_view_split_common(request, biosample_id, slice_number_str, port_no, debug_mode=False)
        
def _viewer_view_split_common(request, biosample_id, slice_number_str, port_no=8000, debug_mode=False):
    logger.info(f"Received request for /viewer/{biosample_id}/{slice_number_str}")
    
    # Validate inputs and get metadata
    validation_response, metadata = validate_inputs_and_get_metadata(biosample_id, slice_number_str, get_all_metadata=True)
    if validation_response:
        return validation_response
    
    jp2_meta = metadata['jp2_meta']
    hae_meta = metadata['hae_meta']
    mri_meta = metadata['mri_meta']
    transform_data = metadata['transform_data']
    
    h_jp2_to_bfi = transform_data.get("H_jp2_to_bfi")
    bfi_natural_dims = transform_data.get("bfi_natural_dims")
    bfi_rotation_raw = transform_data.get("bfi_rotation", 0)
    bfi_rotation = float(bfi_rotation_raw) if bfi_rotation_raw is not None else 0.0

    if not h_jp2_to_bfi:
        logger.error(f"H_jp2_to_bfi matrix is missing from transformation data for biosample {biosample_id}, slice {slice_number_str}.")
        raise Http404(f"Critical transformation matrix H_jp2_to_bfi is missing for slice {slice_number_str}.")
    if not bfi_natural_dims or not isinstance(bfi_natural_dims, list) or len(bfi_natural_dims) < 2:
        logger.error(f"BFI natural dimensions are missing or invalid from transformation data for biosample {biosample_id}, slice {slice_number_str}.")
        raise Http404(f"Critical BFI natural dimensions are missing or invalid for slice {slice_number_str}.")

    # Build URLs
    base_url = "https://apollo2.humanbrain.in/iipsrv/fcgi-bin/iipsrv.fcgi?FIF=/"
    suffix = "&WID=1024&GAM=1.4&MINMAX=1:0,255&MINMAX=2:0,255&MINMAX=3:0,255&JTL={z},{tileIndex}"
    
    def clean_path(path):
        return path[1:] if path.startswith('/') else path
    
    jp2_path = clean_path(jp2_meta['jp2_path_fragment'])
    hae_path = clean_path(hae_meta['jp2_path_fragment'])
    mri_path = clean_path(mri_meta['jp2_path_fragment'])
    
    full_jp2_url = f"{base_url}{jp2_path}{suffix}"
    full_hae_url = f"{base_url}{hae_path}{suffix}"
    full_mri_url = f"{base_url}{mri_path}{suffix}"
    
    bfi_image_url = f"http://dev2imran.humanbrain.in:{port_no}/images/{biosample_id}/bfi-{slice_number_str}.png"
    logger.info(f"Generated BFI Image URL: {bfi_image_url}")

    available_b_ids = get_available_biosample_ids()
    available_s_nums = get_available_slice_numbers_for_biosample(biosample_id)

    if slice_number_str not in available_s_nums and available_s_nums:
        logger.warning(f"Current slice {slice_number_str} for biosample {biosample_id} is not in the dynamically generated available_slice_numbers list: {available_s_nums}. The dropdown might not pre-select it correctly if it's missing from the JSON keys.")

    bfi_w = bfi_natural_dims[0] if bfi_natural_dims and len(bfi_natural_dims) > 0 else 0
    bfi_h = bfi_natural_dims[1] if bfi_natural_dims and len(bfi_natural_dims) > 1 else 0
    print("BFI image url:", json.dumps(bfi_image_url))
    
    if debug_mode:
        print(f"DEBUG: jp2_meta['rotation'] = {jp2_meta['rotation']} (type: {type(jp2_meta['rotation'])})")
        print(f"DEBUG: bfi_rotation = {bfi_rotation} (type: {type(bfi_rotation)})")
        print(f"DEBUG: jp2_meta['rotation'] == bfi_rotation: {jp2_meta['rotation'] == bfi_rotation}")
        print(f"DEBUG: transform_data = {transform_data}")

    template_data = {
        "jp2_map_url": json.dumps(full_jp2_url),
        "jp2_full_size": json.dumps([jp2_meta["width"], jp2_meta["height"]]),
        "jp2_initial_view_rotation_deg": json.dumps(jp2_meta["rotation"]),
        "bfi_image_url": json.dumps(bfi_image_url),
        "bfi_css_rotation_deg": json.dumps(bfi_rotation),
        "h_jp2_to_bfi": json.dumps(h_jp2_to_bfi),
        "bfi_natural_width": json.dumps(bfi_w),
        "bfi_natural_height": json.dumps(bfi_h),
        "test_points_jp2": [],
        "title": f"Slice Viewer - BSID {biosample_id}, Slice {slice_number_str}",
        "current_biosample_id": json.dumps(biosample_id),
        "current_slice_number": json.dumps(slice_number_str),
        "available_biosample_ids": json.dumps(available_b_ids),
        "available_slice_numbers": json.dumps(available_s_nums),
        "mri_image_url": json.dumps(full_mri_url),
        "mri_full_size": json.dumps([mri_meta["width"], mri_meta["height"]]),
        "mri_initial_view_rotation_deg": json.dumps(mri_meta["rotation"]),
        "nissl_image_url": json.dumps(full_hae_url),
        "nissl_full_size": json.dumps([hae_meta["width"], hae_meta["height"]]),
        "nissl_initial_view_rotation_deg": json.dumps(hae_meta["rotation"]),
        "viewer_url_template": json.dumps(f"/viewer/__BID__/__SID__/")
    }
    logger.debug(f"Data being passed to template for /viewer/{biosample_id}/{slice_number_str}: {template_data}")

    return render(request, 'brainviewer/split.html', template_data)
def viewer_view_split_test(request, biosample_id, slice_number_str, port_no=8000):
    return _viewer_view_split_common(request, biosample_id, slice_number_str, port_no, debug_mode=True)
