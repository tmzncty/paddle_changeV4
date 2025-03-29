# -*- coding: utf-8 -*-
import fitz
import json
import os
import re
import time
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import io
import sys
import concurrent.futures
from datetime import datetime, timedelta
from colorama import Fore, Style, init
from natsort import natsorted, ns
import logging
import itertools
import gc


# Initialize colorama
init(autoreset=True)

# --- Configuration Variables ---
OCR_RESULTS_DIR = "C:\\Users\\z\\Desktop\\K0001_001_大般若波羅蜜多經_26\\json"
IMAGE_BASE_DIR = "C:\\Users\\z\\Desktop\\K0001_001_大般若波羅蜜多經_26\\jpg"
OUTPUT_BASE_DIRECTORY = "C:\\Users\\z\\Desktop\\K0001_001_大般若波羅蜜多經_26"
Y_OFFSET = 130 # Vertical shift (positive shifts down)
X_OFFSET = 5   # <<< NEW: Horizontal shift (positive shifts right)
NUM_PROCESSES = 8
SAVE_ENHANCED_IMAGES = False
ENHANCE_IMAGES = False  # Set to False if original image is small.
ENHANCED_IMAGE_SUFFIX = "_enhanced"
# --- End Configuration Variables ---


# --- Logging Setup ---
def setup_logger(log_dir, name):
    """Sets up a logger with both file and console handlers."""
    logger = logging.getLogger(name)
    if logger.hasHandlers(): # Avoid adding multiple handlers if called again
        return logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File Handler
    log_file = os.path.join(log_dir, f"{name}.log")
    # Ensure log directory exists before creating file handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console Handler (for info and higher levels)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # Set to INFO
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
# --- End Logging Setup ---

# --- Spinner ---
def create_spinner():
    """Creates an infinite spinner cycle."""
    return itertools.cycle([".", "..", "..."])

def print_with_spinner(spinner):
    """Prints the next spinner character."""
    sys.stdout.write(f"\r{next(spinner)} ")
    sys.stdout.flush()
# --- End Spinner ---


def get_timestamp():
    """Gets the current timestamp in a standard format."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def print_with_time(message, color=Fore.WHITE, log_level=logging.INFO, logger=None):
    """Prints a message with a timestamp and logs it."""
    timestamp = get_timestamp()
    formatted_message = f"{timestamp} - {message}"
    print(f"{color}{formatted_message}{Style.RESET_ALL}")
    if logger:
        # Map log levels correctly
        log_func = getattr(logger, logging.getLevelName(log_level).lower(), logger.info)
        log_func(message)


def enhance_image(image_path, output_dir=None):
    """Enhances a single image, returning path or bytes, and closing the image."""
    try:
        with Image.open(image_path) as img:  # Use context manager
            img = img.convert('L') # Convert to grayscale
            img_np = np.array(img)

            # Adaptive thresholding (adjust sensitivity if needed)
            threshold_value = np.mean(img_np) - np.std(img_np) / 2
            threshold_value = max(0, min(threshold_value, 255)) # Clamp value
            img_np = np.where(img_np > threshold_value, 255, 0).astype(np.uint8)
            img = Image.fromarray(img_np)

            # Sharpening
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

            if output_dir:
                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.basename(image_path)
                # Use a safe way to join paths
                output_path = os.path.join(output_dir, base_name)
                img.save(output_path, format='JPEG') # Specify format, use JPEG for smaller files
                return output_path, None # Return path
            else:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG') # Specify format
                data = img_byte_arr.getvalue()  # Get the bytes
                # No need to close BytesIO explicitly with getvalue()
                return data, None  # Return bytes
    except Exception as e:
        # Log the error with traceback for better debugging
        logging.exception(f"Error enhancing image {image_path}: {e}") # Use logging.exception
        return None, f"Error enhancing image {image_path}: {e}"
    # Image is automatically closed here due to the 'with' statement


def process_images_in_directory(image_dir, num_processes=NUM_PROCESSES, save_enhanced=SAVE_ENHANCED_IMAGES, logger=None):
    """Enhances images or returns original paths; handles multiprocessing."""
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print_with_time(f"No images found in {image_dir}", color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
        return {}, None # Return empty dict and None for dir

    # Standardize path separators for consistency
    image_dir = os.path.normpath(image_dir)

    if not ENHANCE_IMAGES:
        print_with_time("Skipping image enhancement. Using original images.", color=Fore.YELLOW, logger=logger)
        # Return full paths directly
        return {image_file: os.path.join(image_dir, image_file) for image_file in image_files}, image_dir

    # Determine enhanced image directory path
    base_image_dir_name = os.path.basename(image_dir)
    parent_dir = os.path.dirname(image_dir)
    enhanced_image_dir = os.path.join(parent_dir, base_image_dir_name + ENHANCED_IMAGE_SUFFIX)

    if save_enhanced:
        os.makedirs(enhanced_image_dir, exist_ok=True)
        print_with_time(f"Saving enhanced images to: {enhanced_image_dir}", logger=logger)
        output_target_dir = enhanced_image_dir
    else:
        print_with_time("Enhancing images in memory (not saving files).", logger=logger)
        output_target_dir = None # Signal enhance_image to return bytes


    start_time = time.time()
    total_images = len(image_files)
    processed_count = 0
    errors = 0
    enhanced_paths_or_data = {} # Store either paths or bytes data

    # Use try-except for ProcessPoolExecutor creation
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = {
                executor.submit(enhance_image, os.path.join(image_dir, image_file), output_target_dir): image_file
                for image_file in image_files
            }

            for future in concurrent.futures.as_completed(futures):
                image_file = futures[future]
                try:
                    result, error = future.result()  # Get result (path or bytes)
                    if error:
                        print_with_time(f"Error processing {image_file}: {error}", color=Fore.RED, logger=logger, log_level=logging.ERROR)
                        errors += 1
                    elif result is not None: # Ensure result is not None before storing
                        enhanced_paths_or_data[image_file] = result # Store path or bytes
                        processed_count += 1
                    else: # Handle case where enhance_image returns (None, None) unexpectedly
                         print_with_time(f"Warning: No result returned for {image_file}", color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
                         errors += 1 # Count as error if no result


                    # Progress reporting (inside the loop for more frequent updates)
                    elapsed_time = time.time() - start_time
                    # Avoid division by zero
                    avg_time_per_image = elapsed_time / processed_count if processed_count > 0 else 0
                    remaining_images = total_images - (processed_count + errors) # Adjust remaining based on errors too
                    estimated_remaining_time = avg_time_per_image * remaining_images if avg_time_per_image > 0 else 0
                    current_speed = processed_count / elapsed_time if elapsed_time > 0 else 0
                    eta_datetime = datetime.now() + timedelta(seconds=estimated_remaining_time) if estimated_remaining_time > 0 else None

                    progress = int(50 * (processed_count + errors) / total_images) if total_images > 0 else 0 # Progress includes errors
                    bar = f"[{'=' * progress}{' ' * (50 - progress)}]"
                    eta_str = eta_datetime.strftime('%H:%M:%S') if eta_datetime else "N/A"

                    # Ensure output fits on one line if possible, clear previous line
                    sys.stdout.write("\r" + " " * 120 + "\r") # Clear line before writing new status
                    sys.stdout.write(
                        f"{get_timestamp()} - Enhancing: {bar} {processed_count+errors}/{total_images} "
                        f"| Elapsed: {Fore.BLUE}{timedelta(seconds=int(elapsed_time))}{Style.RESET_ALL} "
                        f"| ETA: {Fore.MAGENTA}{eta_str}{Style.RESET_ALL} "
                        f"| Errors: {Fore.RED}{errors}{Style.RESET_ALL} "
                        f"| Speed: {Fore.YELLOW}{current_speed:.2f} pages/s{Style.RESET_ALL}"
                    )
                    sys.stdout.flush()

                except Exception as e: # Catch exceptions from future.result() itself
                    print_with_time(f"Error getting result for {image_file}: {e}", color=Fore.RED, logger=logger, log_level=logging.ERROR)
                    errors += 1

    except Exception as pool_exc: # Catch errors during executor setup/shutdown
         print_with_time(f"Error during multiprocessing pool execution: {pool_exc}", color=Fore.RED, logger=logger, log_level=logging.CRITICAL)
         return {}, None # Return empty if pool failed

    final_elapsed_time = time.time() - start_time
    final_speed = processed_count / final_elapsed_time if final_elapsed_time > 0 and processed_count > 0 else 0 # Calculate speed based on successfully processed
    print() # Newline after progress bar
    if errors > 0:
         print_with_time(f"Image enhancement completed for {image_dir} with {errors} errors. Processed: {processed_count}/{total_images}. Speed: {final_speed:.2f} pages/s", color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
    else:
        print_with_time(f"Image enhancement completed successfully for {image_dir}. Processed: {processed_count}/{total_images}. Speed: {final_speed:.2f} pages/s", color=Fore.GREEN, logger=logger)

    # Return the dict of paths/data and the directory *where enhanced images are saved*, or None if not saved
    return enhanced_paths_or_data, enhanced_image_dir if save_enhanced else None


def get_image_and_json_paths(enhanced_paths_or_data, json_base_dir, image_file, original_image_base_dir, sub_dir_name, logger=None):
    """
    Gets image data (path or bytes) and JSON path for a given image file.
    Handles naming conventions and finds the corresponding JSON.
    Returns image_data, json_path.
    """
    image_data = enhanced_paths_or_data.get(image_file)
    if image_data is None:
        print_with_time(f"Image data/path not found in provided dictionary for: {image_file}", color=Fore.RED, logger=logger, log_level=logging.ERROR)
        return None, None

    # --- JSON Path ---
    # 1. Extract parts from the image file name (consistent with original structure).
    image_base_name = os.path.splitext(image_file)[0]  # e.g., "K0001_001_26_0001"
    parts = image_base_name.split("_")
    # Expecting format like K0001_001_26_0001 -> 4 parts
    if len(parts) < 4: # Be flexible if there are extra underscores later
        print_with_time(f"Warning: Unexpected image file name format: {image_file}. Attempting to parse.", color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
        # Try to extract based on the last part being the page number if possible
        if len(parts) >= 1 and parts[-1].isdigit():
            page_number_str = parts[-1]
            base_prefix = "_".join(parts[:-1]) # Reconstruct base
            # Heuristic: Assume the JSON name follows the same base prefix + page number
            expected_json_name = f"{base_prefix}_{page_number_str}.json"
        else:
             print_with_time(f"Cannot reliably parse image file name: {image_file}. Skipping JSON search.", color=Fore.RED, logger=logger, log_level=logging.ERROR)
             return image_data, None # Return image data but no JSON path
    else:
         # Standard case K0001_001_26_0001
         expected_json_name = f"{image_base_name}.json"


    # 3. Search for the JSON file within subdirectories of json_base_dir.
    json_path = None
    found = False
    # Normalize base dir path
    json_base_dir = os.path.normpath(json_base_dir)

    # Walk through the json_base_dir to find the file
    for root, _, files in os.walk(json_base_dir):
        if expected_json_name in files:
            json_path = os.path.join(root, expected_json_name)
            found = True
            break # Found it

    if not found:
        # Log only if not found after searching all subdirs
        print_with_time(f"No corresponding JSON file found for image: {image_file} (Expected: {expected_json_name} in {json_base_dir})", color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
        return image_data, None # Return image data, but None for JSON path

    # No scaling needed here anymore, it's calculated in process_and_create_pdfs
    return image_data, json_path


# <<< MODIFIED function signature to accept x_offset >>>
def process_and_create_pdfs(sub_dir_name, sub_dir_path, image_dir, output_base_dir,
                            y_offset=Y_OFFSET, x_offset=X_OFFSET, # <<< Added x_offset parameter
                            save_enhanced=SAVE_ENHANCED_IMAGES, logger=None):
    """Processes a single subdirectory and creates its PDF."""
    start_time = time.time()
    print_with_time(f"Starting PDF creation for: {sub_dir_name}", logger=logger)


    json_dir = OCR_RESULTS_DIR # Use global config

    # Construct output PDF path within the main output directory
    output_pdf_name = f"{sub_dir_name}_searchable.pdf"
    output_pdf_path = os.path.join(output_base_dir, output_pdf_name)

    # Check if PDF exists *before* processing images
    if os.path.exists(output_pdf_path):
        print_with_time(f"Skipping {sub_dir_name} (PDF already exists at {output_pdf_path}).", color=Fore.YELLOW, logger=logger)
        return 0, 0, 0, f"Skipped (PDF exists): {output_pdf_path}" # pages, duration, errors, message

    # --- Image Processing ---
    # Process images (enhance or get original paths)
    # Pass the logger to this function too
    enhanced_paths_or_data, _ = process_images_in_directory(
        image_dir,
        save_enhanced=save_enhanced,
        logger=logger
    )

    if not enhanced_paths_or_data:
        error_message = f"No image data/paths returned from image processing for {sub_dir_name}."
        print_with_time(error_message, color=Fore.RED, logger=logger, log_level=logging.ERROR)
        return 0, time.time() - start_time, 1, error_message # 0 pages, duration, 1 error, msg

    # Sort image files naturally based on the keys (filenames) from the results
    image_files = natsorted(enhanced_paths_or_data.keys(), alg=ns.PATH)


    # --- PDF Creation ---
    total_pages_in_batch = len(image_files)
    processed_pages_count = 0
    pdf_errors = 0
    error_messages_list = [] # Collect specific error messages

    doc = None # Initialize doc to None
    try:
        doc = fitz.open() # Create new PDF document

        for image_file in image_files:
            page_start_time = time.time()
            try:
                # Get image data (path or bytes) and corresponding JSON path
                image_data, json_path = get_image_and_json_paths(
                    enhanced_paths_or_data,
                    json_dir,
                    image_file,
                    IMAGE_BASE_DIR, # Pass original base dir if needed for context
                    sub_dir_name,
                    logger=logger
                )

                # Skip if essential data is missing
                if image_data is None:
                    pdf_errors += 1
                    msg = f"Skipping page (missing image data): {image_file}"
                    error_messages_list.append(msg)
                    print_with_time(msg, color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
                    continue # Skip to next image file

                if json_path is None:
                    pdf_errors += 1
                    msg = f"Skipping page (missing JSON): {image_file}"
                    error_messages_list.append(msg)
                    print_with_time(msg, color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
                    # Decide if you want to add the image anyway or skip
                    # For now, skipping if JSON is missing as text overlay is the goal
                    continue # Skip to next image file

                # --- Load JSON Data ---
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        ocr_data = json.load(f)
                except FileNotFoundError:
                     pdf_errors += 1
                     msg = f"Skipping page (JSON file not found at path): {json_path}"
                     error_messages_list.append(msg)
                     print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                     continue
                except json.JSONDecodeError:
                    pdf_errors += 1
                    msg = f"Skipping page (Invalid JSON format): {json_path}"
                    error_messages_list.append(msg)
                    print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                    continue
                except Exception as json_e: # Catch other potential file reading errors
                    pdf_errors += 1
                    msg = f"Skipping page (Error reading JSON {json_path}): {json_e}"
                    error_messages_list.append(msg)
                    print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                    continue


                # --- Validate JSON Structure ---
                if "chars" not in ocr_data or "coors" not in ocr_data:
                    pdf_errors += 1
                    msg = f"Skipping page (JSON missing 'chars' or 'coors'): {json_path}"
                    error_messages_list.append(msg)
                    print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                    continue
                if len(ocr_data["chars"]) != len(ocr_data["coors"]):
                    pdf_errors += 1
                    msg = f"Skipping page (JSON 'chars' and 'coors' length mismatch): {json_path}"
                    error_messages_list.append(msg)
                    print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                    continue

                # --- Add Page with Image to PDF ---
                page = None # Initialize page variable
                img_width, img_height = 0, 0
                try:
                    if isinstance(image_data, str): # It's a path
                        with Image.open(image_data) as img:
                            img_width, img_height = img.size
                        page = doc.new_page(width=img_width, height=img_height)
                        page.insert_image(page.rect, filename=image_data)
                    else: # It's bytes
                         with Image.open(io.BytesIO(image_data)) as img:
                            img_width, img_height = img.size
                         page = doc.new_page(width=img_width, height=img_height)
                         page.insert_image(page.rect, stream=image_data)

                    # Optional: Clean up image bytes if not saved to disk
                    if not save_enhanced and isinstance(image_data, bytes):
                         del enhanced_paths_or_data[image_file] # Remove from dict to free memory
                         gc.collect()

                except Exception as img_err:
                    pdf_errors += 1
                    msg = f"Skipping page (Error inserting image {image_file}): {img_err}"
                    error_messages_list.append(msg)
                    print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                    if page in doc: # Remove partially created page if error occurred
                        doc.delete_page(page.number)
                    continue # Skip text insertion for this page

                # --- Calculate Scaling Factors ---
                # Use dimensions from the loaded image for the *current* page size
                # Use dimensions from JSON ('Width', 'Height') for the *original* coordinate system
                original_json_width = ocr_data.get("Width")
                original_json_height = ocr_data.get("Height")

                if not original_json_width or not original_json_height:
                     print_with_time(f"Warning: JSON {json_path} missing 'Width' or 'Height'. Scaling might be inaccurate. Falling back to image dims.", color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
                     # Fallback: Use current image dimensions if JSON lacks original size.
                     # This assumes the coordinates in JSON are relative to the image they came from,
                     # even if Width/Height keys are missing. If coords are absolute, this fallback is wrong.
                     original_json_width = original_json_width or img_width
                     original_json_height = original_json_height or img_height


                # Check for zero dimensions to prevent division by zero errors
                if original_json_width == 0 or original_json_height == 0:
                     pdf_errors += 1
                     msg = f"Skipping page (Invalid original dimensions in JSON {json_path}): Width={original_json_width}, Height={original_json_height}"
                     error_messages_list.append(msg)
                     print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                     doc.delete_page(page.number) # Remove the page added earlier
                     continue


                scale_x = img_width / original_json_width
                scale_y = img_height / original_json_height


                # --- Insert Invisible Text ---
                chars_added = 0
                for i, char in enumerate(ocr_data["chars"]):
                    if not char or char.isspace(): # Skip empty strings or whitespace characters
                        continue

                    coors = ocr_data["coors"][i]

                    # Validate coordinate structure
                    if not isinstance(coors, list) or len(coors) not in (4, 8):
                        print_with_time(f"Warning: Invalid coordinate format {coors} in {json_path} for char '{char}'. Skipping character.",
                                        color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
                        continue

                    try:
                        # Convert coordinates to float first for scaling, then to int if needed
                        coors_float = [float(c) for c in coors]

                        if len(coors_float) == 8: # Polygon format [x1,y1, x2,y2, x3,y3, x4,y4]
                            # Find min/max x and y from the polygon points
                            all_x = coors_float[0::2] # x values
                            all_y = coors_float[1::2] # y values
                            xmin = min(all_x) * scale_x
                            ymin = min(all_y) * scale_y
                            xmax = max(all_x) * scale_x
                            ymax = max(all_y) * scale_y
                        else:  # len(coors_float) == 4 -> [xmin, ymin, xmax, ymax]
                            xmin, ymin, xmax, ymax = coors_float
                            xmin *= scale_x
                            ymin *= scale_y
                            xmax *= scale_x
                            ymax *= scale_y

                        # Ensure coordinates are valid numbers
                        if not all(np.isfinite([xmin, ymin, xmax, ymax])):
                             print_with_time(f"Warning: Non-finite coordinates after scaling: {[xmin, ymin, xmax, ymax]} in {json_path}, char '{char}'. Skipping.",
                                             color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
                             continue

                        # Create the rectangle for text insertion
                        # <<< Apply both X and Y offsets here >>>
                        rect = fitz.Rect(xmin + x_offset, ymin + y_offset,
                                         xmax + x_offset, ymax + y_offset)

                        # Ensure rect dimensions are positive
                        if rect.width <= 0 or rect.height <= 0:
                             # This can happen with very small boxes or invalid coordinates
                             print_with_time(f"Warning: Invalid rectangle dimensions {rect} after offset/scaling for char '{char}' in {json_path}. Skipping char.",
                                             color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
                             continue


                        # --- Dynamic Font Size Calculation ---
                        fontsize = rect.height * 0.9 # Start with 90% of height
                        fontsize = max(1, fontsize) # Ensure fontsize is at least 1

                        # Check if text fits horizontally, reduce fontsize if needed
                        # Use a CJK font built into PyMuPDF or ensure one is available
                        # 'china-s' is a common built-in option for simplified Chinese
                        fontname = "china-s"
                        try:
                            text_width = fitz.get_text_length(char, fontname=fontname, fontsize=fontsize)
                            while text_width > rect.width and fontsize > 1:
                                fontsize *= 0.9 # Reduce fontsize proportionally
                                text_width = fitz.get_text_length(char, fontname=fontname, fontsize=fontsize)
                            fontsize = max(1, fontsize) # Ensure it doesn't go below 1
                        except Exception as font_e:
                            # This might happen if the font doesn't support the character
                             print_with_time(f"Warning: Error calculating text width for char '{char}' with font '{fontname}'. Using default size. Error: {font_e}",
                                              color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
                            # Keep the initial calculated fontsize or a default small one
                             fontsize = max(1, rect.height * 0.9)


                        # Insert the text invisibly (render_mode=3)
                        page.insert_text(rect.bottom_left,  # Insertion point (bottom-left for vertical CJK)
                                         char,
                                         fontname=fontname,
                                         fontsize=fontsize,
                                         rotate=0, # Assume horizontal text for now
                                         color=None,      # No visible color needed
                                         fill=None,       # No fill needed
                                         render_mode=3)   # Render mode 3 makes text invisible but searchable/selectable
                        chars_added +=1

                    except (ValueError, TypeError) as coord_err:
                        print_with_time(f"Warning: Invalid coordinate values {coors} in {json_path} for char '{char}'. Skipping character. Error: {coord_err}",
                                        color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
                        continue
                    except Exception as text_ins_e: # Catch errors during text insertion
                        print_with_time(f"Error inserting char '{char}' from {json_path}: {text_ins_e}", color=Fore.RED, logger=logger, log_level=logging.ERROR)
                        # Continue with the next character, but log the error
                        pdf_errors +=1 # Count char insertion errors? Maybe too granular. Count page errors instead.

                # Log page processing time and char count
                page_duration = time.time() - page_start_time
                #print_with_time(f"Processed page {image_file} ({processed_pages_count+1}/{total_pages_in_batch}) with {chars_added} characters in {page_duration:.2f}s.", logger=logger, log_level=logging.DEBUG)
                processed_pages_count += 1

            except Exception as page_proc_e: # Catch broader errors during the processing of a single page/file
                pdf_errors += 1
                msg = f"Critical error processing page {image_file}: {page_proc_e}"
                error_messages_list.append(msg)
                print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                # Attempt to remove the potentially problematic page from the doc if it was added
                if page and page in doc:
                    try:
                        doc.delete_page(page.number)
                        print_with_time(f"Removed page {page.number} due to error.", color=Fore.YELLOW, logger=logger)
                    except Exception as del_e:
                         print_with_time(f"Error trying to remove faulty page {page.number}: {del_e}", color=Fore.RED, logger=logger)
                # Continue to the next file
                continue

        # --- PDF Saving ---
        if processed_pages_count > 0: # Only save if pages were actually added
            print_with_time(f"Saving PDF for {sub_dir_name} with {processed_pages_count} pages...", logger=logger)
            try:
                # Use garbage collection and compression for smaller files
                doc.save(output_pdf_path, garbage=4, deflate=True, clean=True)
                end_time = time.time()
                duration = end_time - start_time
                if pdf_errors == 0:
                    print_with_time(f"Searchable PDF created successfully: {output_pdf_path} in {duration:.2f} seconds.", color=Fore.GREEN, logger=logger)
                else:
                    print_with_time(f"Searchable PDF created with {pdf_errors} errors/skipped pages: {output_pdf_path} in {duration:.2f} seconds.", color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)

            except Exception as save_e:
                pdf_errors += 1 # Increment error count for saving failure
                error_msg = f"FATAL: Error saving PDF {output_pdf_path}: {save_e}"
                error_messages_list.append(error_msg)
                print_with_time(error_msg, color=Fore.RED, logger=logger, log_level=logging.CRITICAL)
                # Attempt to delete potentially corrupted output file
                if os.path.exists(output_pdf_path):
                    try:
                        os.remove(output_pdf_path)
                        print_with_time(f"Removed potentially corrupted PDF: {output_pdf_path}", color=Fore.YELLOW, logger=logger)
                    except OSError as del_err:
                        print_with_time(f"Error removing corrupted PDF {output_pdf_path}: {del_err}", color=Fore.RED, logger=logger)

        elif not doc.is_closed and doc.page_count == 0:
             print_with_time(f"No pages were successfully processed for {sub_dir_name}. PDF not saved.", color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
             # No need to return an error specifically for *not saving* an empty PDF
        else:
            # This case should ideally not be reached if logic is correct
             print_with_time(f"Unknown state for PDF {sub_dir_name}. Processed pages: {processed_pages_count}. PDF page count: {doc.page_count if doc else 'N/A'}.", color=Fore.RED, logger=logger)



        # --- Log Summary for this Subdirectory ---
        duration = time.time() - start_time
        logger.info(f"Finished processing subdirectory: {sub_dir_name}")
        logger.info(f"Pages processed: {processed_pages_count}/{total_pages_in_batch}")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Errors encountered during PDF creation: {pdf_errors}")
        # Log specific errors collected
        for msg in error_messages_list:
            logger.error(msg) # Log specific errors as ERROR level

        # --- Error File Moving --- (Optional: uncomment if needed)
        # if pdf_errors > 0:
        #     error_sub_dir = os.path.join(output_base_dir, "errors", sub_dir_name)
        #     os.makedirs(error_sub_dir, exist_ok=True)
        #     print_with_time(f"Moving related files for errored pages in {sub_dir_name} to {error_sub_dir}", color=Fore.YELLOW, logger=logger)
        #     # Implement file moving logic based on error_messages_list if required
        #     # This requires careful parsing of the error messages to get filenames

        return processed_pages_count, duration, pdf_errors, "\n".join(error_messages_list)

    except Exception as e: # Catch unexpected errors in the main try block
        error_msg = f"An unexpected critical error occurred processing {sub_dir_name}: {e}"
        print_with_time(error_msg, color=Fore.RED, logger=logger, log_level=logging.CRITICAL)
        # Log traceback for critical errors
        logger.exception(f"Traceback for critical error in {sub_dir_name}:")
        if error_messages_list: # Combine with previous errors if any
            error_msg = "\n".join(error_messages_list) + "\n" + error_msg
        # Return estimates: assume 0 pages processed correctly in this case, count this as 1 major error
        return 0, time.time() - start_time, pdf_errors + 1, error_msg

    finally:
        # --- Cleanup ---
        if doc is not None and not doc.is_closed:
            doc.close()
            #print_with_time(f"Closed PDF document object for {sub_dir_name}.", logger=logger, log_level=logging.DEBUG)
        # Explicitly delete large objects and collect garbage
        del doc
        if 'enhanced_paths_or_data' in locals():
             del enhanced_paths_or_data
        gc.collect()
        #print_with_time(f"Garbage collected after processing {sub_dir_name}.", logger=logger, log_level=logging.DEBUG)



def main(ocr_results_dir=OCR_RESULTS_DIR, image_base_dir=IMAGE_BASE_DIR, output_base_dir=OUTPUT_BASE_DIRECTORY,
         y_offset=Y_OFFSET, x_offset=X_OFFSET, # <<< Added x_offset parameter
         num_processes=NUM_PROCESSES, save_enhanced_images=SAVE_ENHANCED_IMAGES):
    """Main function to orchestrate the PDF creation process."""

    # --- Setup Output and Logging ---
    os.makedirs(output_base_dir, exist_ok=True)
    log_dir = os.path.join(output_base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Setup a main logger for the overall process
    main_logger = setup_logger(log_dir, "pdf_creation_main")
    print_with_time("-----------------------------------------", logger=main_logger)
    print_with_time("Starting PDF Creation Process", color=Fore.CYAN, logger=main_logger)
    print_with_time(f"Configuration:", logger=main_logger)
    print_with_time(f"  OCR JSON Dir: {ocr_results_dir}", logger=main_logger)
    print_with_time(f"  Image Base Dir: {image_base_dir}", logger=main_logger)
    print_with_time(f"  Output Dir: {output_base_dir}", logger=main_logger)
    print_with_time(f"  Enhance Images: {ENHANCE_IMAGES}", logger=main_logger)
    print_with_time(f"  Save Enhanced: {save_enhanced_images}", logger=main_logger)
    print_with_time(f"  Y-Offset: {y_offset}", logger=main_logger)
    print_with_time(f"  X-Offset: {x_offset}", logger=main_logger) # <<< Log X_OFFSET
    print_with_time(f"  Max Processes: {num_processes}", logger=main_logger)
    print_with_time("-----------------------------------------", logger=main_logger)


    # --- Identify Subdirectories to Process ---
    sub_dirs_to_process = []
    overall_start_time = time.time()

    # Normalize base image directory path
    image_base_dir = os.path.normpath(image_base_dir)

    if not os.path.isdir(image_base_dir):
        print_with_time(f"Error: Image base directory not found or is not a directory: {image_base_dir}", color=Fore.RED, logger=main_logger, log_level=logging.CRITICAL)
        return # Exit if base dir is invalid

    # Find subdirectories containing images directly within the image_base_dir
    for item_name in os.listdir(image_base_dir):
        item_path = os.path.join(image_base_dir, item_name)
        if os.path.isdir(item_path):
            # Check if the subdirectory contains image files (avoids processing empty dirs)
            has_images = any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(item_path))
            if has_images:
                 # Use the subdirectory path itself as the 'image_dir' for process_and_create_pdfs
                 sub_dirs_to_process.append({
                     'name': item_name,        # Subdirectory name (e.g., "K0001_001_26")
                     'path': item_path,        # Full path to the subdirectory
                     'image_dir': item_path     # The directory containing images for this task
                 })
            else:
                print_with_time(f"Skipping empty or non-image directory: {item_path}", color=Fore.WHITE, logger=main_logger, log_level=logging.DEBUG)


    if not sub_dirs_to_process:
        print_with_time(f"No subdirectories with images found to process in: {image_base_dir}", color=Fore.YELLOW, logger=main_logger, log_level=logging.WARNING)
        return # Exit if nothing to process

    print_with_time(f"Found {len(sub_dirs_to_process)} subdirectories with images to process.", color=Fore.CYAN, logger=main_logger)


    # --- Calculate Total Estimated Image Count (for progress) ---
    total_image_count_estimate = 0
    for dir_info in sub_dirs_to_process:
        try:
            # Count directly from the source image directory
            image_files = [f for f in os.listdir(dir_info['image_dir']) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_image_count_estimate += len(image_files)
        except Exception as count_e:
             print_with_time(f"Warning: Could not count images in {dir_info['image_dir']}: {count_e}", color=Fore.YELLOW, logger=main_logger, log_level=logging.WARNING)


    print_with_time(f"Estimated total images across all directories: {total_image_count_estimate}", color=Fore.CYAN, logger=main_logger)


    # --- Process Subdirectories Concurrently ---
    results = []
    overall_processed_pages = 0
    overall_errors = 0
    processed_pdf_count = 0
    pdf_creation_start_time = time.time() # Start timing specifically for PDF generation loop

    # Create the ProcessPoolExecutor
    try:
         with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Submit tasks
            futures_map = {}
            for dir_info in sub_dirs_to_process:
                # Setup a dedicated logger for each subprocess/task
                process_logger = setup_logger(log_dir, f"process_{dir_info['name']}")
                # Submit the task with all necessary arguments
                # <<< Pass x_offset to the submitted function >>>
                future = executor.submit(process_and_create_pdfs,
                                         dir_info['name'],
                                         dir_info['path'],
                                         dir_info['image_dir'],
                                         output_base_dir,
                                         y_offset,
                                         x_offset, # <<< Pass the value
                                         save_enhanced_images,
                                         process_logger) # Pass the specific logger
                futures_map[future] = dir_info['name'] # Map future back to directory name

            # Process results as they complete
            spinner = create_spinner()
            total_tasks = len(futures_map)

            for future in concurrent.futures.as_completed(futures_map):
                dir_name = futures_map[future]
                print_with_spinner(spinner) # Update spinner
                try:
                    # Get result: pages_in_pdf, duration, errors_in_pdf, error_messages
                    pages_in_pdf, duration, errors_in_pdf, error_messages = future.result()
                    results.append({
                        'dir': dir_name,
                        'pages': pages_in_pdf,
                        'duration': duration,
                        'errors': errors_in_pdf,
                        'messages': error_messages
                    })
                    overall_processed_pages += pages_in_pdf
                    overall_errors += errors_in_pdf
                    processed_pdf_count += 1 # Increment count for completed tasks (even if errors occurred)

                    # --- Progress Reporting ---
                    elapsed_pdf_time = time.time() - pdf_creation_start_time

                    # Calculate ETA based on pages processed so far vs estimated total
                    if overall_processed_pages > 0 and total_image_count_estimate > 0:
                        avg_time_per_page = elapsed_pdf_time / overall_processed_pages
                        remaining_pages_estimate = total_image_count_estimate - overall_processed_pages
                        estimated_remaining_seconds = avg_time_per_page * max(0, remaining_pages_estimate) # Ensure non-negative
                        eta_datetime = datetime.now() + timedelta(seconds=estimated_remaining_seconds)
                        eta_str = eta_datetime.strftime('%H:%M:%S')
                    else:
                        eta_str = "Calculating..." # Not enough data yet

                    # Calculate speed based on pages processed
                    current_speed = overall_processed_pages / elapsed_pdf_time if elapsed_pdf_time > 0 else 0

                    # Progress bar based on completed *tasks* (PDFs)
                    progress = int(50 * processed_pdf_count / total_tasks) if total_tasks > 0 else 0
                    bar = f"[{'=' * progress}{' ' * (50 - progress)}]"

                    # Dynamic ETA color
                    eta_color = Fore.MAGENTA if eta_str != "Calculating..." else Fore.YELLOW

                    # Clear line and print status
                    sys.stdout.write("\r" + " " * 130 + "\r") # Adjust clearing width if needed
                    sys.stdout.write(
                        f"{get_timestamp()} - PDF Creation: {bar} {processed_pdf_count}/{total_tasks} "
                        f"| Pages: {overall_processed_pages} ({overall_errors} errs) "
                        f"| Elapsed: {Fore.BLUE}{timedelta(seconds=int(elapsed_pdf_time))}{Style.RESET_ALL} "
                        f"| ETA: {eta_color}{eta_str}{Style.RESET_ALL} "
                        f"| Speed: {Fore.YELLOW}{current_speed:.2f} pages/s{Style.RESET_ALL} "
                    )
                    sys.stdout.flush()

                except Exception as e: # Catch errors from future.result() itself (e.g., task raised exception)
                    processed_pdf_count += 1 # Count it as processed task, but with error
                    overall_errors += 1 # Increment overall error count significantly
                    print_with_time(f"\nError retrieving result for subdirectory '{dir_name}': {e}", color=Fore.RED, logger=main_logger, log_level=logging.ERROR)
                    # Log the traceback for unexpected errors during future retrieval
                    main_logger.exception(f"Traceback for future result error ({dir_name}):")
                    results.append({ # Add an error record
                         'dir': dir_name,
                         'pages': 0,
                         'duration': 0,
                         'errors': 1, # Mark this task as having a major error
                         'messages': f"Failed to get result: {e}"
                    })


    except Exception as pool_exec_err:
         print_with_time(f"\nCritical error during ProcessPoolExecutor execution: {pool_exec_err}", color=Fore.RED, logger=main_logger, log_level=logging.CRITICAL)
         main_logger.exception("Traceback for ProcessPoolExecutor error:")
         # No reliable summary possible if the pool fails catastrophically


    # --- Final Summary ---
    print() # Newline after the progress bar
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    average_speed = overall_processed_pages / overall_duration if overall_duration > 0 else 0

    print_with_time("-----------------------------------------", logger=main_logger)
    print_with_time("Processing Summary:", color=Fore.CYAN, logger=main_logger)
    print_with_time(f"Total execution time: {timedelta(seconds=int(overall_duration))} ({overall_duration:.2f} seconds)", color=Fore.CYAN, logger=main_logger)
    print_with_time(f"Total PDFs attempted: {processed_pdf_count}/{len(sub_dirs_to_process)}", color=Fore.CYAN, logger=main_logger)
    print_with_time(f"Total pages processed across all PDFs: {overall_processed_pages}", color=Fore.CYAN, logger=main_logger)
    print_with_time(f"Total errors/skipped pages during PDF creation: {overall_errors}", color=Fore.RED if overall_errors > 0 else Fore.CYAN, logger=main_logger)
    print_with_time(f"Average processing speed: {average_speed:.2f} pages/second", color=Fore.CYAN, logger=main_logger)
    print_with_time("-----------------------------------------", logger=main_logger)

    # Optionally print details for PDFs with errors
    errors_found = False
    for res in results:
        if res['errors'] > 0:
            if not errors_found:
                 print_with_time("Details for PDFs with errors:", color=Fore.YELLOW, logger=main_logger, log_level=logging.WARNING)
                 errors_found = True
            print_with_time(f"  - Directory '{res['dir']}': {res['errors']} errors. Pages added: {res['pages']}. Duration: {res['duration']:.2f}s.", color=Fore.YELLOW, logger=main_logger, log_level=logging.WARNING)
            # Log the specific messages for this directory from the main logger as well
            if res['messages']:
                 for line in res['messages'].splitlines():
                    if line.strip(): # Log non-empty lines
                         main_logger.warning(f"    Error detail ({res['dir']}): {line}")

    if not errors_found:
        print_with_time("All PDF tasks completed without reported errors.", color=Fore.GREEN, logger=main_logger)

    print_with_time("PDF Creation Process Finished.", color=Fore.CYAN, logger=main_logger)


if __name__ == "__main__":
    try:
        # Call main with the globally defined constants or allow overrides if needed
        main(ocr_results_dir=OCR_RESULTS_DIR,
             image_base_dir=IMAGE_BASE_DIR,
             output_base_dir=OUTPUT_BASE_DIRECTORY,
             y_offset=Y_OFFSET,
             x_offset=X_OFFSET, # <<< Pass X_OFFSET
             num_processes=NUM_PROCESSES,
             save_enhanced_images=SAVE_ENHANCED_IMAGES)
    except KeyboardInterrupt:
        print_with_time("\nProcess interrupted by user (Ctrl+C). Exiting.", color=Fore.RED)
        # Attempt to shutdown multiprocessing pools gracefully if possible (tricky here)
        sys.exit(1) # Indicate abnormal termination
    except Exception as main_exc:
        # Log any uncaught exception from the main execution flow
        # Note: A main logger might not be initialized if error happens very early
        print(f"{Fore.RED}{get_timestamp()} - A critical error occurred in the main execution block: {main_exc}{Style.RESET_ALL}")
        # Try logging if possible
        try:
            # Assume default log dir if main setup failed
            log_dir_fallback = os.path.join(OUTPUT_BASE_DIRECTORY, "logs")
            os.makedirs(log_dir_fallback, exist_ok=True)
            critical_logger = setup_logger(log_dir_fallback, "critical_error")
            critical_logger.critical(f"Uncaught exception in main: {main_exc}")
            critical_logger.exception("Traceback:")
        except Exception as log_err:
            print(f"{Fore.RED}{get_timestamp()} - Additionally, failed to write error to log: {log_err}{Style.RESET_ALL}")
        sys.exit(1) # Indicate abnormal termination