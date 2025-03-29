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
OCR_RESULTS_DIR = "I:\\PDF_Searchable\\json"
IMAGE_BASE_DIR = "I:\\PDF_Searchable\\output_pdf_png"
OUTPUT_BASE_DIRECTORY = "I:\\PDF_Searchable\\output_pdfs"
Y_OFFSET = 18  # Modified Y offset
X_OFFSET = 3   # New X offset
NUM_PROCESSES = 4
SAVE_ENHANCED_IMAGES = False
ENHANCE_IMAGES = False   # Set to False if original image is small.
ENHANCED_IMAGE_SUFFIX = "_enhanced"
# --- End Configuration Variables ---


# --- Logging Setup ---
def setup_logger(log_dir, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File Handler
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding='utf-8')
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
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def print_with_time(message, color=Fore.WHITE, log_level=logging.INFO, logger=None):
    """Prints a message with a timestamp and logs it."""
    timestamp = get_timestamp()
    formatted_message = f"{timestamp} - {message}"
    print(f"{color}{formatted_message}{Style.RESET_ALL}")
    if logger:
        if log_level == logging.DEBUG:
            logger.debug(message)
        elif log_level == logging.INFO:
            logger.info(message)
        elif log_level == logging.WARNING:
            logger.warning(message)
        elif log_level == logging.ERROR:
            logger.error(message)


def enhance_image(image_path, output_dir=None):
    """Enhances a single image, returning path or bytes, and closing the image."""
    try:
        with Image.open(image_path) as img:  # Use context manager
            img = img.convert('L')
            img_np = np.array(img)

            # Adaptive thresholding
            threshold_value = np.mean(img_np) - np.std(img_np) / 2
            threshold_value = max(0, min(threshold_value, 255))
            img_np = np.where(img_np > threshold_value, 255, 0).astype(np.uint8)
            img = Image.fromarray(img_np)

            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

            if output_dir:
                base_name = os.path.basename(image_path)
                output_path = os.path.join(output_dir, base_name)
                img.save(output_path, format='PNG')
                return output_path, None
            else:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                data = img_byte_arr.getvalue()  # Get the bytes
                img_byte_arr.close() # Close the BytesIO object
                return data, None  # Return bytes
    except Exception as e:
        return None, f"Error enhancing image {image_path}: {e}"
    # Image is automatically closed here due to the 'with' statement


def process_images_in_directory(image_dir, num_processes=NUM_PROCESSES, save_enhanced=SAVE_ENHANCED_IMAGES, logger=None):
    """Enhances images or returns original paths; handles multiprocessing."""
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print_with_time(f"No images found in {image_dir}", color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
        return {}, None

    if not ENHANCE_IMAGES:
        print_with_time("Skipping image enhancement. Using original images.", color=Fore.YELLOW, logger=logger)
        return {image_file: os.path.join(image_dir, image_file) for image_file in image_files}, image_dir

    enhanced_image_dir = os.path.join(os.path.dirname(image_dir), os.path.basename(image_dir) + ENHANCED_IMAGE_SUFFIX)
    if save_enhanced:
        os.makedirs(enhanced_image_dir, exist_ok=True)
    else:
        enhanced_image_dir = None  # Explicitly set to None


    start_time = time.time()
    total_images = len(image_files)
    processed_count = 0
    errors = 0
    enhanced_paths = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {
            executor.submit(enhance_image, os.path.join(image_dir, image_file), enhanced_image_dir if save_enhanced else None): image_file
            for image_file in image_files
        }

        for future in concurrent.futures.as_completed(futures):
            image_file = futures[future]
            try:
                result, error = future.result()  # Get result (path or bytes)
                if error:
                    print_with_time(f"Error processing {image_file}: {error}", color=Fore.RED, logger=logger, log_level=logging.ERROR)
                    errors += 1
                else:
                    enhanced_paths[image_file] = result  # Store path or bytes
                    processed_count += 1

                # Progress reporting (inside the loop for more accurate timing)
                elapsed_time = time.time() - start_time
                avg_time_per_image = elapsed_time / processed_count if processed_count > 0 else 0
                remaining_images = total_images - processed_count
                estimated_remaining_time = avg_time_per_image * remaining_images
                current_speed = processed_count / elapsed_time if elapsed_time > 0 else 0
                eta = datetime.now() + timedelta(seconds=estimated_remaining_time)

                progress = int(50 * processed_count / total_images)
                bar = f"[{'=' * progress}{' ' * (50 - progress)}]"
                sys.stdout.write(
                    f"\r{get_timestamp()} - Enhancing: {bar} {processed_count}/{total_images} "
                    f"| Elapsed: {Fore.BLUE}{timedelta(seconds=int(elapsed_time))}{Style.RESET_ALL} "
                    f"| ETA: {Fore.MAGENTA}{eta.strftime('%H:%M:%S')}{Style.RESET_ALL} "
                    f"| Errors: {Fore.RED}{errors}{Style.RESET_ALL} "
                    f"| Speed: {Fore.YELLOW}{current_speed:.2f} pages/s{Style.RESET_ALL}"
                )
                sys.stdout.flush()

            except Exception as e:
                print_with_time(f"Error processing {image_file}: {e}", color=Fore.RED, logger=logger, log_level=logging.ERROR)
                errors += 1

    final_elapsed_time = time.time() - start_time
    final_speed = total_images / final_elapsed_time if final_elapsed_time > 0 else 0
    print_with_time(f"\nImage enhancement completed for {image_dir}. Speed: {final_speed:.2f} pages/s", color=Fore.GREEN, logger=logger)
    return enhanced_paths, enhanced_image_dir

def get_image_and_json_paths(enhanced_paths, enhanced_image_dir, json_base_dir, image_file, sub_dir_name):
    """Gets image and JSON paths, handling naming conventions and scaling."""

    # --- JSON Path ---
    # 1. Extract the page number from the image file name.
    image_base_name = os.path.splitext(image_file)[0]  # e.g., "page_0001"
    match = re.search(r"page_(\d+)", image_base_name)
    if not match:
        print(f"Invalid image file name format: {image_file}")
        return None, None, None, None
    page_number_str = match.group(1)  # e.g., "0001"
    page_number_int = int(page_number_str)  # Convert to integer
    page_number_padded = str(page_number_int).zfill(2) # Pad to two digits

    # 2. Extract book, volume, and the extra '26' from sub_dir_name.
    parts = sub_dir_name.split("_")
    if len(parts) < 3:  # We need at least book, volume, and the extra part
        print(f"Invalid sub_dir_name format: {sub_dir_name}")
        return None, None, None, None

    book_part = parts[0].lstrip('K').zfill(4)  # K0001 -> 0001
    volume_part = parts[1].zfill(3)  # 001 -> 001
    extra_part = parts[-1] # Get the last part which is '26' in your example

    # 3. Construct the *correct* expected JSON file name.
    expected_json_name = f"{book_part}_{volume_part}_{extra_part}_{page_number_padded}.json"

    # 4. Search for the JSON file within subdirectories of json_base_dir.
    json_path = None
    for item in os.listdir(json_base_dir):
        potential_json_subdir = os.path.join(json_base_dir, item)
        if os.path.isdir(potential_json_subdir):
            for root, _, files in os.walk(potential_json_subdir):
                if expected_json_name in files:
                    json_path = os.path.join(root, expected_json_name)
                    break
            if json_path:
                break # Found it, no need to search further
        elif os.path.isfile(potential_json_subdir) and item == expected_json_name: # Check if it's directly in the base dir
            json_path = potential_json_subdir
            break


    if not json_path:
        print(f"No corresponding JSON file found for image: {image_file} (Expected: {expected_json_name})")
        return None, None, None, None

    # --- Image Path and Scaling Factor --- (No changes here) ---
    image_data = enhanced_paths[image_file]

    if isinstance(image_data, str):
        with Image.open(image_data) as img:
            original_width, original_height = img.size
    else:
        with Image.open(io.BytesIO(image_data)) as img:
            original_width, original_height = img.size

    original_image_path = os.path.join(IMAGE_BASE_DIR, sub_dir_name, image_file)
    with Image.open(original_image_path) as img:
        orig_img_width, orig_img_height = img.size

    scale_x = original_width / orig_img_width
    scale_y = original_height / orig_img_height

    return image_data, json_path, scale_x, scale_y
def process_and_create_pdfs(sub_dir_name, sub_dir_path, image_dir, output_base_dir, y_offset=Y_OFFSET, x_offset = X_OFFSET,
                            save_enhanced=SAVE_ENHANCED_IMAGES, logger=None):
    """Processes a single subdirectory and creates its PDF."""
    start_time = time.time()

    json_dir = OCR_RESULTS_DIR

    output_pdf_name = f"{sub_dir_name}_searchable.pdf"
    output_pdf_path = os.path.join(output_base_dir, output_pdf_name)

    if os.path.exists(output_pdf_path):
        print_with_time(f"Skipping {sub_dir_name} (PDF already exists).", color=Fore.YELLOW, logger=logger)
        return 0, 0, 0, f"Skipped (PDF exists): {output_pdf_path}"

    enhanced_paths, enhanced_image_dir = process_images_in_directory(image_dir, save_enhanced=save_enhanced, logger=logger)
    image_files = natsorted(enhanced_paths.keys())

    if not image_files:
        error_message = f"No image files found in {'enhanced image dir' if enhanced_image_dir else 'image dir'}"
        print_with_time(error_message, color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
        return 0, 0, 0, error_message

    total_pages = len(image_files)
    processed_pages = 0
    errors = 0
    error_messages = []

    doc = None
    try:
        doc = fitz.open()
        for image_file in image_files:
            try:
                enhanced_image_data, json_path, scale_x, scale_y = get_image_and_json_paths(
                    enhanced_paths, enhanced_image_dir, json_dir, image_file, sub_dir_name
                )
                if not json_path or not enhanced_image_data:
                    errors += 1
                    msg = f"Skipping (no JSON or image): {image_file}"
                    error_messages.append(msg)
                    print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                    continue

                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if "FileName" not in data:
                    errors += 1
                    msg = f"Skipping (JSON missing 'FileName' key): {image_file}"
                    error_messages.append(msg)
                    print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                    continue

                if "chars" not in data or "coors" not in data:
                    errors += 1
                    msg = f"Skipping (JSON missing 'chars' or 'coors' data): {image_file}"
                    error_messages.append(msg)
                    print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                    continue

                if len(data["chars"]) != len(data["coors"]):
                    errors += 1
                    msg = f"Skipping (chars and coors length mismatch): {image_file}"
                    error_messages.append(msg)
                    print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                    continue

                if isinstance(enhanced_image_data, str):
                    with Image.open(enhanced_image_data) as img:
                        img_width, img_height = img.size
                        page = doc.new_page(width=img_width, height=img_height)
                        page.insert_image(page.rect, filename=enhanced_image_data)
                else:
                    with Image.open(io.BytesIO(enhanced_image_data)) as img:
                        img_width, img_height = img.size
                        page = doc.new_page(width=img_width, height=img_height)
                        page.insert_image(page.rect, stream=enhanced_image_data)
                    del enhanced_image_data
                    gc.collect()

                # Calculate scaling factors based on *original* JSON dimensions and the *new* image dimensions.
                original_json_width = data.get("Width", img_width)  # Fallback to img_width if not in JSON
                original_json_height = data.get("Height", img_height)  # Fallback to img_height

                scale_x = img_width / original_json_width
                scale_y = img_height / original_json_height
                

                for i, char in enumerate(data["chars"]):
                    coors = data["coors"][i]

                    if not isinstance(coors, list) or len(coors) not in (4, 8):
                        print_with_time(f"Warning: Invalid coordinates: {coors} in {json_path}, skipping character.",
                                        color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
                        continue

                    try:
                        if len(coors) == 8:
                            coors = [int(c) for c in coors]
                            xmin = min(coors[0], coors[6]) * scale_x + x_offset  # Apply X-offset
                            ymin = min(coors[1], coors[3]) * scale_y
                            xmax = max(coors[2], coors[4]) * scale_x + x_offset  # Apply X-offset
                            ymax = max(coors[5], coors[7]) * scale_y
                        else:  # len(coors) == 4
                            xmin, ymin, xmax, ymax = [int(c) for c in coors]
                            xmin = xmin * scale_x + x_offset  # Apply X-offset
                            ymin = ymin * scale_y
                            xmax = xmax * scale_x + x_offset  # Apply X-offset
                            ymax = ymax * scale_y
                    except (ValueError, TypeError):
                        print_with_time(f"Warning: Non-numeric coordinates: {coors} in {json_path}, skipping character.",
                                        color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
                        continue

                    rect = fitz.Rect(xmin, ymin + y_offset, xmax, ymax + y_offset) # Apply Y-offset

                    fontsize = rect.height * 0.9
                    while True:
                        text_width = fitz.get_text_length(char, fontname="china-s", fontsize=fontsize)
                        if text_width <= rect.width or fontsize <= 1:
                            break
                        fontsize -= 1
                    fontsize = max(1, min(fontsize, 100))

                    page.insert_text(rect.top_left, char, fontname="china-s", fontsize=fontsize,
                                    color=(0, 0, 0),
                                    fill=(1, 1, 1),
                                    render_mode=3)

                processed_pages += 1

            except Exception as e:
                errors += 1
                msg = f"Error processing JSON/image {image_file}: {e}"
                error_messages.append(msg)
                print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)

        # --- PDF Saving ---
        try:
            if doc is not None:
                doc.save(output_pdf_path, garbage=4, deflate=True)
                end_time = time.time()
                duration = end_time - start_time
                print_with_time(f"Searchable PDF created: {output_pdf_path} in {duration:.2f} seconds", color=Fore.GREEN, logger=logger)

        except Exception as e:
            errors += 1
            error_msg = f"Error saving PDF: {e}"
            error_messages.append(error_msg)
            print_with_time(error_msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)


        # --- Error Handling and File Moving (After saving) ---
        logger.info(f"Processed: {processed_pages} pages")
        duration = time.time() - start_time
        logger.info(f"Total Duration: {duration:.2f} seconds")
        logger.info(f"PDF Creation Errors: {errors}")
        for msg in error_messages:
            logger.error(msg)

        if errors > 0:
            error_sub_dir = os.path.join(output_base_dir, "errors", sub_dir_name)
            os.makedirs(error_sub_dir, exist_ok=True)

            for error_line in error_messages:
                match = re.search(r"(?<=Skipping \(no JSON or image\): )([\w.-]+)|(?<=Skipping \(JSON file not found\): I:\\\\PDF_Searchable\\\\json\\\\)([\w.-]+)|(?<=Error processing JSON/image )([\w.-]+)", error_line)
                if match:
                    error_file_base = match.group(1) or match.group(2) or match.group(3)

                    for ext in ['.png', '.jpg', '.jpeg', '.json']:
                        if ext == '.json':
                            if error_file_base:
                                image_base_name = os.path.splitext(error_file_base)[0]
                                page_number_part = image_base_name.split("_")[-1]
                                book_volume_part = "_".join(sub_dir_name.split("_")[:2])
                                error_file_name = f"{book_volume_part}_{page_number_part}{ext}" # Corrected

                                for root, _, files in os.walk(OCR_RESULTS_DIR):
                                    src_path = os.path.join(root, error_file_name)
                                    if os.path.exists(src_path):
                                        break
                                else:
                                    src_path = None

                            else:
                                continue
                        else:
                            error_file_name = error_file_base
                            if "." not in error_file_name:
                                error_file_name = error_file_name + ext
                            src_path = os.path.join(image_dir, error_file_name)

                            if not os.path.exists(src_path):
                                src_path = os.path.join(IMAGE_BASE_DIR, sub_dir_name, error_file_name)

                        if src_path and os.path.exists(src_path):
                            dst_path = os.path.join(error_sub_dir, error_file_name)
                            try:
                                os.rename(src_path, dst_path)
                            except Exception as e:
                                print_with_time(f"Error moving file {src_path} to {dst_path}: {e}", color=Fore.RED, logger=logger, log_level=logging.ERROR)

        error_messages.clear()
        return total_pages, duration, errors, "\n".join(error_messages)

    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        print_with_time(error_msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
        if error_messages:
            error_msg = "\n".join(error_messages) + "\n" + error_msg
        return processed_pages, time.time() - start_time, errors + 1, error_msg

    finally:
        if doc is not None:
            doc.close()
        del doc
        gc.collect()




def main(ocr_results_dir=OCR_RESULTS_DIR, image_base_dir=IMAGE_BASE_DIR, output_base_dir=OUTPUT_BASE_DIRECTORY,
         y_offset=Y_OFFSET, x_offset=X_OFFSET, num_processes=NUM_PROCESSES, save_enhanced_images=SAVE_ENHANCED_IMAGES):
    """Main function to orchestrate the PDF creation process."""

    os.makedirs(output_base_dir, exist_ok=True)
    log_dir = os.path.join(output_base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    main_logger = setup_logger(log_dir, "main")

    sub_dirs_to_process = []
    overall_start_time = time.time()
    print_with_time(f"Program started.", color=Fore.CYAN, logger=main_logger)

    for sub_dir_name in os.listdir(image_base_dir):
        sub_dir_path = os.path.join(image_base_dir, sub_dir_name)
        if os.path.isdir(sub_dir_path):
            sub_dirs_to_process.append((sub_dir_name, sub_dir_path, sub_dir_path))


    print_with_time(f"Subdirectories to process: {len(sub_dirs_to_process)}", color=Fore.CYAN, logger=main_logger)

    total_image_count = 0
    for sub_dir_name, sub_dir_path, image_dir in sub_dirs_to_process:
        if ENHANCE_IMAGES:
            enhanced_dir = os.path.join(os.path.dirname(image_dir), os.path.basename(image_dir) + ENHANCED_IMAGE_SUFFIX)
            if os.path.exists(enhanced_dir):
                total_image_count += len([f for f in os.listdir(enhanced_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            else:
                total_image_count += 0
        else:
            total_image_count += len([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])


    print_with_time(f"Total images to process: {total_image_count}", color=Fore.CYAN, logger=main_logger)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_and_create_pdfs, sub_dir_name, sub_dir_path, image_dir,
                                    output_base_dir, y_offset, x_offset, save_enhanced_images,
                                    setup_logger(log_dir, f"process_{sub_dir_name}"))
                   for sub_dir_name, sub_dir_path, image_dir in sub_dirs_to_process]

        total_processed_pages = 0
        processed_pdfs = 0
        pdf_creation_start_time = time.time()

        spinner = create_spinner()

        for future in concurrent.futures.as_completed(futures):
            print_with_spinner(spinner)
            try:
                pages_in_pdf, _, _, _ = future.result()
                total_processed_pages += pages_in_pdf
                processed_pdfs += 1

                elapsed_time = time.time() - pdf_creation_start_time
                if total_processed_pages > 0:
                    avg_time_per_page = elapsed_time / total_processed_pages
                    remaining_pages = total_image_count - total_processed_pages
                    estimated_remaining_time = avg_time_per_page * remaining_pages
                    eta = datetime.now() + timedelta(seconds=estimated_remaining_time)
                    current_speed = total_processed_pages / elapsed_time if elapsed_time > 0 else 0
                else:
                    eta = "Calculating..."
                    current_speed = 0

                progress = int(50 * processed_pdfs / len(sub_dirs_to_process))
                bar = f"[{'=' * progress}{' ' * (50 - progress)}]"

                if eta == "Calculating...":
                    eta_str = f"{Fore.YELLOW}{eta}{Style.RESET_ALL}"
                else:
                    eta_str = f"{Fore.MAGENTA}{eta.strftime('%H:%M:%S')}{Style.RESET_ALL}"
                sys.stdout.write(
                    f"\r{get_timestamp()} - PDF Creation: {bar} {processed_pdfs}/{len(sub_dirs_to_process)} "
                    f"| Elapsed: {Fore.BLUE}{timedelta(seconds=int(elapsed_time))}{Style.RESET_ALL} "
                    f"| ETA: {eta_str} "
                    f"| Speed: {Fore.YELLOW}{current_speed:.2f} pages/s{Style.RESET_ALL} "
                )
                sys.stdout.flush()

            except Exception as e:
                print_with_time(f"Error in processing a subdirectory: {e}", color=Fore.RED, logger=main_logger, log_level=logging.ERROR)

    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    average_speed = total_processed_pages / overall_duration if overall_duration > 0 else 0
    print_with_time(f"\nTotal processing time: {overall_duration:.2f} seconds", color=Fore.CYAN, logger=main_logger)
    print_with_time(f"Total processed pages: {total_processed_pages}", color=Fore.CYAN, logger=main_logger)
    print_with_time(f"Average speed: {average_speed:.2f} pages/s", color=Fore.CYAN, logger=main_logger)



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_with_time("Interrupted by user. Exiting.", color=Fore.RED)