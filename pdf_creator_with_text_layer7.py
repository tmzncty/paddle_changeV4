import fitz
import json
import os
import re
import time
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
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
import psutil

# 导入find_json_parent.py的函数
from find_json_parent import find_unique_json_parent_paths

# Initialize colorama
init(autoreset=True)

# --- Configuration Variables ---
BASE_DIR = '/media/tmzn/DATA5/中国民族民间音乐集成_ocr_paddlev5'
OCR_RESULTS_DIR = BASE_DIR + '/json_output'
IMAGE_BASE_DIR = BASE_DIR + '/temp_images'
OUTPUT_BASE_DIRECTORY = BASE_DIR + '/output_pdfs_with_text_layer'
Y_OFFSET = 0  # 确保 Y_OFFSET 为 0 以进行初始排查
NUM_PROCESSES = 32  # Start with a MUCH smaller number!
SAVE_ENHANCED_IMAGES = False
ENHANCE_IMAGES = False
ENHANCED_IMAGE_SUFFIX = "_enhanced"
CHUNK_SIZE = 50  # Process images in chunks of this size.  Adjust as needed.
CREATE_DEBUG_IMAGES = False # New flag to control debug image generation
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


def get_image_and_json_paths(enhanced_paths, enhanced_image_dir, json_dir, image_file):
    """Helper function to get the correct image and JSON paths."""

    if enhanced_image_dir:  # Using enhanced images (either saved or in-memory)
        enhanced_image_data = enhanced_paths[image_file]
        page_num_match = re.search(r"page_(\d+)", image_file, re.IGNORECASE)

        if page_num_match:
            page_num = int(page_num_match.group(1))
            # 尝试多种JSON文件名格式
            possible_json_files = [
                f"page_{page_num:05}.json",      # page_00001.json
                f"page_{page_num:04}_result.json", # page_0001_result.json
                f"page_{page_num:04}.json",      # page_0001.json
                f"page_{page_num:03}.json",      # page_001.json
            ]

            json_path = None
            for json_file in possible_json_files:
                test_path = os.path.join(json_dir, json_file)
                if os.path.exists(test_path):
                    json_path = test_path
                    break

            if json_path is None:
                return None, None  # No JSON found

            # Return the enhanced image data and JSON path
            return enhanced_image_data, json_path
        else:
            # Handle filenames without "page_xxx"
            base_image_name = os.path.splitext(image_file)[0]

            # Try different JSON naming patterns
            possible_json_names = [
                f"{base_image_name}_result.json",
                f"{base_image_name}.json",
                f"{base_image_name}_ocr.json"
            ]

            json_path = None
            for json_name in possible_json_names:
                test_path = os.path.join(json_dir, json_name)
                if os.path.exists(test_path):
                    json_path = test_path
                    break

            if json_path is None:
                return None, None  # No JSON found

    else:  # Using original images
        page_num_match = re.search(r"page_(\d+)", image_file, re.IGNORECASE)
        if not page_num_match:
            return None, None  # No page number, can't proceed
        page_num = int(page_num_match.group(1))

        # Try multiple JSON file formats for original images
        possible_json_files = [
            f"page_{page_num:05}.json",      # page_00001.json
            f"page_{page_num:04}_result.json", # page_0001_result.json
            f"page_{page_num:04}.json",      # page_0001.json
            f"page_{page_num:03}.json",      # page_001.json
        ]

        json_path = None
        for json_file in possible_json_files:
            test_path = os.path.join(json_dir, json_file)
            if os.path.exists(test_path):
                json_path = test_path
                break

        if json_path is None:
            return None, None  # No JSON found

        # For original images, read directly from file system
        # Convert JSON path to corresponding image path
        # JSON: /base/json_output/category/subcategory/book/
        # Image: /base/temp_images/category/subcategory/book/
        image_dir = json_dir.replace('/json_output/', '/temp_images/')
        image_full_path = os.path.join(image_dir, image_file)

        if os.path.exists(image_full_path):
            with open(image_full_path, 'rb') as f:
                enhanced_image_data = f.read()
        else:
            return None, None  # Image file not found

    return enhanced_image_data, json_path


def process_images_in_directory(image_dir, num_processes=NUM_PROCESSES, save_enhanced=SAVE_ENHANCED_IMAGES, logger=None):
	"""Enhances images or returns original paths; handles multiprocessing."""
	image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
	if not image_files:
		print_with_time(f"No images found in {image_dir}", color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
		return {}, None

	if not ENHANCE_IMAGES:
		print_with_time("Skipping image enhancement.  Using original images.", color=Fore.YELLOW, logger=logger)
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

				# Progress reporting
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

def process_and_create_pdfs(book_name, json_book_path, image_book_path, output_base_dir, y_offset=Y_OFFSET,
                            save_enhanced=SAVE_ENHANCED_IMAGES, logger=None):
    """Processes a single book, creating PDFs in chunks."""
    start_time = time.time()
    json_dir = json_book_path
    output_pdf_name = f"{book_name}_searchable.pdf" if not save_enhanced else f"{book_name}_searchable_enhanced.pdf"
    output_pdf_path = os.path.join(output_base_dir, output_pdf_name)

    if os.path.exists(output_pdf_path):
        print_with_time(f"Skipping {book_name} (PDF already exists).", color=Fore.YELLOW, logger=logger)
        return 0, 0, 0, f"Skipped (PDF exists): {output_pdf_path}"

    enhanced_paths, enhanced_image_dir = process_images_in_directory(image_book_path, save_enhanced=save_enhanced, logger=logger)

    if enhanced_image_dir is None:
        # Get all JSON files and extract corresponding image files
        json_files = [f for f in os.listdir(json_dir) if f.lower().endswith('.json')]
        image_files = []
        for json_file in json_files:
            # Extract page number from JSON file
            page_match = re.search(r"page_(\d+)", json_file, re.IGNORECASE)
            if page_match:
                page_num = int(page_match.group(1))
                # Look for corresponding image file
                for img_ext in ['.png', '.jpg', '.jpeg']:
                    img_file = f"page_{page_num:05}{img_ext}"
                    if os.path.exists(os.path.join(image_book_path, img_file)):
                        image_files.append(img_file)
                        break
        image_files = sorted(image_files, key=lambda x: int(re.search(r"page_(\d+)", x, re.IGNORECASE).group(1)))
    else:
        def sort_key(filename):
            filename_lower = filename.lower()

            if filename_lower.startswith("cov001"):
                return (-1, filename_lower)  # Highest priority: -1
            elif filename_lower.startswith("cov002"):
                return (5, filename_lower)   # Lowest priority (after digits): 5
            elif filename_lower.startswith("bok"):
                return (0, filename_lower)
            elif filename_lower.startswith("leg"):
                return (1, filename_lower)
            elif filename_lower.startswith("fow"):
                return (2, filename_lower)
            elif filename_lower.startswith("!"):
                return (3, filename_lower)
            elif filename_lower[0].isdigit():
                match = re.match(r"^\d+", filename_lower)
                if match:
                    return (4, int(match.group(0)), filename_lower)
                else:
                    return (4, 0, filename_lower)
            else:
                return (6, filename_lower) # Even Lower priority for other files.

        image_files = natsorted(enhanced_paths.keys(), key=sort_key, alg=ns.IGNORECASE)
        
    if not image_files:
        error_message = f"No image files found in {'enhanced image dir' if enhanced_image_dir else 'json dir'}"
        print_with_time(error_message, color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
        return 0, 0, 0, error_message

    total_pages = len(image_files)
    processed_pages = 0
    errors = 0
    error_messages = []

    # Chunking logic
    for chunk_start in range(0, total_pages, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, total_pages)
        chunk_files = image_files[chunk_start:chunk_end]
        doc = None
        try:
            doc = fitz.open()  # Open a NEW document for each chunk

            for image_file in chunk_files:
                img_for_debug_obj = None # Initialize for finally block
                try:  # Inner try block
                    enhanced_image_data, json_path = get_image_and_json_paths(enhanced_paths, enhanced_image_dir, json_dir, image_file)

                    if not json_path or not enhanced_image_data:
                        errors += 1
                        msg = f"Skipping (no JSON or image): {image_file}"
                        error_messages.append(msg)
                        print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                        continue

                    if not os.path.exists(json_path):
                        errors += 1
                        msg = f"Skipping (JSON file not found): {json_path}"
                        error_messages.append(msg)
                        print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                        continue

                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    if 'dt_polys' not in data or 'rec_texts' not in data:
                        errors+=1
                        msg = f"Skipping (JSON missing data): {image_file}"
                        error_messages.append(msg)
                        print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                        continue
                    polygons = data['dt_polys']
                    texts = data['rec_texts']

                    if len(polygons) != len(texts):
                        errors += 1
                        msg = f"Skipping (polygon/text mismatch): {image_file}"
                        error_messages.append(msg)
                        print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                        continue

                    # === START DEBUG VISUALIZATION SETUP ===
                    if CREATE_DEBUG_IMAGES:
                        debug_graphics_output_dir = os.path.join(output_base_dir, "debug_images", book_name)
                        os.makedirs(debug_graphics_output_dir, exist_ok=True)
                        
                        # Try to extract page number for a more specific debug filename
                        page_num_match_debug = re.search(r"page_(\d+)", image_file, re.IGNORECASE)
                        if page_num_match_debug:
                            debug_filename_prefix = f"page_{page_num_match_debug.group(1)}_"
                        else:
                            # Fallback for filenames not matching "page_XXX"
                            debug_filename_prefix = f"{os.path.splitext(image_file)[0]}_"
                        
                        debug_image_save_path = os.path.join(debug_graphics_output_dir, f"debug_{debug_filename_prefix}{os.path.splitext(image_file)[1]}.png")

                        if isinstance(enhanced_image_data, str): # Path
                            img_for_debug_obj = Image.open(enhanced_image_data).convert("RGB")
                        else: # Bytes
                            img_for_debug_obj = Image.open(io.BytesIO(enhanced_image_data)).convert("RGB")
                        
                        current_img_width_for_debug, current_img_height_for_debug = img_for_debug_obj.size # Use different var names to avoid conflict
                        draw_on_img = ImageDraw.Draw(img_for_debug_obj)
                    # === END DEBUG VISUALIZATION SETUP ===

                    # Dimensions are taken from img_for_debug_obj to avoid reopening
                    # Get image dimensions for PDF page creation
                    if isinstance(enhanced_image_data, str):
                        with Image.open(enhanced_image_data) as temp_img_for_size:
                            current_img_width, current_img_height = temp_img_for_size.size
                    else:
                        with Image.open(io.BytesIO(enhanced_image_data)) as temp_img_for_size:
                            current_img_width, current_img_height = temp_img_for_size.size

                    page = doc.new_page(width=current_img_width, height=current_img_height)
                    if isinstance(enhanced_image_data, str):  # It's a path
                        page.insert_image(page.rect, filename=enhanced_image_data)
                    else:  # It's image data (bytes)
                        page.insert_image(page.rect, stream=enhanced_image_data)

                    x_scale = 1.0
                    y_scale = 1.0

                    # --- Start: Reorder OCR elements for better column handling ---
                    combined_ocr_elements = []
                    for i_poly, polygon_coords_from_json_orig in enumerate(polygons):
                        text_orig = texts[i_poly]
                        if not text_orig.strip():
                            continue
                        
                        # Calculate min_x and min_y for sorting
                        # More robust than just polygon_coords_from_json_orig[0][0] or [0][1]
                        temp_x_coords = [p[0] for p in polygon_coords_from_json_orig]
                        temp_y_coords = [p[1] for p in polygon_coords_from_json_orig]
                        min_x = min(temp_x_coords) if temp_x_coords else 0
                        min_y = min(temp_y_coords) if temp_y_coords else 0

                        combined_ocr_elements.append({
                            "polygon": polygon_coords_from_json_orig,
                            "text": text_orig,
                            "min_x": min_x,
                            "min_y": min_y
                        })

                    page_center_x = current_img_width / 2.0

                    def column_sort_key(element):
                        is_left_column = element["min_x"] < page_center_x
                        # Sort order: (is_right_column, y_coordinate)
                        # False (left column) comes before True (right column)
                        return (not is_left_column, element["min_y"])

                    sorted_ocr_elements = sorted(combined_ocr_elements, key=column_sort_key)
                    # --- End: Reorder OCR elements ---

                    # Iterate through the reordered elements
                    for element_data in sorted_ocr_elements:
                        polygon_coords_from_json = element_data["polygon"]
                        text = element_data["text"]
                        # y_offset is from the main function arguments

                        # === DRAWING ON DEBUG IMAGE ===
                        if CREATE_DEBUG_IMAGES and img_for_debug_obj and draw_on_img: # Ensure objects exist
                            # 1. Original polygon (from JSON, blue)
                            original_poly_tuples = [tuple(p) for p in polygon_coords_from_json]
                            draw_on_img.polygon(original_poly_tuples, outline="blue", width=1)

                            # Calculate scaled_polygon (same as used for PDF rect)
                            # y_offset comes from the function arguments (process_and_create_pdfs)
                            scaled_polygon_for_viz = [
                                [int(p[0] * x_scale), int(p[1] * y_scale) + y_offset]
                                for p in polygon_coords_from_json
                            ]
                            # 2. Scaled polygon (green)
                            scaled_poly_tuples_for_viz = [tuple(p) for p in scaled_polygon_for_viz]
                            draw_on_img.polygon(scaled_poly_tuples_for_viz, outline="green", width=1)
                            
                            # This is the 'rect' that will be used by PyMuPDF for text insertion
                            rect_for_pdf_insertion_viz = fitz.Rect(scaled_polygon_for_viz[0][0], scaled_polygon_for_viz[0][1],
                                                               scaled_polygon_for_viz[2][0], scaled_polygon_for_viz[2][1])

                            # 3. PyMuPDF's fitz.Rect (red)
                            draw_on_img.rectangle(
                                (rect_for_pdf_insertion_viz.x0, rect_for_pdf_insertion_viz.y0, rect_for_pdf_insertion_viz.x1, rect_for_pdf_insertion_viz.y1),
                                outline="red",
                                width=2
                            )
                            # Mark the top-left insertion point (red dot)
                            draw_on_img.ellipse(
                                (rect_for_pdf_insertion_viz.tl.x - 2, rect_for_pdf_insertion_viz.tl.y - 2, rect_for_pdf_insertion_viz.tl.x + 2, rect_for_pdf_insertion_viz.tl.y + 2),
                                fill="red", outline="red"
                            )
                        # === END DRAWING ===

                        # PDF Text insertion logic (uses rect_for_pdf_insertion, which is the 'rect' variable)
                        # The actual rect_for_pdf_insertion must be defined outside the CREATE_DEBUG_IMAGES block
                        # as it's used for text insertion regardless of debug mode.
                        current_scaled_polygon = [
                            [int(p[0] * x_scale), int(p[1] * y_scale) + y_offset]
                            for p in polygon_coords_from_json
                        ]
                        rect_for_pdf_insertion = fitz.Rect(current_scaled_polygon[0][0], current_scaled_polygon[0][1],
                                                       current_scaled_polygon[2][0], current_scaled_polygon[2][1])

                        fontsize = rect_for_pdf_insertion.height * 0.9
                        while True:
                            text_width = fitz.get_text_length(text, fontname="china-s", fontsize=fontsize)
                            if text_width <= rect_for_pdf_insertion.width or fontsize <= 1:
                                break
                            fontsize -= 1
                        fontsize = max(1, min(fontsize, 100))

                        # Adjust insertion point for baseline alignment
                        # Shift baseline down from the top of the red box by a factor of the fontsize
                        # This aims to place the top of the text near the top of the red box.
                        adjusted_insertion_y = rect_for_pdf_insertion.top_left.y + (fontsize * 0.85) # Try 85% of fontsize
                        new_insertion_point = fitz.Point(rect_for_pdf_insertion.top_left.x, adjusted_insertion_y)

                        page.insert_text(new_insertion_point, text, fontname="china-s", fontsize=fontsize,
                                            color=(0, 0, 0),      # Make text black - Color is ignored by render_mode=3 but kept for consistency
                                            fill=(1, 1, 1),      # Keep white background for text cell (optional) - Also ignored by render_mode=3
                                            render_mode=3)       # Render mode 3 for invisible, searchable text

                    processed_pages += 1
                    
                    # === SAVE DEBUG IMAGE (after all polygons for this image_file are processed) ===
                    if CREATE_DEBUG_IMAGES and img_for_debug_obj: # Ensure it was created
                        try:
                            img_for_debug_obj.save(debug_image_save_path)
                            # logger.debug(f"Debug image saved: {debug_image_save_path}") # Optional: log debug image saving
                        except Exception as e_save_debug:
                            print_with_time(f"Error saving debug image {debug_image_save_path}: {e_save_debug}", color=Fore.RED, logger=logger, log_level=logging.ERROR)
                    # === END SAVE DEBUG IMAGE ===

                    del polygons
                    del texts
                    del data
                    gc.collect()

                except Exception as e:  # Catch errors *within* the image processing loop
                    errors += 1
                    msg = f"Error processing JSON/image {image_file}: {e}"
                    error_messages.append(msg)
                    print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                finally: # Inner try's finally
                    if CREATE_DEBUG_IMAGES and img_for_debug_obj: # Only close if it was created
                        img_for_debug_obj.close() # Close the image used for drawing
                
                # Clean up enhanced_image_data if it was a byte stream, after debug image is handled
                if not isinstance(enhanced_image_data, str) and 'enhanced_image_data' in locals() and enhanced_image_data is not None:
                    del enhanced_image_data
                    gc.collect()


            # --- PDF Saving (After each chunk) ---
            try:
                if doc is not None:  # Check if doc was created
                    #  Create intermediate output directory
                    intermediate_output_dir = os.path.join(output_base_dir, "intermediate")
                    os.makedirs(intermediate_output_dir, exist_ok=True)
                    intermediate_pdf_path = os.path.join(intermediate_output_dir, f"{book_name}_chunk_{chunk_start // CHUNK_SIZE}.pdf")
                    doc.save(intermediate_pdf_path, garbage=4, deflate=True)
                    print_with_time(f"Intermediate PDF chunk saved: {intermediate_pdf_path}", color=Fore.GREEN, logger=logger)


            except Exception as e:
                errors += 1
                error_msg = f"Error saving intermediate PDF: {e}"
                error_messages.append(error_msg)
                print_with_time(error_msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)

            finally:  # Close the document *after each chunk*
                if doc is not None:
                    doc.close()
                del doc  # Explicitly delete
                gc.collect()


        except Exception as e:
            error_msg = f"An unexpected error occurred during chunk processing: {e}"
            print_with_time(error_msg, color=Fore.RED, logger=logger,log_level=logging.ERROR)
            if error_messages:
                error_msg = "\n".join(error_messages) + "\n" + error_msg
            return processed_pages, time.time() - start_time, errors + 1, error_msg

    # --- Combine Intermediate PDFs (After all chunks are processed) ---
    try:
        final_doc = fitz.open()
        intermediate_dir = os.path.join(output_base_dir, "intermediate")
        if os.path.exists(intermediate_dir):
            intermediate_files = sorted([f for f in os.listdir(intermediate_dir) if f.startswith(f"{book_name}_chunk_") and f.endswith(".pdf")],
                                        key=lambda x: int(x.split("_")[-1].split(".")[0])) #Sort the pdf files.

            for intermediate_file in intermediate_files:
                intermediate_path = os.path.join(intermediate_dir, intermediate_file)
                try:
                    with fitz.open(intermediate_path) as intermediate_doc:
                        final_doc.insert_pdf(intermediate_doc)
                    # Delete intermediate file after successful insertion
                    os.remove(intermediate_path)
                except Exception as e:
                    errors += 1
                    error_msg = f"Error inserting intermediate PDF {intermediate_file}: {e}"
                    error_messages.append(error_msg)
                    print_with_time(error_msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)

            # Remove the intermediate directory if it's empty
            if not os.listdir(intermediate_dir):
                os.rmdir(intermediate_dir)

        final_doc.save(output_pdf_path, garbage=4, deflate=True)
        end_time = time.time()
        duration = end_time-start_time
        print_with_time(f"Final PDF created: {output_pdf_path} in {duration:.2f} seconds", color=Fore.GREEN, logger=logger)

    except Exception as e:
        errors += 1
        error_msg = f"Error combining or saving the final PDF: {e}"
        error_messages.append(error_msg)
        print_with_time(error_msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
    finally:
        if final_doc is not None:
            final_doc.close()
        del final_doc
        gc.collect()


    # --- Error Handling and File Moving (After saving) ---
    # Log file writing (using logger)
    logger.info(f"Processed: {processed_pages} pages")
    duration = time.time() - start_time  # Calculate duration
    logger.info(f"Total Duration: {duration:.2f} seconds")
    logger.info(f"PDF Creation Errors: {errors}")
    for msg in error_messages:
        logger.error(msg)

    # Move Error Files
    if save_enhanced and errors > 0:
        error_sub_dir = os.path.join(output_base_dir, "errors", book_name)
        os.makedirs(error_sub_dir, exist_ok=True)
        for error_line in error_messages:
            match = re.search(r"(page_\d+)", error_line)
            if match:
                error_file_base = match.group(1)
                # Try multiple file extensions and JSON formats
                for ext in ['.png', '.jpg', '.jpeg']:
                    error_file_name = error_file_base + ext
                    if save_enhanced:
                        src_path = os.path.join(image_book_path + ENHANCED_IMAGE_SUFFIX, error_file_name)
                    else:
                        src_path = os.path.join(image_book_path, error_file_name)

                    if not os.path.exists(src_path):
                        src_path = os.path.join(image_book_path, error_file_name)

                    if not os.path.exists(src_path):
                        src_path = os.path.join(json_dir, error_file_name)

                # Also try JSON files with different formats
                for json_ext in ['.json', '_result.json', '_ocr.json']:
                    json_file_name = error_file_base + json_ext
                    src_path = os.path.join(json_dir, json_file_name)

                    if os.path.exists(src_path):
                        dst_path = os.path.join(error_sub_dir, error_file_name)
                        try:
                            os.rename(src_path, dst_path)
                        except Exception as e:
                            print_with_time(f"Error moving file {src_path} to {dst_path}: {e}", color=Fore.RED,
                                            logger=logger, log_level=logging.ERROR)
    error_messages.clear()
    del enhanced_paths
    gc.collect()
    return total_pages, duration, errors, "\n".join(error_messages)



def main(ocr_results_dir=OCR_RESULTS_DIR, image_base_dir=IMAGE_BASE_DIR, output_base_dir=OUTPUT_BASE_DIRECTORY,
         y_offset=Y_OFFSET, num_processes=NUM_PROCESSES, save_enhanced_images=SAVE_ENHANCED_IMAGES):
    """Main function to orchestrate the PDF creation process."""

    os.makedirs(output_base_dir, exist_ok=True)
    log_dir = os.path.join(output_base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    main_logger = setup_logger(log_dir, "main")

    books_to_process = [] # 存储待处理的书籍信息元组 (书名, JSON路径, 图像路径)
    overall_start_time = time.time()
    print_with_time(f"Program started.", color=Fore.CYAN, logger=main_logger)

    # 使用find_json_parent.py的函数来查找所有书籍
    print_with_time("扫描所有JSON文件夹（书籍）...", color=Fore.CYAN, logger=main_logger)
    print_with_time(f"搜索目录: {BASE_DIR}", color=Fore.CYAN, logger=main_logger)
    print_with_time(f"目录是否存在: {os.path.exists(BASE_DIR)}", color=Fore.CYAN, logger=main_logger)

    # 直接实现find_json_parent.py的逻辑
    json_parent_paths = set()
    for root, _, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith(".json"):
                json_parent_paths.add(root)

    print_with_time(f"找到 {len(json_parent_paths)} 本书籍", color=Fore.GREEN, logger=main_logger)

    # 调试：显示前几个路径
    if len(json_parent_paths) > 0:
        print_with_time("前3个路径:", color=Fore.CYAN, logger=main_logger)
        for i, path in enumerate(list(json_parent_paths)[:3]):
            print_with_time(f"  {i+1}: {path}", color=Fore.CYAN, logger=main_logger)

    # 处理每本书，保持相对路径结构
    for json_book_path in json_parent_paths:
        book_name = os.path.basename(json_book_path)

        # 获取相对于BASE_DIR的路径
        rel_path = os.path.relpath(json_book_path, BASE_DIR)

        # 根据实际情况处理路径
        if rel_path.startswith('json_output/'):
            # JSON在json_output目录，图像在temp_images目录
            image_rel_path = rel_path.replace('json_output/', 'temp_images/', 1)
            image_book_path = os.path.join(BASE_DIR, image_rel_path)
        elif rel_path.startswith('temp_images/'):
            # JSON在temp_images目录，图像也在temp_images目录
            image_book_path = json_book_path
        else:
            print_with_time(f"警告: JSON路径格式异常 {json_book_path}", color=Fore.YELLOW, logger=main_logger)
            continue

        # 检查图像路径是否存在
        if os.path.exists(image_book_path):
            # 保持相对路径结构来存储PDF
            # 获取相对于BASE_DIR的路径
            rel_path = os.path.relpath(json_book_path, BASE_DIR)
            if rel_path.startswith('json_output/'):
                # 将json_output替换为output_pdfs_with_text_layer，保持后续路径结构
                pdf_rel_path = rel_path.replace('json_output/', 'output_pdfs_with_text_layer/', 1)
                output_pdf_dir = os.path.join(BASE_DIR, pdf_rel_path)
                output_pdf_name = f"{book_name}_searchable.pdf"
                output_pdf_path = os.path.join(output_pdf_dir, output_pdf_name)
            else:
                # 备用方案：如果路径格式异常，使用原来的方式
                output_pdf_name = f"{book_name}_searchable.pdf"
                output_pdf_path = os.path.join(output_base_dir, output_pdf_name)

            if not os.path.exists(output_pdf_path):
                # 确保输出目录存在
                os.makedirs(output_pdf_dir, exist_ok=True)
                books_to_process.append((book_name, json_book_path, image_book_path, output_pdf_dir))
                print_with_time(f"添加书籍: {book_name}", color=Fore.GREEN, logger=main_logger)
            else:
                print_with_time(f"跳过 {book_name} (PDF已存在)", color=Fore.YELLOW, logger=main_logger)
        else:
            print_with_time(f"警告: 图像目录不存在 {image_book_path}", color=Fore.YELLOW, logger=main_logger)

    print_with_time(f"待处理的书籍数量: {len(books_to_process)}", color=Fore.CYAN, logger=main_logger)

    # --- Calculate Total Image Count ---
    total_image_count = 0
    for book_name, json_book_path, image_book_path, book_output_dir in books_to_process:
        if ENHANCE_IMAGES:
            # Count images in the enhanced directory if enhancing
            enhanced_dir = os.path.join(os.path.dirname(image_book_path), os.path.basename(image_book_path) + ENHANCED_IMAGE_SUFFIX)
            if os.path.exists(enhanced_dir):  # Check if enhanced dir exists
                total_image_count += len([f for f in os.listdir(enhanced_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            else:  # if enhanced_dir not exists, count 0 for it.
                total_image_count += 0
        else:
            # Count images directly in the image directory if not enhancing.
            total_image_count += len([f for f in os.listdir(image_book_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print_with_time(f"预计待处理的总图像数量: {total_image_count}", color=Fore.CYAN, logger=main_logger)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_and_create_pdfs, book_name, json_book_path, image_book_path,
                                    book_output_dir, y_offset, save_enhanced_images,
                                    setup_logger(log_dir, f"process_{book_name}"))
                   for book_name, json_book_path, image_book_path, book_output_dir in books_to_process]

        total_processed_pages = 0
        processed_pdfs = 0  # Keep track of completed PDFs for progress reporting
        pdf_creation_start_time = time.time()  # Keep track the start time

        spinner = create_spinner()

        for future in concurrent.futures.as_completed(futures):
            print_with_spinner(spinner)
            try:
                pages_in_pdf, _, _, _ = future.result()  # Get pages from the result
                total_processed_pages += pages_in_pdf  # Accumulate proccessed pages
                processed_pdfs += 1  # Count pdfs

                # --- Improved ETA Calculation ---
                elapsed_time = time.time() - pdf_creation_start_time
                if total_processed_pages > 0:  # Avoid division by zero
                    avg_time_per_page = elapsed_time / total_processed_pages
                    remaining_pages = total_image_count - total_processed_pages
                    estimated_remaining_time = avg_time_per_page * remaining_pages
                    eta = datetime.now() + timedelta(seconds=estimated_remaining_time)
                    current_speed = total_processed_pages / elapsed_time if elapsed_time > 0 else 0
                else:
                    eta = "Calculating..."  # If no page proccessed, show "Calculating..."
                    current_speed = 0

                # Progress Bar (based on PDFs, but ETA is based on pages)
                progress = int(50 * processed_pdfs / len(books_to_process))
                bar = f"[{'=' * progress}{' ' * (50 - progress)}]"

                # Use different formatting for "Calculating..."
                if eta == "Calculating...":
                    eta_str = f"{Fore.YELLOW}{eta}{Style.RESET_ALL}"
                else:
                    eta_str = f"{Fore.MAGENTA}{eta.strftime('%H:%M:%S')}{Style.RESET_ALL}"
                sys.stdout.write(

                    f"\r{get_timestamp()} - PDF Creation: {bar} {processed_pdfs}/{len(books_to_process)} "
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
