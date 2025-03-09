import paddlex as pdx
import time
import json
import os
from multiprocessing import Pool, cpu_count, get_context
import paddle
import yaml
from datetime import datetime, timedelta
import shutil
from PIL import Image
import sys  # Import sys for stdout manipulation
import logging
import pypdfium2 as pdfium  # For PDF processing
import tempfile
import numpy as np
from threading import Lock, Thread, Semaphore
import fitz  # PyMuPDF

logging.disable(logging.DEBUG)  # 关闭DEBUG日志的打印
logging.disable(logging.WARNING)  # 关闭WARNING日志的打印
# Disable Paddle's signal handler
paddle.disable_signal_handler()

# Global configuration
config_path = "/media/tmzn/DATA5/ocr_paddle/config_paddle/OCR.yaml"  # Update with your actual path


# --- Utility Functions ---

def get_beijing_time():
    """Returns the current time in Beijing (UTC+8)."""
    utc_now = datetime.utcnow()
    beijing_time = utc_now + timedelta(hours=8)
    return beijing_time.strftime("%Y-%m-%d %H:%M:%S")

def colored_output(text, color="green", log_file=None):
    """Prints colored text and logs."""
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "reset": "\033[0m",
    }
    colored_text = f"{colors.get(color, colors['reset'])}{text}{colors['reset']}"
    print(colored_text, flush=True)  # Force immediate output
    if log_file:
        with open(log_file, "a") as f:
            f.write(text + "\n")

def format_timedelta(delta):
    """Formats a timedelta object."""
    total_seconds = int(delta.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def clear_cache():
    """Clears the PaddleX cache."""
    cache_dir = os.path.expanduser("~/.paddlex/temp")
    try:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            colored_output(f"[{get_beijing_time()}] Cache cleared.", "green")
        else:
            colored_output(f"[{get_beijing_time()}] Cache directory not found.", "yellow")
    except Exception as e:
        colored_output(f"[{get_beijing_time()}] Error clearing cache: {e}", "red")

# --- Dummy Stream for Silencing Output ---

class DummyStream:
    """A dummy stream that ignores writes."""
    def write(self, *args, **kwargs):
        pass  # Do nothing

    def flush(self, *args, **kwargs):
        pass # Do nothing

# --- Context Manager for Redirecting stdout ---
class RedirectStdout:
    """Context manager for temporarily redirecting stdout."""
    def __init__(self, new_target=None):
        self.new_target = new_target or DummyStream()
        self.old_target = None

    def __enter__(self):
        self.old_target = sys.stdout
        sys.stdout = self.new_target
        return self  # Important for 'with ... as' usage

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_target
        # If new_target was a file, close it:
        if hasattr(self.new_target, 'close'):
            self.new_target.close()



# --- Multiprocessing Worker Functions ---

def init_worker(config_path, batch_size):
    """Initializes worker process."""
    global global_pipeline
    try:
        global_pipeline = pdx.create_pipeline(config_path, hpi_params={"batch_size": batch_size})
        colored_output(f"[{get_beijing_time()}] Worker process initialized (PID: {os.getpid()})", "green")
    except Exception as e:
        colored_output(f"[{get_beijing_time()}] Error initializing worker: {e}", "red")
        raise

def render_page(args):
    """
    Renders a single PDF page to an image.

    Args:
        args (tuple): Tuple containing page number, PDF path, and output folder.

    Returns:
        str: Path to the image file, or None if rendering failed.
    """
    page_num, pdf_path, output_folder = args
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        image_path = os.path.join(output_folder, f"page_{page_num + 1:04}.png")  # Format filename
        pix.save(image_path)
        doc.close()
        return image_path
    except Exception as e:
        print(f"Error rendering page {page_num + 1}: {e}")
        return None


def pdf_to_images_multiprocess(pdf_path, output_folder, num_processes=16):
    """
    Converts each page of a PDF to images using multiprocessing.

    Args:
        pdf_path (str): Path to the PDF file.
        output_folder (str): Folder to store the images.
        num_processes (int): Number of processes to use.

    Returns:
        list: List of paths to the generated image files.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists
    image_paths = []
    try:
        doc = fitz.open(pdf_path)
        num_pages = doc.page_count

        with Pool(processes=num_processes) as pool:
            # Create a list of all page rendering tasks
            tasks = []
            for page_num in range(num_pages):
                tasks.append((page_num, pdf_path, output_folder))
            # Use imap_unordered for parallel processing
            results = pool.imap_unordered(render_page, tasks)

            start_time = time.time()
            for i, result in enumerate(results):
                if result:
                    image_paths.append(result)
                # Progress display
                elapsed_time = time.time() - start_time
                progress = (i + 1) / num_pages
                speed = (i + 1) / elapsed_time if elapsed_time > 0 else 0
                print(
                    f"PDF to Image Progress: [{progress:.2%}] {i+1}/{num_pages} pages, "
                    f"Speed: {speed:.2f} pages/sec, "
                    f"Elapsed: {elapsed_time:.2f} sec",
                    end="\r"
                )
            print("\nPDF to Image conversion complete.")  # Newline after progress bar

        doc.close()
        # Sort image paths by page number to ensure correct order
        image_paths.sort(key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []  # Return an empty list on error.
    return image_paths



def process_image(image_info):
    """Processes a single image or PDF page image, handles errors."""
    global global_pipeline

    if len(image_info) != 5:
        raise ValueError(f"Invalid image_info format: {image_info}")

    if not all(isinstance(item, str) for item in image_info[:4]):
        raise TypeError(f"Invalid image_info types: {image_info}")

    image_path, output_dir, error_dir, log_file_path, is_pdf = image_info

    if global_pipeline is None:
        raise RuntimeError("Pipeline not initialized!")

    try:
        file_ext = os.path.splitext(image_path)[1].lower()
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input file not found: {image_path}")

        # Handle regular image files (including extracted PDF pages)
        # Validate image integrity
        try:
            img = Image.open(image_path)
            img.verify()
            img.close()
        except (IOError, SyntaxError) as e:
            raise Exception(f"Image validation failed: {e}")

        # Redirect stdout *during* PaddleOCR prediction
        with RedirectStdout():  # Use the context manager
            output = global_pipeline.predict(image_path)

        if output:  # Check if the output is not empty
            for res in output:
                res.save_to_json(
                    save_path=os.path.join(output_dir, f"{base_name}_result.json"),
                    indent=4,
                    ensure_ascii=False,
                )
            return True
        else:
            # Handle cases where OCR returns no results
            colored_output(f"[{get_beijing_time()}] No OCR results for {image_path}", "yellow", log_file_path)
            return False


    except Exception as e:
        colored_output(f"[{get_beijing_time()}] Error processing {image_path}: {e}", "red", log_file_path)
        try:
            error_copy_path = os.path.join(error_dir, f"FAILED_{os.path.basename(image_path)}")
            shutil.copy2(image_path, error_copy_path)
        except Exception as copy_error:
            colored_output(f"[{get_beijing_time()}] Error copying file {image_path}: {copy_error}", "red", log_file_path)
        return False



def monitor_progress(pool, total_files, log_path):
    """Displays real-time processing statistics."""
    global processed_count
    start_time = time.time()
    try:
        while True:
            time.sleep(0.5)

            # Get progress data
            current_processed = processed_count
            progress_percent = current_processed / total_files * 100
            elapsed = time.time() - start_time
            speed = current_processed / elapsed if elapsed > 0 else 0

            # Get recent logs
            log_lines = []
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    log_lines = f.readlines()[-3:]  # Last 3 log entries

            # Print to terminal with border
            output = [
                f"\n{'='*60} OCR Processing Status {'='*60}",
                f"Processed: {current_processed}/{total_files} ({progress_percent:.1f}%)",
                f"Elapsed: {format_timedelta(timedelta(seconds=elapsed))}",
                f"Speed: {speed:.2f} files/sec",
                "Recent Log Entries:",
                *[line.strip() for line in log_lines[-3:]],
                f"{'='*140}\n"
            ]
            print('\n'.join(output), flush=True)

    except KeyboardInterrupt:
        return



def main():
    global num_images, processed_count, error_count  # Declare first
    num_images = 0  # Initialize num_images
    global image_root_dir
    global error_dir
    global log_file_path
    global start_time

    image_root_dir = "/media/tmzn/DATA5/ocr_paddle/tmppdf"  #  PDF folder
    output_root_dir = "/media/tmzn/DATA5/ocr_paddle/tmpicture_ocr_results2" # OCR results folder
    log_and_error_dir = "/media/tmzn/DATA5/ocr_paddle/ocr_logs_and_errors2" # Log and error folder
    error_dir = os.path.join(log_and_error_dir, "error_images")
    log_file_path = os.path.join(log_and_error_dir, "ocr_log.txt")
    delete_temp_images = True  # Set to True to delete temp images, False to keep

    os.makedirs(output_root_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)
    os.makedirs(log_and_error_dir, exist_ok=True)

    start_time = time.time()
    start_time_str = get_beijing_time()
    colored_output(f"[{start_time_str}] OCR process started.", "green", log_file_path)
    colored_output(f"[{start_time_str}] PDF support enabled.", "green", log_file_path)

    num_processes = max(1, cpu_count() - 16) # Number of processes for PDF conversion and OCR
    batch_size = 64
    use_cpu = False  # Set to True to use CPU, False to use GPU

    if use_cpu:
        config_to_use = modify_config_for_cpu(config_path)
    else:
        config_to_use = config_path

    def pdf_path_generator(pdf_root_dir, output_root_dir, error_dir, log_file_path):
        """Generates a sequence of PDF file paths and related output directories."""
        skipped_count = 0
        for root, _, files in os.walk(pdf_root_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, pdf_root_dir)

                    # Create a separate output directory for each PDF
                    pdf_name = os.path.splitext(file)[0]
                    output_dir = os.path.join(output_root_dir, relative_path, pdf_name)
                    os.makedirs(output_dir, exist_ok=True)

                    # Check for existing results (skip if found)
                    result_file = os.path.join(output_dir, f"{pdf_name}_result.json") # Combined result file
                    if os.path.exists(result_file):
                        skipped_count += 1
                        continue

                    yield (pdf_path, output_dir, error_dir, log_file_path, True) # is_pdf = True
        if skipped_count > 0:
            colored_output(f"[{get_beijing_time()}] Skipped {skipped_count} existing files.", "yellow", log_file_path)

    pdf_generator = pdf_path_generator(image_root_dir, output_root_dir, error_dir, log_file_path)
    pdf_list = list(pdf_generator)  # Convert to list for len()
    num_pdfs = len(pdf_list)

    colored_output(f"[{get_beijing_time()}] Using {num_processes} processes.", "blue", log_file_path)
    colored_output(f"[{get_beijing_time()}] Batch size: {batch_size}", "blue", log_file_path)
    colored_output(f"[{get_beijing_time()}] Total PDFs to process: {num_pdfs}", "blue", log_file_path)

    # Use 'spawn' context for multiprocessing
    with get_context("spawn").Pool(
        processes=num_processes,
        initializer=init_worker,
        initargs=(config_to_use, batch_size),
    ) as pool:

        # Process each PDF *sequentially* but use multiprocessing for page extraction/OCR within each PDF
        processed_count = 0
        error_count = 0
        for pdf_path, output_dir, error_dir, log_file_path, _ in pdf_list:
            # 1. Convert PDF to images (using multiprocessing)
            temp_image_folder = os.path.join(output_dir, "temp_images") #  temp image folder within output_dir
            image_paths = pdf_to_images_multiprocess(pdf_path, temp_image_folder, num_processes)


            # 2. Prepare image_info for OCR processing (now operates on the temp image folder)
            image_info_list = [
                (image_path, output_dir, error_dir, log_file_path, False)  # is_pdf=False for images
                for image_path in image_paths
            ]

            # 3. Process the extracted images (using multiprocessing)
            ocr_results = pool.imap_unordered(process_image, image_info_list)

            # Gather and save OCR results, count errors
            pdf_base_name = os.path.splitext(os.path.basename(pdf_path))
            all_ocr_results = []
            for result in ocr_results:
                processed_count += 1 # Increment for each *page* processed
                if not result:
                    error_count += 1
                # We don't need to append the 'result' itself, as saving happens within process_image
                # But if process_image *returned* the dict, we would collect them here:
                # if result:  # Assuming process_image would return the result dict on success
                #    all_ocr_results.append(result)

            # 4.  (Optional) Combine results into a single JSON file (if you collected all_ocr_results)
            #     This part is *not* strictly necessary if you save individual page results.
            # with open(os.path.join(output_dir, f"{pdf_base_name}_result.json"), 'w', encoding='utf-8') as f:
            #     json.dump(all_ocr_results, f, indent=4, ensure_ascii=False)

            # 5. Delete temporary images (if desired)
            if delete_temp_images:
                try:
                    shutil.rmtree(temp_image_folder)
                    colored_output(f"[{get_beijing_time()}] Deleted temporary image folder: {temp_image_folder}", "green", log_file_path)
                except Exception as e:
                    colored_output(f"[{get_beijing_time()}] Error deleting temporary image folder: {e}", "red", log_file_path)


            # Progress reporting (after each PDF is fully processed)
            elapsed_time = time.time() - start_time
            speed = processed_count / elapsed_time if processed_count > 0 else 0  # Pages per second
            remaining_pdfs = num_pdfs - (processed_count/ (num_images if num_images > 0 else 1) )  # Correct remaining PDFs
            #  Estimate remaining time (more complex, as PDFs have varying page counts)
            #  A simple, but potentially inaccurate estimate:
            remaining_time = (num_pdfs - processed_count/ (num_images if num_images > 0 else 1) )/ (processed_count / (elapsed_time * (num_images if num_images > 0 else 1))) * elapsed_time if processed_count >0 else 0
            eta = (datetime.now() + timedelta(seconds=remaining_time)).strftime("%Y-%m-%d %H:%M:%S")
            remaining_formatted = format_timedelta(timedelta(seconds=remaining_time))
            num_images = 0
            for root, _, files in os.walk(image_root_dir):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".pdf")):
                        num_images+=1


            colored_output(
                f"[{get_beijing_time()}] Processed PDF {pdf_path}... "
                f"({processed_count}/{num_images} pages, {processed_count/num_images:.1%} complete), "
                f"Speed: {speed:.2f} pages/sec, ETA: {eta} ({remaining_formatted}), Errors: {error_count}",
                "blue", log_file_path
            )

            if processed_count % 100 == 0: #adjust 100
                colored_output(f"Checkpoint: Processed {processed_count} pages", "yellow")

    #  No need to close the main pool here, `with` statement handles it

    # Final cleanup and reporting
    end_time = time.time()
    total_time = end_time - start_time
    colored_output(f"[{get_beijing_time()}] OCR results saved to: {output_root_dir}", "green", log_file_path)
    colored_output(f"[{get_beijing_time()}] Total processing time: {total_time:.2f} seconds", "green", log_file_path)
    if num_images >0:
      colored_output(f"[{get_beijing_time()}] Average time per page: {total_time / num_images:.3f} seconds", "green", log_file_path)
    colored_output(f"[{get_beijing_time()}] Total errors: {error_count}", "red", log_file_path)

def modify_config_for_cpu(config_path):
    """Modifies config for CPU."""
    base, ext = os.path.splitext(config_path)
    new_config_path = f"{base}_cpu{ext}"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["Global"]["device"] = "cpu"
    if "use_gpu" in config["Global"]:
        del config["Global"]["use_gpu"]
    if "gpu_id" in config["Global"]:
        del config["Global"]["gpu_id"]
    with open(new_config_path, "w") as f:
        yaml.dump(config, f)
    return new_config_path

if __name__ == "__main__":
    main()