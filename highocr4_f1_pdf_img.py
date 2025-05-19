import paddlex as pdx
import time
import json
import os
from multiprocessing import Pool, cpu_count, get_context, Process, Manager # Manager for non-daemon workaround if needed
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
import types # <--- ADDED IMPORT

# =====================================================================================
# 用户可配置参数 (User Configurable Parameters)
# =====================================================================================

# PaddleX缓存主目录 (PaddleX Cache Home Directory)
# 脚本会尝试通过设置 PADDLEX_HOME 环境变量来引导 PaddleX 使用此路径。
# !! 强烈建议: 如果 PaddleX 仍然将临时文件写入默认的 ~/.paddlex/temp/ 并导致磁盘空间不足，
# !! 请手动将 ~/.paddlex/temp 符号链接到此 PADDLEX_HOME_OVERRIDE 下的 temp 目录。
# !! 例如: ln -s /media/tmzn/DATA5/paddlex_cache_home/temp ~/.paddlex/temp
PADDLEX_HOME_OVERRIDE = "/media/tmzn/DATA5/paddlex_cache_home"  # 例如: "/mnt/large_disk/paddlex_cache"

# PaddleX配置文件路径 (PaddleX Configuration File Path)
CONFIG_PATH_PADDLE = "/media/tmzn/DATA5/ocr_paddle/config_paddle/OCR.yaml"

# 输入源配置 (Input Sources Configuration)
# 列表中的每个字典代表一个输入源。
# 'path': 输入文件夹的绝对路径。
# 'type': 'pdf' 或 'image'。
INPUT_SOURCES = [
    {"path": "/media/tmzn/DATA5/ocr_paddle/词典pdf", "type": "pdf"},  # PDF文件夹路径
    {"path": "/media/tmzn/DATA4/splitdict/汉语", "type": "image"}     # 图片文件夹路径
]

# OCR结果统一输出根目录 (Root Output Directory for OCR Results)
OUTPUT_ROOT_DIR = "/media/tmzn/DATA5/ocr_paddle/词典pdf_ocr_result"

# 日志和错误图片存放目录 (Directory for Logs and Error Images)
# 注意：如果修改此路径，请确保 error_dir 和 log_file_path 也相应更新或基于此动态生成。
LOG_AND_ERROR_DIR_BASE = "/media/tmzn/DATA5/ocr_paddle/词典pdf_ocr_result_log"

# 是否删除PDF转换产生的临时图片 (Delete Temporary Images Generated from PDF Conversion)
# 'True' 表示删除，'False' 表示保留。这些图片是指从PDF页面转换得到的PNG文件，
# 通常位于类似 'OUTPUT_ROOT_DIR/.../pdf_name/temp_images_from_pdf/' 的路径下。
# 此设置不影响原始输入图片文件夹中的文件。
DELETE_TEMP_IMAGES_AFTER_PDF_PROCESSING = False

# 每个PDF渲染时的并发进程数 (Number of Concurrent Processes for Rendering Pages within a Single PDF)
# 根据CPU核心数调整，例如：cpu_count() // 2 或一个固定值。用户已设置为24。
# 注意：在当前版本中，由于prepare_single_pdf_for_ocr内部页面渲染改为串行，此参数不再直接用于创建子Pool，
# 但保留以备将来可能恢复嵌套并行。render_page函数本身仍然是为并行设计的。
NUM_RENDER_PROCESSES_PER_PDF = 24 # 当前版本中，此值在prepare_single_pdf_for_ocr中未直接用于创建Pool

# 主OCR任务的并发进程数 (Number of Concurrent Processes for Main OCR Tasks)
# 用户已固定为16 (User has fixed this to 16)
NUM_OCR_PROCESSES = 16

# PaddleX OCR模型的批处理大小 (Batch Size for PaddleX OCR Model)
# 尝试增大此值（如 48, 64）可能提高GPU利用率，但需注意显存。
OCR_BATCH_SIZE = 32

# 新增：并行准备PDF文件的进程数 (Number of Concurrent Processes for Preparing PDF files)
# 这个值决定了同时有多少个PDF文件可以被并行地进行页面提取和待OCR列表的生成。
# 根据CPU核心数和I/O能力调整。例如：cpu_count() // 2 或一个固定值如 4 或 8。
# 如果PDF文件不多，或者主要瓶颈在后续的OCR，此值不宜过大。
NUM_CONCURRENT_PDF_PREP_PROCESSES = 8 # 示例值，用户可以根据情况调整

# 是否强制使用CPU进行OCR (Force Use CPU for OCR)
# 如果为 True，脚本会尝试修改PaddleX配置以使用CPU。
USE_CPU_FOR_OCR = False

# =====================================================================================
# 全局变量和初始化 (Global Variables and Initialization)
# =====================================================================================

# 1. 尽早设置 PADDLEX_HOME 环境变量并创建目标目录 (Set PADDLEX_HOME environment variable and create target dirs early)
if PADDLEX_HOME_OVERRIDE:
    os.environ['PADDLEX_HOME'] = PADDLEX_HOME_OVERRIDE
    print(f"[信息 主程序早期] PADDLEX_HOME 环境变量已设置为: {os.environ.get('PADDLEX_HOME')}")
    try:
        os.makedirs(PADDLEX_HOME_OVERRIDE, exist_ok=True)
        os.makedirs(os.path.join(PADDLEX_HOME_OVERRIDE, "temp"), exist_ok=True)
        print(f"[信息 主程序早期] 已确保PADDLEX_HOME覆盖目录存在: {PADDLEX_HOME_OVERRIDE}/temp")
    except Exception as e:
        print(f"[错误 主程序早期] 创建PADDLEX_HOME覆盖目录失败: {e}")
else:
    print("[信息 主程序早期]未设置 PADDLEX_HOME_OVERRIDE。PaddleX可能使用其默认主路径 (~/.paddlex)。")

logging.disable(logging.DEBUG)  # 关闭DEBUG日志的打印
logging.disable(logging.WARNING)  # 关闭WARNING日志的打印
# Disable Paddle's signal handler
paddle.disable_signal_handler()

# 全局配置文件路径变量 (Global config path variable, now derived from top section)
config_path = CONFIG_PATH_PADDLE

# --- Utility Functions ---

def get_paddlex_temp_dir():
    """Determines the PaddleX temporary directory based on PADDLEX_HOME environment variable or default."""
    # Since pdx.env.get_paddlex_home() is not available in the user's PaddleX version,
    # we rely solely on the PADDLEX_HOME environment variable or the default path.
    paddlex_home_env = os.environ.get('PADDLEX_HOME')
    # print(f"[DEBUG get_paddlex_temp_dir] os.environ.get('PADDLEX_HOME') returned: {paddlex_home_env}") # Optional debug
    if paddlex_home_env:
        return os.path.join(paddlex_home_env, "temp")
    else:
        default_path = os.path.expanduser("~/.paddlex/temp")
        # print(f"[DEBUG get_paddlex_temp_dir] Falling back to default path: {default_path}") # Optional debug
        return default_path

def get_beijing_time():
    """Returns the current time in Beijing (UTC+8)."""
    utc_now = datetime.utcnow()
    beijing_time = utc_now + timedelta(hours=8)
    return beijing_time.strftime("%Y-%m-%d %H:%M:%S")

def colored_output(text, color="green", log_file=None, stdout_too=True):
    """Prints colored text and logs."""
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "reset": "\033[0m",
    }
    colored_text = f"{colors.get(color, colors['reset'])}{text}{colors['reset']}"
    if stdout_too:
        print(colored_text, flush=True)  # Force immediate output
    if log_file:
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(text + "\n")

def format_timedelta(delta):
    """Formats a timedelta object."""
    total_seconds = int(delta.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def clear_cache():
    """Clears the PaddleX cache.
    Manages the PADDLEX_HOME_OVERRIDE specified temp directory and the default ~/.paddlex/temp.
    Prioritizes using the PADDLEX_HOME_OVERRIDE path and aims to preserve a symlink
    from ~/.paddlex/temp to the override path if present.
    """
    current_log_file = globals().get('log_file_path')

    # Define the primary target directory from PADDLEX_HOME_OVERRIDE
    override_target_dir = None
    # PADDLEX_HOME_OVERRIDE is a global variable defined at the top of the script
    # os.environ['PADDLEX_HOME'] should have been set to PADDLEX_HOME_OVERRIDE if it's defined
    paddlex_home_env_val = os.environ.get('PADDLEX_HOME')

    if paddlex_home_env_val: # This should be the value of PADDLEX_HOME_OVERRIDE
        override_target_dir = os.path.join(paddlex_home_env_val, "temp")
        colored_output(f"[{get_beijing_time()}] 正在确保PADDLEX_HOME目标目录: {override_target_dir}", "blue", current_log_file)
        try:
            if os.path.islink(override_target_dir):
                # This is unusual for the actual target, PADDLEX_HOME_OVERRIDE/temp should be a real directory.
                colored_output(f"[{get_beijing_time()}] 警告: PADDLEX_HOME目标 {override_target_dir} 是一个符号链接。将移除链接并创建为目录。", "yellow", current_log_file)
                os.unlink(override_target_dir)
            elif os.path.isdir(override_target_dir):
                shutil.rmtree(override_target_dir)
            elif os.path.exists(override_target_dir): # It's a file, not a dir or link
                os.remove(override_target_dir)
            
            os.makedirs(override_target_dir, exist_ok=True)
            colored_output(f"[{get_beijing_time()}] PADDLEX_HOME目标目录 {override_target_dir} 已(重新)创建成功。", "green", current_log_file)
        except Exception as e:
            colored_output(f"[{get_beijing_time()}] 管理PADDLEX_HOME目标 {override_target_dir} 时出错: {e}。尝试确保其存在。", "red", current_log_file)
            try:
                os.makedirs(override_target_dir, exist_ok=True) # Best effort to ensure it exists for PaddleX
            except Exception as e_mkdir:
                colored_output(f"[{get_beijing_time()}] 严重错误: 创建PADDLEX_HOME目标 {override_target_dir} 失败: {e_mkdir}", "red", current_log_file)
    else:
        colored_output(f"[{get_beijing_time()}] 未设置PADDLEX_HOME_OVERRIDE。将主要使用PaddleX的默认路径。", "yellow", current_log_file)

    # Define the default PaddleX temporary directory
    default_paddlex_temp_dir = os.path.expanduser("~/.paddlex/temp")

    # Check if the default directory is correctly symlinked to the override target
    is_symlinked_correctly_to_override = False
    if override_target_dir and \
       os.path.islink(default_paddlex_temp_dir) and \
       os.path.realpath(default_paddlex_temp_dir) == os.path.realpath(override_target_dir):
        is_symlinked_correctly_to_override = True

    if is_symlinked_correctly_to_override:
        colored_output(f"[{get_beijing_time()}] 默认路径 {default_paddlex_temp_dir} 已正确符号链接到PADDLEX_HOME目标 {override_target_dir}。此路径本身无需操作。", "green", current_log_file)
        # Ensure the symlink is not broken (target 'override_target_dir' should exist due to handling above)
        if not os.path.exists(os.readlink(default_paddlex_temp_dir)): # Symlink exists but target is broken
             colored_output(f"[{get_beijing_time()}] 警告: Symlink {default_paddlex_temp_dir} target does not exist. Attempting to recreate.", "yellow", current_log_file)
             try:
                 os.unlink(default_paddlex_temp_dir) # Remove broken symlink
                 os.symlink(override_target_dir, default_paddlex_temp_dir, target_is_directory=True) # Recreate
                 colored_output(f"[{get_beijing_time()}] Symlink {default_paddlex_temp_dir} recreated to point to {override_target_dir}.", "green", current_log_file)
             except Exception as e_relink:
                 colored_output(f"[{get_beijing_time()}] Error recreating symlink {default_paddlex_temp_dir}: {e_relink}", "red", current_log_file)
    else:
        # Default path is not symlinked to override, or PADDLEX_HOME_OVERRIDE was not set.
        # We need to manage default_paddlex_temp_dir directly.
        colored_output(f"[{get_beijing_time()}] 正在管理默认的PaddleX临时目录: {default_paddlex_temp_dir}", "blue", current_log_file)
        if override_target_dir: # PADDLEX_HOME_OVERRIDE was set, but symlink is missing or incorrect
            colored_output(f"[{get_beijing_time()}] 警告: 默认路径 {default_paddlex_temp_dir} 未正确符号链接到PADDLEX_HOME目标 {override_target_dir}。", "yellow", current_log_file)
            colored_output(f"[{get_beijing_time()}] 为获得最佳效果，请手动创建此符号链接: ln -sfn {override_target_dir} {default_paddlex_temp_dir}", "yellow", current_log_file)
        
        try:
            if os.path.islink(default_paddlex_temp_dir):
                # It's a symlink, but not to our override_target_dir (that case handled above).
                colored_output(f"[{get_beijing_time()}] {default_paddlex_temp_dir} 是一个意外的符号链接。将移除并重新创建为目录。", "yellow", current_log_file)
                os.unlink(default_paddlex_temp_dir) # Remove incorrect symlink
                os.makedirs(default_paddlex_temp_dir, exist_ok=True)
            elif os.path.isdir(default_paddlex_temp_dir):
                # It's a directory. THIS IS WHERE THE HANG LIKELY OCCURRED.
                colored_output(f"[{get_beijing_time()}] 尝试清理目录 {default_paddlex_temp_dir}。", "blue", current_log_file)
                colored_output(f"[{get_beijing_time()}] !!! 重要提示 !!! 如果脚本在此处挂起或非常缓慢，目录 {default_paddlex_temp_dir} 可能存在问题 (例如，文件过多、权限问题或文件被锁定)。", "yellow", current_log_file)
                colored_output(f"[{get_beijing_time()}] 如果挂起，您可能需要手动中断 (Ctrl+C)，然后在终端中清理此目录 (例如，'sudo rm -rf {default_paddlex_temp_dir}') 后再重新运行。", "yellow", current_log_file)
                shutil.rmtree(default_paddlex_temp_dir) # Potentially hanging call
                os.makedirs(default_paddlex_temp_dir, exist_ok=True)
            elif os.path.exists(default_paddlex_temp_dir): # It's a file, not a dir or link
                os.remove(default_paddlex_temp_dir)
                os.makedirs(default_paddlex_temp_dir, exist_ok=True)
            else: # Does not exist
                os.makedirs(default_paddlex_temp_dir, exist_ok=True)
            colored_output(f"[{get_beijing_time()}] 默认PaddleX临时目录 {default_paddlex_temp_dir} 已(重新)创建成功。", "green", current_log_file)
        except KeyboardInterrupt:
            colored_output(f"[{get_beijing_time()}] 在管理 {default_paddlex_temp_dir} 期间收到KeyboardInterrupt。脚本将终止。", "red", current_log_file)
            raise # Re-raise to terminate script
        except Exception as e:
            colored_output(f"[{get_beijing_time()}] 管理默认路径 {default_paddlex_temp_dir} 时出错: {e}。尝试确保其存在。", "red", current_log_file)
            try:
                os.makedirs(default_paddlex_temp_dir, exist_ok=True) # Best effort
            except Exception as e_mkdir_default:
                colored_output(f"[{get_beijing_time()}] 严重错误: 创建默认PaddleX临时目录 {default_paddlex_temp_dir} 失败: {e_mkdir_default}", "red", current_log_file)

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

def init_worker(paddlex_config_path, batch_size_ocr, worker_paddlex_home_override):
    """Initializes worker process.
    Ensures PADDLEX_HOME is set correctly within the worker.
    """
    global global_pipeline
    try:
        worker_pid = os.getpid()
        # print(f"[Worker PID: {worker_pid}] Initializing... Target PADDLEX_HOME_OVERRIDE is: {worker_paddlex_home_override})
        
        if worker_paddlex_home_override:
            # 确保环境变量在工作进程中也设置 (Ensure env var is also set in worker)
            os.environ['PADDLEX_HOME'] = worker_paddlex_home_override
            # print(f"[Worker PID: {worker_pid}]   os.environ['PADDLEX_HOME'] set to: {os.environ.get('PADDLEX_HOME')})
            
            # 确保目标缓存目录结构存在 (Ensure target cache directory structure exists)
            try:
                os.makedirs(worker_paddlex_home_override, exist_ok=True)
                worker_temp_dir_to_create = os.path.join(worker_paddlex_home_override, "temp")
                os.makedirs(worker_temp_dir_to_create, exist_ok=True)
                # print(f"[Worker PID: {worker_pid}]   Ensured PADDLEX_HOME_OVERRIDE directories exist: {worker_temp_dir_to_create})
            except Exception as e_mkdir:
                print(f"[Worker PID: {worker_pid}]   ERROR creating PADDLEX_HOME_OVERRIDE directories: {e_mkdir}")
        # else:
            # print(f"[Worker PID: {worker_pid}] PADDLEX_HOME_OVERRIDE not passed. PaddleX may use default home.")

        # # 这部分日志可以保留，以观察PaddleX实际使用的临时目录
        # # effective_temp_dir = get_paddlex_temp_dir()
        # # print(f"[Worker PID: {worker_pid}] Effective PaddleX temp dir (via get_paddlex_temp_dir): {effective_temp_dir})
        global_pipeline = pdx.create_pipeline(paddlex_config_path, hpi_params={"batch_size": batch_size_ocr})
        final_pdx_home_env_in_worker = os.environ.get('PADDLEX_HOME', 'Not Set')
        colored_output(f"[{get_beijing_time()}] Worker (PID:{worker_pid}) initialized. PADDLEX_HOME env: {final_pdx_home_env_in_worker}", "green")

    except Exception as e:
        worker_pid_err = os.getpid()
        pdx_home_at_err = os.environ.get('PADDLEX_HOME', 'Not Set') 
        effective_temp_dir_at_err = get_paddlex_temp_dir() # Get it even on error for logging

        colored_output(f"[{get_beijing_time()}] Error initializing worker (PID: {worker_pid_err}): {e}", "red")
        colored_output(f"[{get_beijing_time()}] Worker (PID: {worker_pid_err}) PADDLEX_HOME env var at error: {pdx_home_at_err}", "red")
        colored_output(f"[{get_beijing_time()}] Worker (PID: {worker_pid_err}) Effective temp dir (get_paddlex_temp_dir) at error: {effective_temp_dir_at_err}", "red")
        raise

def render_page(args):
    """
    Renders a single PDF page to an image. Checks if image already exists.
    Handles potential errors during PDF processing more gracefully.
    """
    page_num, pdf_path, output_folder, image_path = args  # Unpack expected_image_path
    try:
        if os.path.exists(image_path):
            return image_path, True  # Return path and indicate skipped rendering

        # Attempt to open the document for each page individually within the process
        # This might be more robust than keeping a global document object or passing it around.
        doc = None # Initialize doc to None
        try:
            doc = fitz.open(pdf_path) # Potentially problematic line
            page = doc.load_page(page_num) # load the page
        except Exception as fitz_error: # Catch any fitz error during open/load
            # Specific check for common messages, though exact error types/codes are better if available
            err_str = str(fitz_error).lower()
            if "closed" in err_str or "encrypted" in err_str or "format" in err_str or "damaged" in err_str:
                print(f"Error opening/loading page {page_num + 1} from {pdf_path} (likely PDF issue: {fitz_error}). Skipping page.")
                if doc: doc.close() # Ensure doc is closed if it was opened partially
                return None, False # Indicate failure, not skipped due to existence
            else:
                raise # Re-raise other fitz errors

        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        pix = page.get_pixmap(dpi=300)
        pix.save(image_path)
        if doc: doc.close() # Ensure doc is closed after successful processing
        return image_path, False  # Return path and indicate it was rendered
    except Exception as e: # Catch-all for other errors during rendering (e.g., save issues)
        print(f"Error rendering page {page_num + 1} from {pdf_path} to {image_path}: {e}")
        # Ensure doc is closed if an error occurs after it was opened
        if 'doc' in locals() and doc is not None and hasattr(doc, 'close') and not doc.is_closed:
             try: doc.close() 
             except: pass
        return None, False # Indicate failure


def pdf_to_images_and_submit_ocr(
    pdf_path,
    temp_image_folder_for_pdf,  # Folder where PNGs for this PDF go
    output_dir_for_pdf_pages,   # Folder where JSON results for this PDF's pages go
    ocr_pool,
    ocr_futures_list,
    error_dir,
    log_file_path,
    global_skipped_ocr_count_list, # list containing a single int
    global_submitted_to_ocr_count_list, # list containing a single int
    num_render_processes_per_pdf,
    total_pages_for_this_pdf_list, # list to update total pages for progress
    processed_pages_for_this_pdf_list, # list to update rendered pages for progress
    lock # for thread-safe counter updates
):
    """
    Converts a single PDF to images and submits pages for OCR if not already processed.
    Uses a Pool for rendering pages of this specific PDF.
    """
    os.makedirs(temp_image_folder_for_pdf, exist_ok=True)
    os.makedirs(output_dir_for_pdf_pages, exist_ok=True)

    pdf_filename = os.path.basename(pdf_path)
    colored_output(f"[{get_beijing_time()}] Starting PDF processing for: {pdf_filename}", "blue", log_file_path)
    
    pages_submitted_for_ocr_this_pdf = 0
    pages_skipped_rendering_this_pdf = 0
    pages_ocr_skipped_this_pdf = 0

    try:
        doc = fitz.open(pdf_path)
        num_pages = doc.page_count
        doc.close()
        with lock:
            total_pages_for_this_pdf_list[0] = num_pages
            processed_pages_for_this_pdf_list[0] = 0


        if num_pages == 0:
            colored_output(f"[{get_beijing_time()}] PDF {pdf_filename} has 0 pages.", "yellow", log_file_path)
            return

        tasks = []
        for page_num in range(num_pages):
            image_filename = f"page_{page_num + 1:04}.png"
            expected_image_path = os.path.join(temp_image_folder_for_pdf, image_filename)
            tasks.append((page_num, pdf_path, temp_image_folder_for_pdf, expected_image_path))

        # Using a new pool for rendering pages of the current PDF
        with Pool(processes=num_render_processes_per_pdf) as render_pool:
            render_results = render_pool.imap_unordered(render_page, tasks)

            for i, (rendered_image_path, was_skipped_render) in enumerate(render_results):
                with lock:
                    processed_pages_for_this_pdf_list[0] +=1
                
                current_progress = processed_pages_for_this_pdf_list[0] / total_pages_for_this_pdf_list[0] if total_pages_for_this_pdf_list[0] > 0 else 0
                print(
                    f"PDF to Image - {pdf_filename}: [{current_progress:.2%}] {processed_pages_for_this_pdf_list[0]}/{total_pages_for_this_pdf_list[0]} pages rendered/checked. ",
                    end="\r",
                    flush=True
                )

                if was_skipped_render:
                    pages_skipped_rendering_this_pdf += 1

                if rendered_image_path:
                    page_base_name = os.path.splitext(os.path.basename(rendered_image_path))[0]
                    expected_json_output_path = os.path.join(output_dir_for_pdf_pages, f"{page_base_name}_result.json")

                    if os.path.exists(expected_json_output_path):
                        with lock:
                            global_skipped_ocr_count_list[0] += 1
                        pages_ocr_skipped_this_pdf +=1
                        continue  # Skip OCR submission

                    # Submit to OCR pool
                    task_args = (rendered_image_path, output_dir_for_pdf_pages, error_dir, log_file_path, False) # False means not a direct PDF path, but an image
                    future = ocr_pool.apply_async(process_image, args=(task_args,))
                    ocr_futures_list.append(future)
                    with lock:
                        global_submitted_to_ocr_count_list[0] += 1
                    pages_submitted_for_ocr_this_pdf +=1
            print() # Newline after PDF rendering progress
            colored_output(f"[{get_beijing_time()}] PDF {pdf_filename}: Rendered/checked {num_pages} pages. Skipped rendering: {pages_skipped_rendering_this_pdf}. OCR skipped (JSON exists): {pages_ocr_skipped_this_pdf}. Submitted to OCR: {pages_submitted_for_ocr_this_pdf}.", "green", log_file_path)

    except Exception as e:
        colored_output(f"[{get_beijing_time()}] Error processing PDF {pdf_filename}: {e}", "red", log_file_path)


def pdf_to_images_multiprocess(pdf_path, output_folder, num_processes=16):
    # This function is now mostly superseded by pdf_to_images_and_submit_ocr for the main flow.
    # It can be kept for other purposes or removed if no longer needed.
    # For now, let's comment out its direct usage or ensure it's not called in the main path.
    # If needed, it should also be updated to use the new render_page return signature.
    os.makedirs(output_folder, exist_ok=True)
    image_paths = []
    try:
        doc = fitz.open(pdf_path)
        num_pages = doc.page_count

        tasks = []
        for page_num in range(num_pages):
            # Generate expected image path to pass to render_page
            image_filename = f"page_{page_num + 1:04}.png"
            expected_image_path = os.path.join(output_folder, image_filename)
            tasks.append((page_num, pdf_path, output_folder, expected_image_path))
        
        with Pool(processes=num_processes) as pool:
            results = pool.imap_unordered(render_page, tasks)
            start_time_pdf_render = time.time()
            processed_render_count = 0
            for i, (result_path, was_skipped) in enumerate(results):
                processed_render_count +=1
                if result_path:
                    image_paths.append(result_path)
                
                elapsed_time = time.time() - start_time_pdf_render
                progress = processed_render_count / num_pages if num_pages > 0 else 0
                speed = processed_render_count / elapsed_time if elapsed_time > 0 else 0
                print(
                    f"Legacy PDF to Image Progress: [{progress:.2%}] {processed_render_count}/{num_pages} pages, "
                    f"Speed: {speed:.2f} pages/sec, "
                    f"Elapsed: {elapsed_time:.2f} sec",
                    end="\r"
                )
            print("\nLegacy PDF to Image conversion complete.")

        doc.close()
        image_paths.sort(key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
    except Exception as e:
        print(f"Error in legacy pdf_to_images_multiprocess for {pdf_path}: {e}")
        return []
    return image_paths



def process_image(image_info):
    """Processes a single image or PDF page image, handles errors.
    Adds detailed timing for sub-steps.
    """
    global global_pipeline
    worker_pid = os.getpid() # Get worker PID for logging

    if len(image_info) != 5:
        # Log detailed error with PID
        error_msg = f"[工作进程 PID: {worker_pid}] 无效的 image_info 格式: {image_info}"
        # Attempt to log to file if log_file_path is somehow available or use print as fallback
        log_file_path_local = image_info[3] if len(image_info) > 3 and isinstance(image_info[3], str) else None
        if log_file_path_local:
            colored_output(f"[{get_beijing_time()}] {error_msg}", "red", log_file_path_local, stdout_too=False) # 只记录到文件
        else:
            print(f"[{get_beijing_time()}] {error_msg}") # Fallback
        raise ValueError(error_msg)

    image_path, output_dir, error_dir, log_file_path, is_pdf = image_info

    if not all(isinstance(item, str) for item in [image_path, output_dir, error_dir, log_file_path]):
        error_msg = f"[工作进程 PID: {worker_pid}] 无效的 image_info 类型: {image_info}"
        colored_output(f"[{get_beijing_time()}] {error_msg}", "red", log_file_path, stdout_too=False) # 只记录到文件
        raise TypeError(error_msg)


    if global_pipeline is None:
        error_msg = f"[工作进程 PID: {worker_pid}] Pipeline未初始化! 正在处理 {image_path}"
        colored_output(f"[{get_beijing_time()}] {error_msg}", "red", log_file_path, stdout_too=False) # 只记录到文件
        raise RuntimeError(error_msg)

    step_start_time = time.time()
    stage_times = [] # To store (stage_name, duration)

    try:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # Start processing log - keep this on stdout for initial user feedback if desired, or set stdout_too=False
        colored_output(f"[{get_beijing_time()}] [PID:{worker_pid}] 开始处理 {image_path}", "blue", log_file_path, stdout_too=False)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"输入文件未找到: {image_path}")

        # 1. Image Validation
        val_start_time = time.time()
        try:
            img = Image.open(image_path)
            img.verify()  # Restore validation
            img.close()
            val_duration = time.time() - val_start_time
            stage_times.append(("Validation", val_duration))
            colored_output(f"[{get_beijing_time()}] [PID:{worker_pid}] {image_path}: 校验耗时={val_duration:.3f}s.", "blue", log_file_path, stdout_too=False)
        except (IOError, SyntaxError, Exception) as e: # Catch generic Exception too for safety
            # If validation fails, it's an error for this image, treat it like other processing errors
            raise Exception(f"图片校验失败: {e} (PIL/Pillow 错误)")


        # 2. OCR Prediction
        ocr_pred_start_time = time.time()
        with RedirectStdout():  # Use the context manager
            output = global_pipeline.predict(image_path) # This is ocr_output_from_predict
        ocr_pred_duration = time.time() - ocr_pred_start_time
        stage_times.append(("OCR_Predict", ocr_pred_duration))
        colored_output(f"[{get_beijing_time()}] [PID:{worker_pid}] {image_path}: OCR预测耗时={ocr_pred_duration:.3f}s.", "blue", log_file_path, stdout_too=False)


        # 3. Save Results
        save_start_time = time.time()
        num_results_saved = 0
        ocr_output_from_predict = output # Keep original variable name for clarity here

        if ocr_output_from_predict:
            actual_result_objects = []
            if isinstance(ocr_output_from_predict, types.GeneratorType):
                try:
                    actual_result_objects = list(ocr_output_from_predict) # Exhaust generator
                except Exception as gen_ex:
                    colored_output(f"[{get_beijing_time()}] [PID:{worker_pid}] 转换来自 predict() 的生成器时出错 ({image_path}): {gen_ex}", "red", log_file_path, stdout_too=False)
            elif isinstance(ocr_output_from_predict, list):
                actual_result_objects = ocr_output_from_predict
            elif hasattr(ocr_output_from_predict, 'save_to_json'): # Single result object not in a list
                actual_result_objects = [ocr_output_from_predict]
            else:
                colored_output(f"[{get_beijing_time()}] [PID:{worker_pid}] 来自 predict() 的OCR输出类型意外 ({image_path}). 类型: {type(ocr_output_from_predict)}. 输出片段: {str(ocr_output_from_predict)[:200]}", "yellow", log_file_path, stdout_too=False)

            if not actual_result_objects:
                colored_output(f"[{get_beijing_time()}] [PID:{worker_pid}] 处理 predict() 输出后未找到具体OCR项 ({image_path}). 原始输出类型 {type(output)}", "yellow", log_file_path, stdout_too=False)
            else:
                for i, res_item in enumerate(actual_result_objects):
                    if res_item and hasattr(res_item, 'save_to_json'):
                        json_filename = f"{base_name}_result.json"
                        # If predict somehow gave multiple json-able results for one image, name them uniquely.
                        if len(actual_result_objects) > 1:
                            json_filename = f"{base_name}_result_part_{i}.json"
                        
                        try:
                            res_item.save_to_json(
                                save_path=os.path.join(output_dir, json_filename),
                                indent=4,
                                ensure_ascii=False,
                            )
                            num_results_saved += 1
                        except Exception as save_ex:
                            colored_output(f"[{get_beijing_time()}] [PID:{worker_pid}] 调用 save_to_json 时出错 ({image_path}, part {i}): {save_ex}", "red", log_file_path, stdout_too=False)

                    elif res_item: # It's an item, but can't be saved
                        colored_output(f"[{get_beijing_time()}] [PID:{worker_pid}] OCR结果项 ({image_path}, 索引 {i}) 非None但无 save_to_json 方法. 项类型: {type(res_item)}", "yellow", log_file_path, stdout_too=False)
                    # If res_item is None, it's implicitly skipped by `if res_item`
            
            if num_results_saved == 0 and actual_result_objects: # Had items, but none were saveable or saved successfully
                 colored_output(f"[{get_beijing_time()}] [PID:{worker_pid}] 为 {image_path} 处理了 {len(actual_result_objects)} 个结果项, 但无一成功保存为JSON。", "yellow", log_file_path, stdout_too=False)

        else: # ocr_output_from_predict is None or evaluates to False (e.g. empty list from predict)
            colored_output(f"[{get_beijing_time()}] [PID:{worker_pid}] {image_path} 无OCR输出 (predict() 返回 None/空)。未保存JSON。", "yellow", log_file_path, stdout_too=False)
        
        save_duration = time.time() - save_start_time
        stage_times.append(("Save_JSON", save_duration))
        # Log if save was attempted, took time, or if initial output existed but yielded no concrete items to save.
        if num_results_saved > 0 or save_duration > 0.001 or (output and not actual_result_objects and not num_results_saved):
            colored_output(f"[{get_beijing_time()}] [PID:{worker_pid}] {image_path}: 保存JSON耗时={save_duration:.3f}s, 保存结果数={num_results_saved}.", "blue", log_file_path, stdout_too=False)

        total_ocr_task_time = time.time() - step_start_time
        # Final success message for this image - keep on stdout for main progress or set stdout_too=False
        colored_output(f"[{get_beijing_time()}] [PID:{worker_pid}] 处理完成 {image_path}. 总耗时={total_ocr_task_time:.3f}s.", "green", log_file_path, stdout_too=False) 
        return total_ocr_task_time, image_path

    except Exception as e:
        total_ocr_task_time_on_error = time.time() - step_start_time
        # Log stage times accumulated so far, if any
        for stage_name, duration in stage_times:
            colored_output(f"[{get_beijing_time()}] [PID:{worker_pid}] {image_path}: (错误路径) {stage_name}_错误前耗时={duration:.3f}s.", "yellow", log_file_path, stdout_too=False)

        colored_output(f"[{get_beijing_time()}] [PID:{worker_pid}] 处理 {image_path} 时出错: {e}. 错误前总耗时={total_ocr_task_time_on_error:.3f}s", "red", log_file_path, stdout_too=False)
        try:
            error_copy_path = os.path.join(error_dir, f"FAILED_{os.path.basename(image_path)}")
            shutil.copy2(image_path, error_copy_path)
        except Exception as copy_error:
            colored_output(f"[{get_beijing_time()}] [PID:{worker_pid}] 复制失败文件 {image_path} 时出错: {copy_error}", "red", log_file_path, stdout_too=False)
        return None, image_path  # Return None for time, but still the path


def main():
    global num_images, processed_count, error_count # These will be redefined or reused carefully
    # global image_root_dir # No longer needed
    global error_dir
    global log_file_path
    global start_time

    # 使用顶部定义的参数 (Using parameters defined at the top)
    input_sources = INPUT_SOURCES
    output_root_dir = OUTPUT_ROOT_DIR
    log_and_error_dir = LOG_AND_ERROR_DIR_BASE # Base directory for logs and errors
    
    # 从基础目录派生 error_dir 和 log_file_path (Derive error_dir and log_file_path from the base)
    error_dir = os.path.join(log_and_error_dir, "error_images")
    log_file_path = os.path.join(log_and_error_dir, "ocr_log.txt")
    
    delete_temp_images_after_pdf_processing = DELETE_TEMP_IMAGES_AFTER_PDF_PROCESSING
    num_render_processes_per_pdf_val = NUM_RENDER_PROCESSES_PER_PDF
    num_concurrent_pdf_prep_processes_val = NUM_CONCURRENT_PDF_PREP_PROCESSES # 使用新参数

    os.makedirs(output_root_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)
    os.makedirs(log_and_error_dir, exist_ok=True) 

    # 调用 clear_cache()，此时 log_file_path 已定义 (Call clear_cache() now that log_file_path is defined)
    # clear_cache 会使用 PADDLEX_HOME_OVERRIDE (clear_cache will use PADDLEX_HOME_OVERRIDE)
    clear_cache()

    start_time = time.time()
    start_time_str = get_beijing_time()
    colored_output(f"[{start_time_str}] OCR process started.", "green", log_file_path)

    num_ocr_processes_val = NUM_OCR_PROCESSES
    batch_size_val = OCR_BATCH_SIZE
    use_cpu_val = USE_CPU_FOR_OCR
    
    # config_to_use 的确定依赖于 config_path 和 use_cpu_val
    # config_path 在全局作用域中已经从 CONFIG_PATH_PADDLE 初始化
    config_to_use = modify_config_for_cpu(config_path) if use_cpu_val else config_path


    ocr_futures = []
    skipped_ocr_count_due_to_json = [0] 
    submitted_to_ocr_count = [0]
    
    processed_ocr_tasks_count = 0
    error_ocr_tasks_count = 0
    
    temp_folders_to_delete = [] 

    #统计PDF页面转换的变量
    total_pages_for_current_pdf_list = [0]
    processed_pages_for_current_pdf_list = [0]
    progress_lock = Lock()


    colored_output(f"[{get_beijing_time()}] Scanning input sources...", "blue", log_file_path)
    colored_output(f"[{get_beijing_time()}] Using {num_ocr_processes_val} processes for OCR.", "blue", log_file_path)
    colored_output(f"[{get_beijing_time()}] Using up to {num_render_processes_per_pdf_val} processes per PDF for page rendering.", "blue", log_file_path)
    colored_output(f"[{get_beijing_time()}] Using up to {num_concurrent_pdf_prep_processes_val} processes for concurrent PDF file preparation.", "blue", log_file_path) # 新日志
    colored_output(f"[{get_beijing_time()}] Batch size for OCR pipeline: {batch_size_val}", "blue", log_file_path)
    colored_output(f"[{get_beijing_time()}] PADDLEX_HOME override: {PADDLEX_HOME_OVERRIDE if PADDLEX_HOME_OVERRIDE else '未设置 (使用默认路径)'}", "blue", log_file_path)


    all_image_tasks_for_ocr = [] # 存储所有最终需要OCR的图片任务参数 (image_path, output_dir, error_dir, log_file_path, False)
    aggregated_skipped_ocr_count = 0
    # temp_folders_to_delete 列表的收集方式也需要调整，在 prepare_single_pdf_for_ocr 中决定

    pdf_prep_tasks = []

    # --- 阶段1: 扫描输入源，构建初始任务列表 ---
    scan_start_time = time.time()
    colored_output(f"[{get_beijing_time()}] 开始扫描输入源以构建任务列表...", "blue", log_file_path)

    for source_info in input_sources:
        current_input_root_path = source_info["path"]
        input_type = source_info["type"]
        colored_output(f"[{get_beijing_time()}] 正在扫描源: {current_input_root_path} (类型: {input_type})", "blue", log_file_path)

        for root, _, files in os.walk(current_input_root_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path_from_current_input_root = os.path.relpath(root, current_input_root_path)
                
                if input_type == "pdf" and file.lower().endswith(".pdf"):
                    pdf_name_without_ext = os.path.splitext(file)[0]
                    # 输出目录结构保持不变
                    output_dir_for_this_pdf_ocr_results = os.path.join(output_root_dir, relative_path_from_current_input_root, pdf_name_without_ext)
                    temp_image_folder_for_this_pdf = os.path.join(output_dir_for_this_pdf_ocr_results, "temp_images_from_pdf")
                    
                    # 参数包，用于传递给 prepare_single_pdf_for_ocr
                    pdf_prep_args = {
                        "pdf_path": file_path,
                        "temp_image_folder_for_pdf": temp_image_folder_for_this_pdf,
                        "output_dir_for_pdf_ocr_results": output_dir_for_this_pdf_ocr_results,
                        "num_render_processes_per_pdf": num_render_processes_per_pdf_val,
                        "log_file_path": log_file_path,
                        "error_dir": error_dir, # error_dir for OCR, not for PDF rendering errors directly handled by prepare_single_pdf_for_ocr
                        "pdf_filename_for_log": file # 用于日志
                    }
                    pdf_prep_tasks.append(pdf_prep_args)
                
                elif input_type == "image" and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                    image_name_without_ext = os.path.splitext(file)[0]
                    output_dir_for_this_image_ocr_results = os.path.join(output_root_dir, relative_path_from_current_input_root)
                    os.makedirs(output_dir_for_this_image_ocr_results, exist_ok=True)
                    
                    expected_json_output_path = os.path.join(output_dir_for_this_image_ocr_results, f"{image_name_without_ext}_result.json")
                    if os.path.exists(expected_json_output_path):
                        aggregated_skipped_ocr_count += 1
                        continue
                    # 直接添加到总的OCR任务列表
                    all_image_tasks_for_ocr.append((file_path, output_dir_for_this_image_ocr_results, error_dir, log_file_path, False))

    scan_duration = time.time() - scan_start_time
    colored_output(f"[{get_beijing_time()}] 输入源扫描完成。耗时: {format_timedelta(timedelta(seconds=scan_duration))}", "green", log_file_path)
    colored_output(f"[{get_beijing_time()}] 共找到 {len(pdf_prep_tasks)} 个PDF文件待准备，{len(all_image_tasks_for_ocr)} 个图片文件待直接OCR。", "blue", log_file_path)

    # --- 阶段2: PDF 准备阶段 ---
    if pdf_prep_tasks:
        pdf_prep_stage_start_time = time.time()
        colored_output(f"[{get_beijing_time()}] 开始PDF准备阶段，处理 {len(pdf_prep_tasks)} 个PDF文件，使用 {num_concurrent_pdf_prep_processes_val} 个并行进程...", "blue", log_file_path)
        prepared_pdf_count = 0
        
        # 使用非守护进程的自定义Pool替代方法，或者直接管理Process对象
        # 对于 Python 3.8+ 和 spawn/forkserver context, Pool 进程默认非守护
        # 如果是 fork context (Linux默认), Pool 进程是守护的。
        # get_context("spawn").Pool 创建的进程是非守护的，所以嵌套应该没问题。
        # 问题可能是 Pool 内部实现与 Fitz 的交互问题，或者资源耗尽。
        # 为了简化并解决 daemonic process error, prepare_single_pdf_for_ocr 内部的 Pool 将被移除。
        # 页面渲染将在 prepare_single_pdf_for_ocr 中串行进行。

        with get_context("spawn").Pool(processes=num_concurrent_pdf_prep_processes_val) as pdf_prep_pool:
            all_pdf_prep_results = []
            for result in pdf_prep_pool.imap_unordered(prepare_single_pdf_for_ocr, pdf_prep_tasks):
                if result:
                    all_pdf_prep_results.append(result)
                prepared_pdf_count += 1
                colored_output(f"[{get_beijing_time()}] PDF准备进度: {prepared_pdf_count}/{len(pdf_prep_tasks)} 个PDF已由准备工作进程处理完毕。", "yellow", log_file_path)
        
        # 确保在Pool关闭后再处理结果
        for prep_result_item in all_pdf_prep_results:
            # 正确解包 prepare_single_pdf_for_ocr 返回的4个值
            if len(prep_result_item) == 4:
                img_paths, skipped_count, temp_folder, ocr_output_dir = prep_result_item
                if img_paths:
                    for img_path in img_paths:
                        task_args = (img_path, ocr_output_dir, error_dir, log_file_path, False)
                        all_image_tasks_for_ocr.append(task_args)
                aggregated_skipped_ocr_count += skipped_count
                if temp_folder:
                    temp_folders_to_delete.append(temp_folder)
            else:
                colored_output(f"[{get_beijing_time()}] 警告: PDF准备结果格式不符合预期 (应为4个元素): {prep_result_item}", "red", log_file_path)

        pdf_prep_stage_duration = time.time() - pdf_prep_stage_start_time
        colored_output(f"[{get_beijing_time()}] PDF准备阶段完成。耗时: {format_timedelta(timedelta(seconds=pdf_prep_stage_duration))}", "green", log_file_path)
        colored_output(f"[{get_beijing_time()}] PDF准备后，总计待OCR图片/页面数: {len(all_image_tasks_for_ocr)}, 总计跳过(JSON已存在): {aggregated_skipped_ocr_count}.", "blue", log_file_path)

    # --- 阶段3: OCR 执行阶段 ---
    ocr_stage_start_time = time.time()
    submitted_to_ocr_actual_count = len(all_image_tasks_for_ocr)
    colored_output(f"[{get_beijing_time()}] 开始OCR执行阶段。总计待OCR图片/页面数: {submitted_to_ocr_actual_count}.", "blue", log_file_path)

    with get_context("spawn").Pool(
        processes=num_ocr_processes_val,
        initializer=init_worker,
        initargs=(config_to_use, batch_size_val, PADDLEX_HOME_OVERRIDE),
    ) as ocr_pool:
        if not all_image_tasks_for_ocr:
            colored_output(f"[{get_beijing_time()}] 在准备阶段之后，没有图片或PDF页面需要OCR。", "yellow", log_file_path)
        else:
            for task_args in all_image_tasks_for_ocr:
                future = ocr_pool.apply_async(process_image, args=(task_args,))
                ocr_futures.append(future)
        
        colored_output(f"[{get_beijing_time()}] 所有OCR任务已提交给工作进程。开始监控进度...", "blue", log_file_path)
        
        ocr_pool.close()
        # Progress monitoring for OCR tasks
        total_ocr_tasks_to_monitor = submitted_to_ocr_actual_count # Use the actual count of submitted tasks
        if total_ocr_tasks_to_monitor == 0:
             colored_output(f"[{get_beijing_time()}] No new OCR tasks to process.", "yellow", log_file_path)
        else:
            colored_output(f"[{get_beijing_time()}] Monitoring OCR progress for {total_ocr_tasks_to_monitor} tasks...", "blue", log_file_path)
        
        while processed_ocr_tasks_count + error_ocr_tasks_count < total_ocr_tasks_to_monitor:
            # Check futures for completion
            # A more robust way for a large number of futures might involve a queue for results or callbacks
            # For simplicity with apply_async, we iterate and check `ready()`
            current_loop_processed_futures = 0
            remaining_futures_this_loop = []
            for future_idx, future in enumerate(ocr_futures):
                if future.ready():
                    current_loop_processed_futures +=1
                    try:
                        ocr_time, processed_img_path = future.get() # process_image returns (total_ocr_task_time, image_path)
                        if ocr_time is None: # Indicates an error from process_image
                            error_ocr_tasks_count += 1
                            # colored_output(f"[{get_beijing_time()}] OCR任务失败（从future获取）: {processed_img_path if processed_img_path else 'Unknown path'}", "red", log_file_path) # Logged in process_image
                        else:
                            processed_ocr_tasks_count += 1
                            # colored_output(f"[{get_beijing_time()}] OCR任务成功: {processed_img_path}, 耗时: {ocr_time:.3f}s", "blue", log_file_path) # Logged in process_image
                    except Exception as e_get:
                        colored_output(f"[{get_beijing_time()}] 获取OCR结果时发生严重错误: {e_get}", "red", log_file_path)
                        error_ocr_tasks_count += 1 # Count as error if future.get() fails
                else:
                    remaining_futures_this_loop.append(future)
            
            ocr_futures = remaining_futures_this_loop # Update list to only non-ready futures

            # Update progress display
            current_processed_total = processed_ocr_tasks_count + error_ocr_tasks_count
            if total_ocr_tasks_to_monitor > 0 :
                elapsed_time_ocr = time.time() - start_time # Use overall start_time for ETA
                avg_time_per_item_ocr = elapsed_time_ocr / current_processed_total if current_processed_total > 0 else 0
                remaining_items_ocr = total_ocr_tasks_to_monitor - current_processed_total
                remaining_time_seconds_ocr = remaining_items_ocr * avg_time_per_item_ocr if avg_time_per_item_ocr > 0 else 0
                eta_datetime_ocr = (datetime.now() + timedelta(seconds=remaining_time_seconds_ocr))
                eta_str_ocr = eta_datetime_ocr.strftime("%Y-%m-%d %H:%M:%S")
                remaining_formatted_str_ocr = format_timedelta(timedelta(seconds=remaining_time_seconds_ocr))
                current_time_str = datetime.now().strftime("%H:%M:%S") # Get current time for progress bar

                print(
                    f"[{current_time_str}] OCR Progress: Processed {current_processed_total}/{total_ocr_tasks_to_monitor} "
                    f"({current_processed_total/total_ocr_tasks_to_monitor*100:.1f}%), "
                    f"Avg. Speed: {avg_time_per_item_ocr:.3f} s/item, ETA: {eta_str_ocr} ({remaining_formatted_str_ocr}), "
                    f"Success: {processed_ocr_tasks_count}, Errors: {error_ocr_tasks_count}   ",
                    end="\r",
                    flush=True
                )
                if current_processed_total > 0 and current_processed_total % 100 == 0 and current_processed_total < total_ocr_tasks_to_monitor:
                    colored_output(f"\n[{get_beijing_time()}] OCR Checkpoint: Processed {current_processed_total} items.", "yellow", log_file_path)

            if not ocr_futures and (processed_ocr_tasks_count + error_ocr_tasks_count < total_ocr_tasks_to_monitor) :
                # This case should ideally not be hit if logic is correct, but as a safeguard:
                # If all futures are processed but counts don't match, log and break to avoid infinite loop.
                colored_output(f"[{get_beijing_time()}] 警告: Futures已耗尽，但OCR计数不匹配。可能是任务在提交或获取结果时丢失。正在跳出循环。", "red", log_file_path)
                colored_output(f"[{get_beijing_time()}] 期望任务数: {total_ocr_tasks_to_monitor}, 已处理成功: {processed_ocr_tasks_count}, 已处理错误: {error_ocr_tasks_count}", "red", log_file_path)
                break
            
            # Log if no futures were processed in this loop iteration but still tasks pending
            if current_loop_processed_futures == 0 and ocr_futures and (processed_ocr_tasks_count + error_ocr_tasks_count < total_ocr_tasks_to_monitor):
                colored_output(f"[{get_beijing_time()}] OCR监控: 本轮未处理任何future，但仍有 {len(ocr_futures)} 个任务待完成。总任务数 {total_ocr_tasks_to_monitor}, 已完成 {processed_ocr_tasks_count}, 错误 {error_ocr_tasks_count}", "yellow", log_file_path)

            time.sleep(1) # Polling interval

        print() # Newline after OCR progress bar
        ocr_pool.join() # Wait for all worker processes to exit
        ocr_stage_duration = time.time() - ocr_stage_start_time
        colored_output(f"[{get_beijing_time()}] OCR执行阶段完成。耗时: {format_timedelta(timedelta(seconds=ocr_stage_duration))}", "green", log_file_path)

    # Cleanup and Summary
    if delete_temp_images_after_pdf_processing:
        colored_output(f"[{get_beijing_time()}] Deleting temporary image folders from PDF conversions...", "blue", log_file_path)
        # Deduplicate temp_folders_to_delete before iterating
        unique_temp_folders = sorted(list(set(temp_folders_to_delete)))
        for temp_folder in unique_temp_folders:
            if os.path.exists(temp_folder): # Check if it still exists
                try:
                    shutil.rmtree(temp_folder)
                    colored_output(f"[{get_beijing_time()}] Deleted: {temp_folder}", "green", log_file_path)
                except Exception as e:
                    colored_output(f"[{get_beijing_time()}] Error deleting {temp_folder}: {e}", "red", log_file_path)
            else:
                 colored_output(f"[{get_beijing_time()}] Temporary folder already deleted or moved: {temp_folder}", "yellow", log_file_path)
    else:
        colored_output(f"[{get_beijing_time()}] Temporary image folders from PDF conversions were kept.", "yellow", log_file_path)

    end_time = time.time()
    total_time = end_time - start_time
    colored_output(f"[{get_beijing_time()}] OCR process finished.", "green", log_file_path)
    colored_output(f"[{get_beijing_time()}] OCR results saved to: {output_root_dir}", "green", log_file_path)
    colored_output(f"[{get_beijing_time()}] Total processing time: {format_timedelta(timedelta(seconds=total_time))}", "green", log_file_path)
    
    actual_ocr_processed_successfully = processed_ocr_tasks_count
    total_ocr_tasks_handled = processed_ocr_tasks_count + error_ocr_tasks_count

    if total_ocr_tasks_handled > 0:
        colored_output(f"[{get_beijing_time()}] Average time per OCR item (image/page): {total_time / total_ocr_tasks_handled:.3f} seconds", "green", log_file_path)
    
    colored_output(f"[{get_beijing_time()}] Total OCR tasks submitted to queue: {submitted_to_ocr_actual_count}", "green", log_file_path)
    colored_output(f"[{get_beijing_time()}] OCR tasks successfully processed: {actual_ocr_processed_successfully}", "green", log_file_path)
    colored_output(f"[{get_beijing_time()}] OCR tasks with errors: {error_ocr_tasks_count}", "red", log_file_path)
    colored_output(f"[{get_beijing_time()}] Files/Pages skipped (JSON existed before OCR): {aggregated_skipped_ocr_count}", "yellow", log_file_path)

# Placeholder for the new PDF preparation worker function
# This function will be called by the pdf_prep_pool
# It needs to handle rendering for a single PDF and return a list of images that need OCR
def prepare_single_pdf_for_ocr(pdf_prep_args):
    pdf_path = pdf_prep_args["pdf_path"]
    temp_image_folder_for_pdf = pdf_prep_args["temp_image_folder_for_pdf"]
    output_dir_for_pdf_ocr_results = pdf_prep_args["output_dir_for_pdf_ocr_results"] # This is the OCR JSON output dir
    num_render_processes = pdf_prep_args["num_render_processes_per_pdf"] # Renamed for clarity in this scope
    log_file_path = pdf_prep_args["log_file_path"]
    error_dir = pdf_prep_args["error_dir"]
    pdf_filename_for_log = pdf_prep_args["pdf_filename_for_log"]

    images_needing_ocr = []
    skipped_ocr_count_this_pdf = 0
    pages_rendered_this_pdf = 0
    pages_skipped_rendering_this_pdf = 0

    try:
        os.makedirs(temp_image_folder_for_pdf, exist_ok=True)
        os.makedirs(output_dir_for_pdf_ocr_results, exist_ok=True)

        colored_output(f"[{get_beijing_time()}] PDF Prep Worker: Starting {pdf_filename_for_log}", "blue", log_file_path)

        doc = fitz.open(pdf_path)
        num_pages = doc.page_count
        doc.close() # Close after getting page count, individual pages will reopen in render_page

        if num_pages == 0:
            colored_output(f"[{get_beijing_time()}] PDF Prep Worker: {pdf_filename_for_log} has 0 pages.", "yellow", log_file_path)
            return [], 0, temp_image_folder_for_pdf, output_dir_for_pdf_ocr_results # Return temp_image_folder_for_pdf for potential deletion

        render_tasks = []
        for page_num in range(num_pages):
            image_filename = f"page_{page_num + 1:04}.png"
            expected_image_path = os.path.join(temp_image_folder_for_pdf, image_filename)
            # render_page args: page_num, pdf_path, output_folder (temp_image_folder_for_pdf), image_path (expected_image_path)
            render_tasks.append((page_num, pdf_path, temp_image_folder_for_pdf, expected_image_path))
        
        # Progress tracking for rendering within this PDF prep worker
        processed_render_tasks_count = 0
        total_render_tasks = len(render_tasks)
        # Lock for print to avoid garbled multi-process output for this specific PDF's progress
        # This lock would need to be managed if prepare_single_pdf_for_ocr itself is run by many processes
        # For now, simple print without per-PDF progress bar to avoid complexity of shared locks across pdf_prep_pool workers

        # with Pool(processes=num_render_processes) as page_render_pool:
        #     for rendered_image_path, was_skipped_render in page_render_pool.imap_unordered(render_page, render_tasks):
        for task_args in render_tasks: # 串行处理
            rendered_image_path, was_skipped_render = render_page(task_args)
            processed_render_tasks_count += 1
            if processed_render_tasks_count % 100 == 0 or processed_render_tasks_count == total_render_tasks: # 每100页或最后一页记录日志
                colored_output(f"[{get_beijing_time()}] PDF准备工作进程 ({pdf_filename_for_log}): 页面渲染/检查进度 {processed_render_tasks_count}/{total_render_tasks}。", "blue", log_file_path)

            if was_skipped_render:
                pages_skipped_rendering_this_pdf += 1
            else:
                pages_rendered_this_pdf +=1 # if not skipped, it was rendered (or failed to render)

            if rendered_image_path:
                page_base_name = os.path.splitext(os.path.basename(rendered_image_path))[0]
                expected_json_output_path = os.path.join(output_dir_for_pdf_ocr_results, f"{page_base_name}_result.json")

                if os.path.exists(expected_json_output_path):
                    skipped_ocr_count_this_pdf += 1
                else:
                    images_needing_ocr.append(rendered_image_path)
        
        colored_output(f"[{get_beijing_time()}] PDF Prep Worker: Finished {pdf_filename_for_log}. Pages rendered: {pages_rendered_this_pdf}, Pages skipped rendering (existed): {pages_skipped_rendering_this_pdf}, Pages needing OCR: {len(images_needing_ocr)}, Pages OCR skipped (JSON existed): {skipped_ocr_count_this_pdf}.", "green", log_file_path)

    except Exception as e:
        colored_output(f"[{get_beijing_time()}] PDF Prep Worker: Error processing {pdf_filename_for_log}: {e}", "red", log_file_path)
        # In case of error, return empty list and 0 count, but still return temp_folder for potential deletion if it was created
        # Ensure it returns the output_dir_for_pdf_ocr_results as well for consistent tuple structure, even if empty.
        return [], 0, temp_image_folder_for_pdf, output_dir_for_pdf_ocr_results 

    # Return: list of image paths needing OCR, count of OCRs skipped for this PDF, temp image folder path, OCR JSON output directory for these images
    return images_needing_ocr, skipped_ocr_count_this_pdf, temp_image_folder_for_pdf, output_dir_for_pdf_ocr_results


def modify_config_for_cpu(pdx_config_path): # 参数名更改以避免与全局变量冲突
    """Modifies config for CPU."""
    base, ext = os.path.splitext(pdx_config_path)
    new_config_path = f"{base}_cpu{ext}"
    with open(pdx_config_path, "r") as f:
        config_data = yaml.safe_load(f) # 变量名更改
    config_data["Global"]["device"] = "cpu"
    if "use_gpu" in config_data["Global"]:
        del config_data["Global"]["use_gpu"]
    if "gpu_id" in config_data["Global"]:
        del config_data["Global"]["gpu_id"]
    with open(new_config_path, "w") as f:
        yaml.dump(config_data, f)
    return new_config_path

if __name__ == "__main__":
    main()
