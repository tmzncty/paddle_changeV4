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
from collections import Counter
import cv2

# Initialize colorama
init(autoreset=True)

# --- Configuration Variables ---
OCR_RESULTS_DIR = "/media/tmzn/DATA5/ocr_paddle/testpdf_ocr_result"
IMAGE_BASE_DIR = "/media/tmzn/DATA5/ocr_paddle/testpdf/tmpicture"
OUTPUT_BASE_DIRECTORY = "/media/tmzn/DATA5/ocr_paddle/output_pdfs_text_layer再说意境_叶朗"
Y_OFFSET = 0  # 确保 Y_OFFSET 为 0 以进行初始排查
NUM_PROCESSES = 32  # Start with a MUCH smaller number!
SAVE_ENHANCED_IMAGES = False
ENHANCE_IMAGES = False
ENHANCEMENT_METHOD = "default" #如果为 "clahe"，则使用 OpenCV 的 CLAHE 进行对比度增强。
ENHANCED_IMAGE_SUFFIX = "_enhanced"
CHUNK_SIZE = 50  # Process images in chunks of this size.  Adjust as needed.
CREATE_DEBUG_IMAGES = False # New flag to control debug image generation
# EXPECTED_NUMBER_OF_COLUMNS = 2 # Removed: Will now attempt auto-detection
# --- End Configuration Variables ---

# --- 日志设置 ---
def setup_logger(log_dir, name): # 日志记录器设置
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 文件处理器
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 控制台处理器 (INFO 及更高级别)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO) # 设置为 INFO 级别
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
# --- 结束日志设置 ---

# --- 旋转动画 ---
def create_spinner(): # 创建一个无限循环的旋转动画字符序列。
    return itertools.cycle([".", "..", "..."])

def print_with_spinner(spinner): # 打印旋转动画序列中的下一个字符。
    sys.stdout.write(f"\r{next(spinner)} ")
    sys.stdout.flush()
# --- 结束旋转动画 ---


def get_timestamp(): # 获取当前时间戳字符串
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def print_with_time(message, color=Fore.WHITE, log_level=logging.INFO, logger=None): # 带时间戳和颜色打印消息，并记录到日志。
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


def enhance_image(image_path, output_dir=None): # 增强单个图像，返回路径或字节流，并关闭图像。
    try:
        with Image.open(image_path) as img:  # 使用上下文管理器确保图像在处理后被关闭。
            img_l = img.convert('L') # 转换为灰度图
            
            if ENHANCEMENT_METHOD == "clahe":
                # 使用 CLAHE 增强对比度
                img_np = np.array(img_l)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img_np_clahe = clahe.apply(img_np)
                enhanced_img = Image.fromarray(img_np_clahe)
            else:
                # 默认增强方法: 自适应阈值 + 非锐化掩模
                img_np = np.array(img_l)
                # 自适应阈值处理
                threshold_value = np.mean(img_np) - np.std(img_np) / 2
                threshold_value = max(0, min(threshold_value, 255))
                img_np_thresh = np.where(img_np > threshold_value, 255, 0).astype(np.uint8)
                img_thresh = Image.fromarray(img_np_thresh)
                enhanced_img = img_thresh.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

            if output_dir: # 如果指定输出目录，则保存增强后的图像
                base_name = os.path.basename(image_path)
                output_path = os.path.join(output_dir, base_name)
                enhanced_img.save(output_path, format='PNG')
                return output_path, None
            else: # 否则返回图像字节流
                img_byte_arr = io.BytesIO()
                enhanced_img.save(img_byte_arr, format='PNG')
                data = img_byte_arr.getvalue() # 获取字节数据
                img_byte_arr.close() # 关闭 BytesIO 对象
                return data, None  # 返回字节数据
    except Exception as e:
        return None, f"增强图像 {image_path} 时出错: {e}"
    # 由于使用了 'with' 语句，图像会在此处自动关闭。


def get_image_and_json_paths(enhanced_paths, enhanced_image_dir, json_dir, image_file): # 获取正确的图像和JSON文件路径的辅助函数。
    # enhanced_image_dir 非 None 表示正在使用增强图像（无论是已保存的还是内存中的）
    if enhanced_image_dir: 
        enhanced_image_data = enhanced_paths[image_file]
        page_num_match = re.search(r"page_(\d+)", image_file, re.IGNORECASE)

        if page_num_match:
            page_num = int(page_num_match.group(1))
            json_file = f"page_{page_num:04}_result.json"
            json_path = os.path.join(json_dir, json_file)
        else:
            # 处理不含 "page_xxx" 格式的文件名
            json_file = None
            base_image_name = os.path.splitext(image_file)[0]
            for f in os.listdir(json_dir):
                if f.startswith(base_image_name) and f.endswith('_result.json'):
                    json_file = f
                    break
            if json_file:
                json_path = os.path.join(json_dir, json_file)
            else:
                return None, None  # 未找到对应的JSON文件

    else:  # 使用原始图像
        # 更稳健地替换扩展名以找到JSON文件
        json_path = os.path.join(json_dir, image_file.replace(".png", "_result.json").replace(".jpg", "_result.json").replace(".jpeg", "_result.json"))  
        page_num_match = re.search(r"page_(\d+)", image_file, re.IGNORECASE)
        if not page_num_match: # 如果没有页码，无法继续
            return None, None  
        page_num = int(page_num_match.group(1))

        original_image_file = None
        # 查找对应的图像文件 (更稳健的匹配方式)
        for key in enhanced_paths.keys():
            if f"page_{page_num}" in key.lower():
                original_image_file = key
                break
        if original_image_file is None:
            # 尝试使用文件名前缀作为后备匹配方案
            prefix_match = re.match(r"^[a-zA-Z!]+", image_file)
            if prefix_match:
                prefix = prefix_match.group(0)
                for key in enhanced_paths.keys():
                    if key.startswith(prefix):
                        original_image_file = key
                        break
        if original_image_file is None: # 如果仍然没有找到对应的图像文件
            return None, None  

        enhanced_image_data = enhanced_paths[original_image_file]

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

def process_and_create_pdfs(sub_dir_name, sub_dir_path, image_dir, output_base_dir, y_offset=Y_OFFSET,
                            save_enhanced=SAVE_ENHANCED_IMAGES, logger=None): # 处理单个子目录并创建PDF
    start_time = time.time()
    json_dir = sub_dir_path
    # 根据是否启用了图像增强（ENHANCE_IMAGES总开关）来决定输出PDF的文件名
    # 注意：这里的save_enhanced参数主要用于process_images_in_directory决定是否保存中间增强图片，
    # 而最终PDF名应取决于ENHANCE_IMAGES这个全局配置。
    if ENHANCE_IMAGES:
        output_pdf_name = f"{sub_dir_name}_searchable_enhanced.pdf"
    else:
        output_pdf_name = f"{sub_dir_name}_searchable.pdf"
    output_pdf_path = os.path.join(output_base_dir, output_pdf_name)


    if os.path.exists(output_pdf_path):
        print_with_time(f"跳过 {sub_dir_name} (PDF 已存在).", color=Fore.YELLOW, logger=logger)
        return 0, 0, 0, f"已跳过 (PDF存在): {output_pdf_path}"

    # process_images_in_directory 现在会根据 ENHANCE_IMAGES 和 ENHANCEMENT_METHOD 进行增强
    enhanced_paths, enhanced_image_dir = process_images_in_directory(image_dir, save_enhanced=save_enhanced, logger=logger)

    if enhanced_image_dir is None: # 如果没有增强图像目录（例如，未开启增强或增强后不保存到单独目录）
        # 基于JSON文件名（包含page_xxx）进行排序
        image_files_or_json_keys = sorted(
            [f for f in os.listdir(json_dir) if f.lower().endswith(('_result.json'))],
            key=lambda x: int(re.search(r"page_(\d+)", x, re.IGNORECASE).group(1)) # 按页码数字排序
        )
    else: # 如果使用了增强图像（无论保存与否，enhanced_paths都会有key）
        # 使用natsorted对enhanced_paths的键（原始文件名）进行自然排序
        # 这个排序逻辑与之前版本保持一致，处理特殊前缀
        def sort_key_for_enhanced(filename):
            filename_lower = filename.lower()
            if filename_lower.startswith("cov001"): return (-1, filename_lower) # 封面1最优先
            elif filename_lower.startswith("cov002"): return (5, filename_lower)   # 封面2（或其他低优先级封面）靠后
            elif filename_lower.startswith("bok"): return (0, filename_lower)   # 书名页
            elif filename_lower.startswith("leg"): return (1, filename_lower)   # 版权页
            elif filename_lower.startswith("fow"): return (2, filename_lower)   # 前言
            elif filename_lower.startswith("!"): return (3, filename_lower)    # 特殊符号开头
            elif filename_lower[0].isdigit(): # 数字开头
                match = re.match(r"^\d+", filename_lower)
                return (4, int(match.group(0)), filename_lower) if match else (4, 0, filename_lower) # 按数字大小排序
            else: return (6, filename_lower) # 其他文件类型，优先级较低
        image_files_or_json_keys = natsorted(enhanced_paths.keys(), key=sort_key_for_enhanced, alg=ns.IGNORECASE)
        
    if not image_files_or_json_keys:
        error_message = f"在 {'增强图像目录' if enhanced_image_dir else 'JSON目录'} 中未找到图像文件或对应的JSON键"
        print_with_time(error_message, color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
        return 0, 0, 0, error_message

    total_pages = len(image_files_or_json_keys) # 当前子目录的总页数
    processed_pages = 0 # 已成功处理的页数
    errors = 0 # 错误计数
    error_messages = [] # 存储错误信息

    for chunk_start in range(0, total_pages, CHUNK_SIZE): # 按 CHUNK_SIZE 分块处理
        chunk_end = min(chunk_start + CHUNK_SIZE, total_pages)
        current_chunk_keys = image_files_or_json_keys[chunk_start:chunk_end]
        doc = None # 每个块使用新的PDF文档对象
        try:
            doc = fitz.open() # 为当前块打开一个新的PDF文档

            for image_file_key in current_chunk_keys: # image_file_key 是原始图像文件名或JSON文件名的一部分
                img_for_debug_obj = None # 用于调试图像的对象，在finally中确保关闭
                try:  # 内层 try 块，处理单个图像和JSON
                    # enhanced_image_data可能是路径或字节流
                    # json_path是对应的JSON文件路径
                    # image_file_key 此处用作查找 enhanced_paths 和构造 json_path 的基础
                    enhanced_image_data, json_path = get_image_and_json_paths(enhanced_paths, enhanced_image_dir, json_dir, image_file_key)

                    if not json_path or not enhanced_image_data: # 如果缺少JSON或图像数据
                        errors += 1; msg = f"跳过 (无JSON或图像): {image_file_key}"; error_messages.append(msg); print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR); continue
                    if not os.path.exists(json_path): # 如果JSON文件不存在
                        errors += 1; msg = f"跳过 (JSON文件未找到): {json_path}"; error_messages.append(msg); print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR); continue

                    with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
                    if 'dt_polys' not in data or 'rec_text' not in data: # 如果JSON缺少必要数据
                        errors+=1; msg = f"跳过 (JSON缺少数据): {image_file_key}"; error_messages.append(msg); print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR); continue
                    
                    polygons = data['dt_polys']
                    texts = data['rec_text']
                    if len(polygons) != len(texts): # 如果多边形和文本数量不匹配
                        errors += 1; msg = f"跳过 (多边形/文本数量不匹配): {image_file_key}"; error_messages.append(msg); print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR); continue

                    # --- 开始: 填充 combined_ocr_elements (包含详细坐标信息) ---
                    combined_ocr_elements = []
                    if polygons and texts: 
                        for i_poly, polygon_coords_from_json_orig in enumerate(polygons):
                            if i_poly < len(texts):
                                text_orig = texts[i_poly]
                                if not text_orig.strip(): continue # 跳过空文本
                                
                                temp_x_coords = [p[0] for p in polygon_coords_from_json_orig]
                                temp_y_coords = [p[1] for p in polygon_coords_from_json_orig]
                                
                                min_x = min(temp_x_coords) if temp_x_coords else 0
                                max_x = max(temp_x_coords) if temp_x_coords else 0 
                                min_y = min(temp_y_coords) if temp_y_coords else 0
                                max_y = max(temp_y_coords) if temp_y_coords else 0 

                                combined_ocr_elements.append({
                                    "polygon": polygon_coords_from_json_orig,
                                    "text": text_orig,
                                    "min_x": min_x, "max_x": max_x,
                                    "min_y": min_y, "max_y": max_y,
                                    "box_center_x": (min_x + max_x) / 2.0 
                                })
                            else:
                                logger.warning(f"图像 {image_file_key} 中，多边形 {i_poly} 缺少对应的文本条目。已跳过此多边形。")
                    # --- 结束: 填充 combined_ocr_elements ---
                    
                    # 获取图像尺寸用于PDF页面创建和列检测
                    current_img_width, current_img_height = 0, 0
                    if isinstance(enhanced_image_data, str): # 如果是路径
                        with Image.open(enhanced_image_data) as temp_img_for_size:
                            current_img_width, current_img_height = temp_img_for_size.size
                    else: # 如果是字节流
                        with Image.open(io.BytesIO(enhanced_image_data)) as temp_img_for_size:
                            current_img_width, current_img_height = temp_img_for_size.size
                    
                    if CREATE_DEBUG_IMAGES: # 调试图像相关设置
                        debug_graphics_output_dir = os.path.join(output_base_dir, "debug_images", sub_dir_name)
                        os.makedirs(debug_graphics_output_dir, exist_ok=True)
                        page_num_match_debug = re.search(r"page_(\d+)", image_file_key, re.IGNORECASE)
                        debug_filename_prefix = f"page_{page_num_match_debug.group(1)}_" if page_num_match_debug else f"{os.path.splitext(image_file_key)[0]}_"
                        debug_image_save_path = os.path.join(debug_graphics_output_dir, f"debug_{debug_filename_prefix}{os.path.splitext(image_file_key)[1]}.png")
                        if isinstance(enhanced_image_data, str): img_for_debug_obj = Image.open(enhanced_image_data).convert("RGB")
                        else: img_for_debug_obj = Image.open(io.BytesIO(enhanced_image_data)).convert("RGB")
                        draw_on_img = ImageDraw.Draw(img_for_debug_obj)

                    page = doc.new_page(width=current_img_width, height=current_img_height) # 创建PDF页面
                    if isinstance(enhanced_image_data, str): page.insert_image(page.rect, filename=enhanced_image_data) # 从路径插入图像
                    else: page.insert_image(page.rect, stream=enhanced_image_data) # 从字节流插入图像

                    x_scale, y_scale = 1.0, 1.0 # 缩放因子 (当前未使用，保留供将来扩展)

                    # --- 开始: 新的自动列检测和重排序逻辑 (Y轴重叠和水平计数法) ---
                    if combined_ocr_elements:
                        column_counts_at_y_levels = [] # 存储在不同Y层级检测到的列数
                        for i in range(len(combined_ocr_elements)):
                            box_i = combined_ocr_elements[i]
                            
                            # 查找与 box_i 垂直重叠的所有其他box
                            vertically_overlapping_boxes_indices = []
                            for j in range(len(combined_ocr_elements)):
                                box_j = combined_ocr_elements[j]
                                # 判断Y轴重叠条件: 两个box的Y轴投影有交集
                                if max(box_i["min_y"], box_j["min_y"]) < min(box_i["max_y"], box_j["max_y"]):
                                    vertically_overlapping_boxes_indices.append(j)
                            
                            # 获取实际重叠的box对象，并按其最小x坐标排序 (从左到右)
                            actual_overlapping_boxes = sorted(
                                [combined_ocr_elements[k] for k in vertically_overlapping_boxes_indices],
                                key=lambda b: b["min_x"]
                            )
                            
                            distinct_horizontal_count = 0 # 在此Y层级，水平方向上不重叠的box数量
                            if actual_overlapping_boxes:
                                distinct_horizontal_count = 1 # 至少有一个box
                                last_distinct_box_max_x = actual_overlapping_boxes[0]["max_x"]
                                for k_overlap in range(1, len(actual_overlapping_boxes)):
                                    current_overlap_box = actual_overlapping_boxes[k_overlap]
                                    # 如果当前box的左边缘在上一个独立box的右边缘的右侧 (允许5像素容差)
                                    if current_overlap_box["min_x"] > last_distinct_box_max_x - 5: 
                                        distinct_horizontal_count += 1
                                        last_distinct_box_max_x = current_overlap_box["max_x"]
                                    else: # 如果重叠，则扩展上一个独立box的右边界，以正确处理包含关系或紧密相邻的box
                                        last_distinct_box_max_x = max(last_distinct_box_max_x, current_overlap_box["max_x"])
                            column_counts_at_y_levels.append(distinct_horizontal_count)

                        detected_num_columns = 1 # 默认为单栏
                        if column_counts_at_y_levels:
                            filtered_counts = [c for c in column_counts_at_y_levels if c > 0] # 过滤掉可能的0计数（无box的Y层）
                            if filtered_counts:
                                # 取最常见的列数作为检测结果
                                detected_num_columns = Counter(filtered_counts).most_common(1)[0][0]
                        
                        # 根据检测到的列数，为每个元素分配 runtime_column_id
                        # 注意：这里的 current_img_width 是在循环外，针对当前图像的
                        column_width_approx = current_img_width / detected_num_columns if detected_num_columns > 0 else current_img_width
                        for elem in combined_ocr_elements:
                            elem_col_id = 0
                            if detected_num_columns > 0 : # 避免除以零
                                # 根据box中心点在哪一"等分列宽"区域内来分配列ID
                                elem_col_id = int(elem["box_center_x"] // column_width_approx)
                            # 确保列ID在有效范围内 [0, detected_num_columns - 1]
                            elem["runtime_column_id"] = min(max(0, elem_col_id), detected_num_columns - 1 if detected_num_columns > 0 else 0)
                    
                    # 定义排序键函数
                    def column_sort_key(element):
                        col_id = element.get("runtime_column_id", 0) # 获取动态计算的列ID，若无则默认为0
                        return (col_id, element["min_y"]) # 按 (列ID, Y坐标) 排序

                    sorted_ocr_elements = sorted(combined_ocr_elements, key=column_sort_key)
                    # --- 结束: 自动列检测和重排序逻辑 ---

                    for element_data in sorted_ocr_elements: # 遍历排序后的元素
                        polygon_coords_from_json = element_data["polygon"]
                        text_content = element_data["text"] # 重命名以避免与外部texts变量混淆
                        
                        if CREATE_DEBUG_IMAGES and img_for_debug_obj and draw_on_img: # 如果启用调试，绘制各种框体
                            original_poly_tuples = [tuple(p) for p in polygon_coords_from_json] # 原始OCR多边形(蓝色)
                            draw_on_img.polygon(original_poly_tuples, outline="blue", width=1)
                            scaled_polygon_for_viz = [[int(p[0]*x_scale), int(p[1]*y_scale)+y_offset] for p in polygon_coords_from_json] # 缩放/偏移后多边形(绿色)
                            scaled_poly_tuples_for_viz = [tuple(p) for p in scaled_polygon_for_viz]
                            draw_on_img.polygon(scaled_poly_tuples_for_viz, outline="green", width=1)
                            rect_for_pdf_insertion_viz = fitz.Rect(scaled_polygon_for_viz[0][0], scaled_polygon_for_viz[0][1], scaled_polygon_for_viz[2][0], scaled_polygon_for_viz[2][1]) # PyMuPDF 使用的矩形(红色)
                            draw_on_img.rectangle((rect_for_pdf_insertion_viz.x0, rect_for_pdf_insertion_viz.y0, rect_for_pdf_insertion_viz.x1, rect_for_pdf_insertion_viz.y1), outline="red", width=2)
                            draw_on_img.ellipse((rect_for_pdf_insertion_viz.tl.x-2, rect_for_pdf_insertion_viz.tl.y-2, rect_for_pdf_insertion_viz.tl.x+2, rect_for_pdf_insertion_viz.tl.y+2), fill="red", outline="red") # 标记插入点(红点)

                        # 实际用于PDF文本插入的矩形 (应用缩放和Y偏移)
                        current_scaled_polygon = [[int(p[0]*x_scale), int(p[1]*y_scale)+y_offset] for p in polygon_coords_from_json]
                        rect_for_pdf_insertion = fitz.Rect(current_scaled_polygon[0][0], current_scaled_polygon[0][1], current_scaled_polygon[2][0], current_scaled_polygon[2][1])
                        
                        fontsize = rect_for_pdf_insertion.height * 0.9 # 根据矩形高度估算字体大小
                        while True: # 动态调整字体大小以适应矩形宽度
                            text_width = fitz.get_text_length(text_content, fontname="china-s", fontsize=fontsize)
                            if text_width <= rect_for_pdf_insertion.width or fontsize <= 1: break
                            fontsize -= 1
                        fontsize = max(1, min(fontsize, 100)) # 限制字体大小在合理范围
                        
                        adjusted_insertion_y = rect_for_pdf_insertion.top_left.y + (fontsize * 0.85) # 调整Y插入点以优化基线对齐
                        new_insertion_point = fitz.Point(rect_for_pdf_insertion.top_left.x, adjusted_insertion_y)
                        # 插入不可见的、可搜索的文本
                        page.insert_text(new_insertion_point, text_content, fontname="china-s", fontsize=fontsize, color=(0,0,0), fill=(1,1,1), render_mode=3)

                    processed_pages += 1 # 成功处理页数加一
                    if CREATE_DEBUG_IMAGES and img_for_debug_obj: # 如果启用调试，保存调试图像
                        try: img_for_debug_obj.save(debug_image_save_path)
                        except Exception as e_save_debug: print_with_time(f"保存调试图像 {debug_image_save_path} 时出错: {e_save_debug}", color=Fore.RED, logger=logger, log_level=logging.ERROR)
                    
                    # 清理当前图片处理循环中的一些变量，释放内存
                    del combined_ocr_elements, sorted_ocr_elements 
                    del polygons # 原始的polygons和texts也应清理
                    del texts
                    del data
                    gc.collect()

                except Exception as e:  # 捕获处理单个JSON/图像时的错误
                    errors += 1; msg = f"处理 JSON/图像 {image_file_key} 时出错: {e}"; error_messages.append(msg); print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                finally: # 内层try的finally块
                    if CREATE_DEBUG_IMAGES and img_for_debug_obj: img_for_debug_obj.close() # 关闭调试图像对象
                # 清理增强图像数据（如果是字节流），在调试图像处理完毕后
                if not isinstance(enhanced_image_data, str) and 'enhanced_image_data' in locals() and enhanced_image_data is not None: del enhanced_image_data; gc.collect()
            
            # --- PDF 分块保存 (在每个块处理完毕后) ---
            try:
                if doc is not None: # 确保文档对象已创建
                    intermediate_output_dir = os.path.join(output_base_dir, "intermediate") # 中间文件输出目录
                    os.makedirs(intermediate_output_dir, exist_ok=True)
                    intermediate_pdf_path = os.path.join(intermediate_output_dir, f"{sub_dir_name}_chunk_{chunk_start // CHUNK_SIZE}.pdf")
                    doc.save(intermediate_pdf_path, garbage=4, deflate=True) # 保存PDF块，启用垃圾回收和压缩
                    print_with_time(f"中间PDF块已保存: {intermediate_pdf_path}", color=Fore.GREEN, logger=logger)
            except Exception as e:
                errors += 1; error_msg = f"保存中间PDF时出错: {e}"; error_messages.append(error_msg); print_with_time(error_msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
            finally: # 块处理的finally块
                if doc is not None: doc.close() # 关闭当前块的PDF文档对象
                del doc; gc.collect() #显式删除并回收内存
        except Exception as e: # 捕获整个块处理过程中的意外错误
            error_msg = f"分块处理过程中发生意外错误: {e}"
            print_with_time(error_msg, color=Fore.RED, logger=logger,log_level=logging.ERROR)
            if error_messages: error_msg = "\n".join(error_messages) + "\n" + error_msg # 合并已有的错误信息
            return processed_pages, time.time() - start_time, errors + 1, error_msg

    # --- 合并中间PDF (在所有块处理完毕后) ---
    final_doc_obj = None # 确保变量在finally中可被引用
    try:
        final_doc_obj = fitz.open() # 重命名以避免与之前的doc冲突，用于最终合并的PDF文档
        intermediate_dir = os.path.join(output_base_dir, "intermediate")
        if os.path.exists(intermediate_dir):
            intermediate_files = sorted(
                [f for f in os.listdir(intermediate_dir) if f.startswith(f"{sub_dir_name}_chunk_") and f.endswith(".pdf")],
                key=lambda x: int(x.split("_")[-1].split(".")[0]) # 按块编号排序中间PDF文件
            )
            for intermediate_file in intermediate_files:
                intermediate_path = os.path.join(intermediate_dir, intermediate_file)
                try:
                    with fitz.open(intermediate_path) as intermediate_doc: # 打开一个中间PDF块
                        final_doc_obj.insert_pdf(intermediate_doc) # 将其内容插入到最终文档中
                    os.remove(intermediate_path) # 成功插入后删除中间文件
                except Exception as e:
                    errors += 1; error_msg = f"插入中间PDF {intermediate_file} 时出错: {e}"; error_messages.append(error_msg); print_with_time(error_msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
            if not os.listdir(intermediate_dir): os.rmdir(intermediate_dir) # 如果中间目录为空，则删除它
        
        final_doc_obj.save(output_pdf_path, garbage=4, deflate=True) # 保存最终的PDF文档
        end_time = time.time(); duration = end_time-start_time
        print_with_time(f"最终PDF已创建: {output_pdf_path} 用时 {duration:.2f} 秒", color=Fore.GREEN, logger=logger)
    except Exception as e:
        errors += 1; error_msg = f"合并或保存最终PDF时出错: {e}"; error_messages.append(error_msg); print_with_time(error_msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
    finally:
        if final_doc_obj is not None: final_doc_obj.close() # 关闭最终的PDF文档对象
        del final_doc_obj; gc.collect()

    # --- 错误处理和文件移动 (在PDF保存之后) ---
    logger.info(f"已处理: {processed_pages} 页") # 记录已处理的页数
    duration = time.time() - start_time # 计算总时长
    logger.info(f"总用时: {duration:.2f} 秒")
    logger.info(f"PDF创建错误: {errors}") # 记录错误总数
    for msg in error_messages: logger.error(msg) # 记录所有收集到的错误信息

    if save_enhanced and errors > 0: # 注意: 此处的save_enhanced是函数参数，可能与全局配置不同
        # 如果启用了保存增强图像（通过函数参数判断）并且发生了错误，则移动相关文件到错误目录
        error_sub_dir = os.path.join(output_base_dir, "errors", sub_dir_name)
        os.makedirs(error_sub_dir, exist_ok=True)
        for error_line in error_messages:
            match = re.search(r"(page_\d+)", error_line) # 尝试从错误信息中匹配文件名 (如 page_0001)
            if match:
                error_file_base = match.group(1) 
                # 根据 ENHANCE_IMAGES 全局配置来确定源路径
                # (这里逻辑可能需要审视，因为错误文件移动与 save_enhanced 参数和 ENHANCE_IMAGES 全局配置都有关)
                # 简单处理：优先从增强目录找，再从原始目录
                possible_extensions = ['.png', '.jpg', '.jpeg']
                found_source = False
                for ext in possible_extensions:
                    error_file_name_img = error_file_base + ext
                    # 检查增强目录 (如果ENHANCE_IMAGES为True且enhanced_image_dir存在)
                    if ENHANCE_IMAGES and enhanced_image_dir and os.path.exists(os.path.join(enhanced_image_dir, error_file_name_img)):
                         src_path = os.path.join(enhanced_image_dir, error_file_name_img)
                         found_source = True; break
                    # 检查原始图像目录
                    elif os.path.exists(os.path.join(image_dir, error_file_name_img)):
                         src_path = os.path.join(image_dir, error_file_name_img)
                         found_source = True; break
                
                if found_source: # 如果找到了源图像文件
                    dst_path = os.path.join(error_sub_dir, os.path.basename(src_path))
                    try: os.rename(src_path, dst_path) # 移动图像文件
                    except Exception as e_mv: print_with_time(f"移动文件 {src_path} 至 {dst_path} 出错: {e_mv}", color=Fore.RED, logger=logger, log_level=logging.ERROR)

                # 尝试移动对应的JSON文件
                json_file_to_move_match = re.search(r"page_(\d+)", error_file_base) # 确保是page_ddd格式
                if json_file_to_move_match:
                    json_page_num = int(json_file_to_move_match.group(1))
                    error_json_file = f"page_{json_page_num:04}_result.json"
                    src_json_path = os.path.join(json_dir, error_json_file)
                    if os.path.exists(src_json_path):
                        dst_json_path = os.path.join(error_sub_dir, error_json_file)
                        try: os.rename(src_json_path, dst_json_path) # 移动JSON文件
                        except Exception as e_mv_json: print_with_time(f"移动JSON文件 {src_json_path} 至 {dst_json_path} 出错: {e_mv_json}", color=Fore.RED, logger=logger, log_level=logging.ERROR)
    
    error_messages.clear() # 清空错误信息列表，为下一个子目录处理做准备
    if 'enhanced_paths' in locals(): del enhanced_paths # 清理 enhanced_paths 字典，释放内存
    gc.collect()
    return total_pages, duration, errors, "\n".join(error_messages) # 返回时error_messages已清空，这里返回的是空字符串



def main(ocr_results_dir=OCR_RESULTS_DIR, image_base_dir=IMAGE_BASE_DIR, output_base_dir=OUTPUT_BASE_DIRECTORY,
         y_offset=Y_OFFSET, num_processes=NUM_PROCESSES, save_enhanced_images_param=SAVE_ENHANCED_IMAGES): # 主函数，编排PDF创建流程。参数名用_param区分以避免与全局变量混淆。
    os.makedirs(output_base_dir, exist_ok=True) # 创建输出根目录
    log_dir = os.path.join(output_base_dir, "logs") # 日志子目录
    os.makedirs(log_dir, exist_ok=True)
    main_logger = setup_logger(log_dir, "main") # 设置主日志记录器

    sub_dirs_to_process = [] # 存储待处理的子目录信息元组 (子目录名, JSON路径, 图像路径)
    overall_start_time = time.time() # 程序总起始时间
    print_with_time(f"程序启动。", color=Fore.CYAN, logger=main_logger)

    # 遍历OCR结果目录，查找有效的子目录
    for sub_dir_name in os.listdir(ocr_results_dir):
        sub_dir_path = os.path.join(ocr_results_dir, sub_dir_name) # OCR结果子目录的完整路径 (包含JSON文件)
        if os.path.isdir(sub_dir_path):
            image_dir = os.path.join(image_base_dir, sub_dir_name) # 对应的图像子目录的完整路径
            if os.path.exists(image_dir): # 确保对应的图像子目录存在
                sub_dirs_to_process.append((sub_dir_name, sub_dir_path, image_dir))
            else:
                print_with_time(f"警告: OCR结果目录 {sub_dir_path} 存在，但未找到对应的图像目录 {image_dir}。已跳过。", color=Fore.YELLOW, logger=main_logger, log_level=logging.WARNING)


    print_with_time(f"待处理的子目录数量: {len(sub_dirs_to_process)}", color=Fore.CYAN, logger=main_logger)

    # --- 计算预计待处理的总图像数量 (用于ETA估算) ---
    total_image_count = 0
    for sub_dir_name_count, _, image_dir_count in sub_dirs_to_process:
        # 图像计数应基于实际要处理的图像（考虑是否增强）
        # 如果开启增强，并且增强后的图片保存在单独目录，则应统计增强目录
        # 但目前 process_images_in_directory 返回的是 enhanced_paths 字典或原始路径字典
        # 更准确的计数是统计 image_dir 中的原始图像，因为无论是否增强，都会基于这些原始图像
        # process_and_create_pdfs 中的 total_pages 是基于 image_files_or_json_keys 长度
        # 所以这里也应该类似地统计
        
        # 简化：直接统计原始图像目录中的图片数量
        # 注意：如果ENHANCE_IMAGES为True但某些子目录的图像增强失败，这里的计数可能与实际处理的略有不同
        # 但作为预估ETA的基数是可以的
        try:
            count = len([f for f in os.listdir(image_dir_count) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            total_image_count += count
        except FileNotFoundError:
            print_with_time(f"警告: 预估总图像数时，目录 {image_dir_count} 未找到。此目录可能已被跳过或处理失败。", color=Fore.YELLOW, logger=main_logger, log_level=logging.WARNING)


    print_with_time(f"预计待处理的总图像数量: {total_image_count}", color=Fore.CYAN, logger=main_logger)

    # 使用进程池并行处理子目录
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_and_create_pdfs, sd_name, sd_path, img_dir,
                                    output_base_dir, y_offset, save_enhanced_images_param, # 使用参数名以区分全局配置
                                    setup_logger(log_dir, f"process_{sd_name}")) # 为每个子进程创建单独的日志记录器
                   for sd_name, sd_path, img_dir in sub_dirs_to_process]

        total_processed_pages_agg = 0 # 聚合的总计划页数 (所有子任务计划处理的页数之和)
        processed_pdfs_count = 0 # 已完成的PDF（子目录）计数，用于进度条
        pdf_creation_start_time = time.time() # PDF创建阶段的开始时间
        spinner = create_spinner() # 创建旋转动画

        for future in concurrent.futures.as_completed(futures): # 迭代已完成的future对象
            print_with_spinner(spinner) # 打印旋转动画
            try:
                # process_and_create_pdfs 返回 (total_pages_in_subdir, duration, errors, error_messages_str)
                # 我们需要的是 total_pages_in_subdir，它代表这个子目录计划处理的页数
                pages_in_pdf_from_result, _, _, _ = future.result() # 获取子任务的结果
                total_processed_pages_agg += pages_in_pdf_from_result # 累加的是子目录的计划总页数
                processed_pdfs_count += 1 # 已完成的PDF（子目录）数量加一

                # ---改进的ETA计算---
                elapsed_time = time.time() - pdf_creation_start_time # 已用时间
                if total_processed_pages_agg > 0: # 避免除以零
                    # ETA计算基于已处理的PDF所代表的总页数 与 总预估页数
                    # 注意: total_image_count 是预估，total_processed_pages_agg 是基于已完成的子任务的计划页数累加
                    # 如果一个子任务完成了，意味着它的所有页都已"认领"处理，无论成功与否
                    avg_time_per_page = elapsed_time / total_processed_pages_agg if total_processed_pages_agg > 0 else 0 # 每页平均处理时间
                    remaining_pages = total_image_count - total_processed_pages_agg # 剩余页数
                    estimated_remaining_time = avg_time_per_page * remaining_pages if remaining_pages > 0 else 0 # 预计剩余时间（秒）
                    eta_datetime = datetime.now() + timedelta(seconds=estimated_remaining_time if estimated_remaining_time > 0 else 0) # 预计完成的绝对时间
                    current_speed = total_processed_pages_agg / elapsed_time if elapsed_time > 0 else 0 # 当前处理速度 (页/秒)
                    eta_str_val = eta_datetime.strftime('%H:%M:%S') # ETA格式化字符串
                else:
                    eta_str_val = "计算中..." # 如果还没有处理任何页面，显示计算中
                    current_speed = 0

                # 进度条 (基于已完成的PDF数量)
                progress = int(50 * processed_pdfs_count / len(sub_dirs_to_process)) if len(sub_dirs_to_process) > 0 else 0
                bar = f"[{'=' * progress}{' ' * (50 - progress)}]"
                eta_color = Fore.MAGENTA if eta_str_val != "计算中..." else Fore.YELLOW # ETA显示颜色
                
                # 打印进度信息到控制台
                sys.stdout.write(
                    f"\r{get_timestamp()} - PDF 创建: {bar} {processed_pdfs_count}/{len(sub_dirs_to_process)} "
                    f"| 已用时间: {Fore.BLUE}{timedelta(seconds=int(elapsed_time))}{Style.RESET_ALL} "
                    f"| 预计剩余: {eta_color}{eta_str_val}{Style.RESET_ALL} "
                    f"| 速度: {Fore.YELLOW}{current_speed:.2f} 页/秒{Style.RESET_ALL} "
                )
                sys.stdout.flush()
            except Exception as e:
                print_with_time(f"处理子目录时发生主循环错误: {e}", color=Fore.RED, logger=main_logger, log_level=logging.ERROR)

    overall_end_time = time.time() # 程序总结束时间
    overall_duration = overall_end_time - overall_start_time # 程序总运行时长
    # 平均速度应基于实际成功处理的页数，但这难以从 process_and_create_pdfs 的返回值直接获得
    # 这里使用 total_processed_pages_agg 作为总页数计算，它代表了所有子任务计划处理的总页数
    average_speed = total_processed_pages_agg / overall_duration if overall_duration > 0 else 0
    print_with_time(f"\n总处理时间: {overall_duration:.2f} 秒", color=Fore.CYAN, logger=main_logger)
    print_with_time(f"聚合的总计划页数: {total_processed_pages_agg}", color=Fore.CYAN, logger=main_logger) # 改为聚合的计划页数
    print_with_time(f"平均速度: {average_speed:.2f} 页/秒", color=Fore.CYAN, logger=main_logger)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_with_time("用户中断。正在退出。", color=Fore.RED)
    except Exception as e_global: # 捕获其他所有未处理的全局异常
        # 为 main 函数的任何未捕获异常添加日志记录
        # (假设 main_logger 可能未初始化，所以直接打印)
        print_with_time(f"程序发生未处理的全局错误: {e_global}", color=Fore.RED)
        # 如果有logger实例，则使用它 (这部分代码块可以取消注释，如果需要更详细的traceback记录到日志文件)
        # global_logger = logging.getLogger("main") # 尝试获取主日志记录器
        # if global_logger and global_logger.hasHandlers(): # 检查记录器是否有效并有处理器
        #    global_logger.error(f"程序发生未处理的全局错误: {e_global}", exc_info=True) # exc_info=True 会记录完整的堆栈跟踪
        # else: # 如果日志记录器不可用，则回退到标准traceback打印
        #    import traceback
        #    traceback.print_exc()
