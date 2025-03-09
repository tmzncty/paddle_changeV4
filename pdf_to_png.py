import fitz
import os
import time
from multiprocessing import Pool, Manager

def render_page(args):
    """
    渲染单个PDF页面为图片。

    Args:
        args (tuple): 包含页面编号、PDF路径和输出文件夹的元组。

    Returns:
        str: 图片文件的路径，如果渲染失败则返回None。
    """
    page_num, pdf_path, output_folder = args
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        image_path = os.path.join(output_folder, f"page_{page_num + 1:04}.png")  # 格式化文件名
        pix.save(image_path)
        doc.close()
        return image_path
    except Exception as e:
        print(f"Error rendering page {page_num + 1}: {e}")
        return None


def pdf_to_images_multiprocess(pdf_path, output_folder, num_processes=16):
    """
    使用多进程将PDF的每一页转换为图片。

    Args:
        pdf_path (str): PDF文件的路径。
        output_folder (str): 存放图片的文件夹。
        num_processes (int): 使用的进程数。

    Returns:
        list: 包含所有图片路径的列表。
    """
    os.makedirs(output_folder, exist_ok=True)  # 确保输出文件夹存在
    image_paths = []
    try:
        doc = fitz.open(pdf_path)
        num_pages = doc.page_count

        with Pool(processes=num_processes) as pool:
            # 创建一个包含所有页面渲染任务的列表
            tasks = []
            for page_num in range(num_pages):
                # 注意这里，将doc.load_page(page_num)改成load_page, fitz.open(pdf_path),这样可以避免在子进程中出现问题
                tasks.append((page_num, pdf_path, output_folder))
            # 使用imap_unordered进行并行处理
            results = pool.imap_unordered(render_page, tasks)

            start_time = time.time()
            for i, result in enumerate(results):
                if result:
                    image_paths.append(result)
                # 进度显示
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


def process_all_pdfs(pdf_folder, base_output_folder, num_processes=16):
    """
    处理一个文件夹中的所有PDF文件，每个PDF文件生成一个单独的图片文件夹。

    Args:
        pdf_folder (str): 包含PDF文件的文件夹路径。
        base_output_folder (str): 存放所有输出图片文件夹的根目录。
        num_processes (int): 使用的进程数。
    """
    
    # 确保输出文件夹存在
    if not os.path.exists(base_output_folder):
      os.makedirs(base_output_folder)
    
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            # 为每个PDF创建一个单独的输出文件夹
            pdf_name = os.path.splitext(pdf_file)[0]  # 获取PDF文件名（不含扩展名）
            output_folder = os.path.join(base_output_folder, pdf_name)
            print(f"Processing PDF: {pdf_file}")
            pdf_to_images_multiprocess(pdf_path, output_folder, num_processes)


if __name__ == "__main__":
    pdf_folder = "/media/tmzn/DATA5/ocr_paddle/tmppdf"  # PDF文件夹路径
    base_output_folder = "/media/tmzn/DATA5/ocr_paddle/tmpicture"  # 输出文件夹路径
    num_processes = 8  # 根据您的CPU核心数调整
    process_all_pdfs(pdf_folder, base_output_folder, num_processes)