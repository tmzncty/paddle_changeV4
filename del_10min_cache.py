import os
import time
import threading
from datetime import datetime, timedelta

# 要监控和清理的目录
TARGET_DIR = "/media/tmzn/DATA5/paddlex_cache_home"
# 文件过期时间（分钟）
EXPIRATION_MINUTES = 10
# 扫描间隔时间（秒）
SCAN_INTERVAL_SECONDS = 60
# 删除操作的最大线程数
MAX_THREADS = 5  # 你可以根据你的CPU核心数和IO性能调整这个值

def get_files_to_delete(directory, expiration_minutes):
    """
    获取目录下所有超过指定过期时间的文件列表。
    """
    files_to_delete = []
    now = datetime.now()
    expiration_delta = timedelta(minutes=expiration_minutes)

    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                # 获取文件的最后修改时间
                modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                if now - modified_time > expiration_delta:
                    files_to_delete.append(filepath)
            except FileNotFoundError:
                # 文件可能在扫描过程中被删除
                print(f"警告：文件 {filepath} 在扫描时未找到，可能已被删除。")
            except Exception as e:
                print(f"错误：获取文件 {filepath} 修改时间失败：{e}")
    return files_to_delete

def delete_file(filepath):
    """
    删除单个文件。
    """
    try:
        os.remove(filepath)
        print(f"已删除：{filepath}")
    except FileNotFoundError:
        print(f"警告：尝试删除文件 {filepath} 时未找到，可能已被其他进程删除。")
    except OSError as e:
        print(f"错误：删除文件 {filepath} 失败：{e}")
    except Exception as e:
        print(f"未知错误：删除文件 {filepath} 时发生错误：{e}")

def delete_files_worker(files_chunk):
    """
    线程工作函数，删除一部分文件。
    """
    for filepath in files_chunk:
        delete_file(filepath)

def main():
    """
    主函数，持续监控和删除文件。
    """
    print(f"开始监控目录：{TARGET_DIR}")
    print(f"将删除超过 {EXPIRATION_MINUTES} 分钟未修改的文件。")
    print(f"每隔 {SCAN_INTERVAL_SECONDS} 秒扫描一次。")
    print(f"使用最多 {MAX_THREADS} 个线程进行删除操作。")
    print("按 Ctrl+C 退出程序。")

    try:
        while True:
            print(f"\n[{datetime.now()}] 开始扫描...")
            files_to_remove = get_files_to_delete(TARGET_DIR, EXPIRATION_MINUTES)

            if not files_to_remove:
                print("没有找到需要删除的文件。")
            else:
                print(f"找到 {len(files_to_remove)} 个需要删除的文件。")
                
                # 将文件列表分块，以便多线程处理
                # 计算每个线程大致处理的文件数量
                num_files = len(files_to_remove)
                chunk_size = (num_files + MAX_THREADS - 1) // MAX_THREADS  # 向上取整

                threads = []
                for i in range(0, num_files, chunk_size):
                    files_chunk = files_to_remove[i:i + chunk_size]
                    if files_chunk: # 确保块不为空
                        thread = threading.Thread(target=delete_files_worker, args=(files_chunk,))
                        threads.append(thread)
                        thread.start()
                
                # 等待所有删除线程完成
                for thread in threads:
                    thread.join()
                
                print("本轮删除操作完成。")

            print(f"等待 {SCAN_INTERVAL_SECONDS} 秒后进行下一次扫描...")
            time.sleep(SCAN_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n程序被用户中断。正在退出...")
    except Exception as e:
        print(f"发生未预料的错误：{e}")
    finally:
        print("程序已停止。")

if __name__ == "__main__":
    if not os.path.isdir(TARGET_DIR):
        print(f"错误：目录 '{TARGET_DIR}' 不存在或不是一个有效的目录。请检查路径。")
    else:
        main()
