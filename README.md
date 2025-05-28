# paddle_change
把paddle改的更好用，最大限度利用硬件资源。
# NOTICE
**！！！**
**最好在ubuntu22.04+CUDA=11.8 Python 3.9.21的情况下**

**这个东西可能对你的CPU GPU 内存 磁盘造成严重负载（大概就是总有一个到达瓶颈）。**

**请小心使用，记得看温度，使用btop和nvitip做好监测，防止烧毁硬件。**

**同时对磁盘的4K读写及寿命要求极高，很容易出现谈笑之间写了一个T的情况。如果是一般消费级磁盘请谨慎使用。（谁用消费级磁盘处理大量数据啊？）**


# conda初始化
对于windows
set-ExecutionPolicy RemoteSigned
A


# 主要功能
## 0. pdf拆图
请使用pdf_to_png.py完成

## 1. 大量图片进行ocr处理
请使用highocr3_f2.py完成。
## 2. 制作可以搜索的pdf

请使用pdf_creator_with_text_layer5.py完成。
## 3. 直接pdf进行ocr
请使用highocr3_f2_pdf2.py完成

--- 
# 代码说明



# 0. PDF拆图


## PDF拆图工具使用说明 

这个工具可以帮助你把 **PDF文件** 变成一张张 **图片**。  如果你有一些PDF文件，想把它们里面的每一页都变成图片（比如PNG格式），这个工具就非常有用。 它可以一次性处理很多PDF文件，并且速度很快。

**1. 准备工作：安装工具**

首先，你的电脑需要安装一个叫做 **Python** 的程序，并且安装一个叫做 **PyMuPDF** 的“小工具”。  不用担心，安装过程很简单，就像给手机安装APP一样。

  * **检查Python:**  大部分电脑可能已经安装了Python。你可以打开电脑的 **命令提示符** (Windows 用户) 或者 **终端** (Mac/Linux 用户)，输入 `python --version`  或者  `python3 --version`  然后按下回车键。 如果显示了Python的版本号，就说明你已经安装了Python，可以跳过安装Python的步骤。 如果提示找不到命令，就需要先安装Python。

      * **如果需要安装Python:**  你可以去Python官网下载安装程序：[https://www.python.org/downloads/](https://www.google.com/url?sa=E&source=gmail&q=https://www.python.org/downloads/)  下载最新版本的Python安装包，然后按照提示一步一步安装就可以了。  安装的时候注意勾选 "Add Python to PATH" 选项，这样方便在命令行中使用Python。

  * **安装 PyMuPDF:**  安装好Python之后，继续在 **命令提示符** 或 **终端** 中输入下面这条命令，然后按下回车键：

    ```bash
    pip install pymupdf
    ```

    或者 (如果 `pip` 命令不行，尝试使用 `pip3`)

    ```bash
    pip3 install pymupdf
    ```

    这条命令会自动从网上下载并安装 PyMuPDF 这个“小工具”。  安装过程中可能会显示一些信息，等待安装完成即可。  如果看到 "Successfully installed PyMuPDF" 这样的提示，就说明安装成功了。

**2. 下载和放置代码文件**

你需要把上面你看到的那些代码（从 `# 0. PDF拆图` 到 `process_all_pdfs(pdf_folder, base_output_folder, num_processes)`  以及最后的 `if __name__ == "__main__":` 部分）复制下来，保存到一个文本文件里。  你可以使用电脑自带的 **记事本** (Windows) 或者 **文本编辑器** (Mac)。

保存的时候，注意把文件后缀名改成 `.py`，例如你可以把文件命名为 `pdf_to_image.py`  或者 `pdf_converter.py`  等等，文件名你可以自己取，但是 **后缀名一定要是 `.py`** 。  保存文件的位置你也可以自己选择，比如你可以保存在桌面或者你常用的文件夹里。

**3. 准备PDF文件和设置输出文件夹**

  * **PDF文件夹:**  你需要准备一个文件夹，里面存放你想要转换成图片的 **PDF文件**。  比如你可以新建一个文件夹，命名为 `我的PDF文件`，然后把你想要处理的所有PDF文件都放进去。  **记住这个文件夹的路径**，后面要用到。  例如，如果你的 "我的PDF文件" 文件夹放在桌面上，那么它的路径可能是类似于  `C:\Users\你的用户名\Desktop\我的PDF文件` (Windows) 或者 `/Users/你的用户名/Desktop/我的PDF文件` (Mac)。  **路径要写完整**。

  * **输出文件夹:**  你还需要准备一个 **空的文件夹**，用来存放转换出来的图片。  比如你可以再新建一个文件夹，命名为 `输出图片`。  **同样要记住这个文件夹的路径**。  例如，如果你的 "输出图片" 文件夹也放在桌面上，那么它的路径可能是类似于 `C:\Users\你的用户名\Desktop\输出图片` (Windows) 或者 `/Users/你的用户名/Desktop/输出图片` (Mac)。 **这个文件夹一开始必须是空的，程序会自动在里面创建子文件夹来存放图片。**

**4. 修改代码中的路径**

打开你刚刚保存的 `.py` 代码文件 (比如 `pdf_to_image.py`)。  在代码中，你会看到下面这两行：

```python
pdf_folder = "/media/tmzn/DATA5/ocr_paddle/tmppdf"  # PDF文件夹路径
base_output_folder = "/media/tmzn/DATA5/ocr_paddle/tmpicture"  # 输出文件夹路径
```

你需要把这两行代码中的路径 **替换成你电脑上实际的 PDF文件夹路径 和 输出文件夹路径**。

  * 把 `pdf_folder = "/media/tmzn/DATA5/ocr_paddle/tmppdf"`  改成  `pdf_folder = "你的PDF文件夹的路径"`  ，  **把双引号里面的内容替换成你上面准备好的 PDF文件夹的完整路径**。
  * 把 `base_output_folder = "/media/tmzn/DATA5/ocr_paddle/tmpicture"`  改成  `base_output_folder = "你的输出文件夹的路径"`  ， **把双引号里面的内容替换成你上面准备好的 输出文件夹的完整路径**。

**注意:**  路径要用 **英文双引号**  `"`  括起来。  路径中如果包含反斜杠 `\` (Windows 路径)，在Python代码中可能需要写成双反斜杠 `\\` 或者用正斜杠 `/` 代替。  例如，Windows路径 `C:\Users\User\Documents\PDFs`  在代码中可以写成  `"C:\\Users\\User\\Documents\\PDFs"`  或者 `"C:/Users/User/Documents/PDFs"`。  Mac 和 Linux 路径一般用正斜杠 `/`。

改完之后，**保存你的代码文件**。

**5. 运行代码，开始转换**

打开 **命令提示符** (Windows) 或者 **终端** (Mac/Linux)。

1.  **切换到代码文件所在的目录:**  使用 `cd` 命令。  例如，如果你的 `pdf_to_image.py` 文件保存在桌面，你可能需要先输入 `cd Desktop` (Windows)  或者 `cd ~/Desktop` (Mac/Linux)  ，然后按下回车键。  如果你的代码文件保存在其他文件夹，就需要使用 `cd` 命令切换到相应的文件夹。

2.  **运行代码:**  在命令提示符或终端中输入下面这条命令，然后按下回车键：

    ```bash
    python pdf_to_image.py
    ```

    或者 (如果 `python` 命令不行，尝试使用 `python3`)

    ```bash
    python3 pdf_to_image.py
    ```

    （请把 `pdf_to_image.py` 替换成你实际保存的代码文件名）。

    按下回车后，程序就开始运行了。  你会看到命令行窗口中显示转换进度，例如：

    ```
    Processing PDF: example1.pdf
    PDF to Image Progress: [10.00%] 1/10 pages, Speed: 0.50 pages/sec, Elapsed: 2.00 sec
    PDF to Image Progress: [20.00%] 2/10 pages, Speed: 0.67 pages/sec, Elapsed: 3.00 sec
    ...
    PDF to Image conversion complete.
    Processing PDF: example2.pdf
    ...
    ```

    程序会一个接一个地处理你 PDF文件夹 中的所有 PDF 文件。  每处理完一个 PDF 文件，就会在你的 **输出文件夹** 中创建一个以 PDF 文件名命名的文件夹，然后把这个 PDF 文件转换成的图片都放在这个新文件夹里。

**6. 查看转换结果**

等待程序运行完成 (当命令行窗口不再显示进度信息，并且回到可以输入命令的状态时，就表示程序运行完了)。  打开你之前设置的 **输出文件夹**。  你会看到，里面多了一些文件夹，每个文件夹的名字都和你处理的 PDF 文件名对应。  打开这些文件夹，就可以看到 PDF 文件转换成的 PNG 图片了。  图片文件会按照页码顺序命名，例如 `page_0001.png`, `page_0002.png`  等等。

**7.  关于 "进程数" (num\_processes)**

在代码中，你还会看到这样一行：

```python
num_processes = 8  # 根据您的CPU核心数调整
```

这个 `num_processes = 8`  的意思是程序在转换 PDF 文件的时候，会同时使用 8 个“小帮手”一起工作，这样可以加快转换速度。  `8`  这个数字你可以根据你的电脑配置进行调整。  一般来说，如果你的电脑配置比较好 (CPU 核心数比较多)，可以适当调大这个数字，例如 12 或者 16。  如果电脑配置一般，或者不确定，保持默认的 `8`  或者设置成更小的数字 (例如 4 或者 2) 也可以。  对于普通用户来说，不修改这个数字通常就可以了。

**常见问题和注意事项:**

  * **路径错误:**  最常见的问题是 PDF文件夹路径 或者 输出文件夹路径 设置错误。  请仔细检查你代码中修改的路径是否正确，路径是否写完整，文件夹是否真的存在。
  * **权限问题:**  如果程序运行出错，提示权限不足，可能是因为程序没有权限访问你的 PDF文件夹 或者 输出文件夹。  你可以尝试把 PDF文件夹 和 输出文件夹 放在一个你确定有读写权限的位置，例如桌面或者 "文档" 文件夹。
  * **缺少依赖库:**  如果在运行代码的时候提示缺少 `fitz` 库，请再次检查你是否成功安装了 PyMuPDF (执行 `pip install pymupdf` 命令)。
  * **文件名乱码:**  如果 PDF 文件名或者输出文件夹名包含中文或者特殊字符，可能会出现乱码问题。  尽量使用英文文件名和文件夹名可以减少出现问题的可能性。
  * **内存占用:**  转换大型 PDF 文件或者同时处理很多 PDF 文件可能会占用较多内存。  如果你的电脑内存较小，可能会出现卡顿或者程序崩溃的情况。  可以尝试减少 `num_processes` 的值，或者一次性处理的 PDF 文件数量不要太多。








## highocr3_f2.py 图片文字识别工具使用说明

这个工具可以帮助你识别 **图片中的文字**，并且把识别结果保存下来。  如果你有一些图片，想提取出里面的文字（比如扫描件、截图等），这个工具就非常适合。 它可以一次性处理很多图片，并且利用电脑的多核处理器来加速识别。

**1. 准备工作：安装工具**

和之前的 PDF 拆图工具类似，你需要先安装一些“小工具”才能让这个文字识别工具工作起来。  这次需要安装的是 **Python**, **PaddlePaddle**, **PaddleX**, **PyYAML** 和 **Pillow (PIL)**。

* **检查Python:**  跟之前一样，先检查你的电脑有没有安装 Python。打开 **命令提示符** (Windows) 或者 **终端** (Mac/Linux)，输入 `python --version` 或 `python3 --version`，看看有没有显示 Python 的版本号。  如果已经安装了，跳过安装 Python 的步骤。  如果需要安装，参考之前的 PDF 拆图工具的说明进行安装。

* **安装 PaddlePaddle, PaddleX, PyYAML, Pillow:**  安装好 Python 之后，在 **命令提示符** 或 **终端** 中依次输入以下命令，**每输入一条命令后都按下回车键**：

    ```bash
    pip install paddlepaddle
    pip install paddlex
    pip install pyyaml
    pip install Pillow
    ```

    或者 (如果 `pip` 命令不行，尝试使用 `pip3`)

    ```bash
    pip3 install paddlepaddle
    pip3 install paddlex
    pip3 install pyyaml
    pip3 install Pillow
    ```

    这些命令会分别安装 PaddlePaddle (百度的深度学习框架，文字识别的核心), PaddleX (PaddlePaddle 的一个工具包，简化开发流程), PyYAML (用来读取配置文件), 和 Pillow (用来处理图片)。  安装过程中可能会显示很多信息，等待安装完成。 如果看到 "Successfully installed ..." 这样的提示，就说明安装成功了。

**2. 下载代码和配置文件**

你需要把上面你看到的那些代码（从 `# import paddlex as pdx` 到 `if __name__ == "__main__":` 部分）复制下来，保存到一个文本文件里，后缀名改成 `.py`，例如 `image_ocr.py`。  保存位置随意。

除了代码文件，这个工具还需要一个 **配置文件**  `OCR.yaml`。  **你需要自己创建一个名为 `OCR.yaml` 的文件**，并且把它放在和代码文件 (`image_ocr.py`) **同一个文件夹** 里。

* **`OCR.yaml` 文件的内容:**  `OCR.yaml` 文件告诉程序如何进行文字识别。  **对于初次使用的用户，你可以先使用默认的配置。**  你可以从 PaddleOCR 的官方网站或者 PaddleX 的文档中找到 `OCR.yaml` 的示例文件。  最简单的，你可以先创建一个空的文本文件，然后命名为 `OCR.yaml`， 暂时不需要修改里面的内容。  (更高级的用户可以根据自己的需求调整 `OCR.yaml` 里面的参数，例如选择不同的识别模型，修改识别精度等等，但对于小白用户，默认配置通常就够用了)。  **请务必确保 `OCR.yaml` 文件和 `image_ocr.py` 文件在同一个文件夹下。**

**3. 准备图片文件夹和设置输出文件夹**

  * **图片文件夹:**  你需要准备一个文件夹，里面存放你想要识别文字的 **图片文件**。  支持常见的图片格式，例如 JPG, JPEG, PNG, BMP。  比如你可以新建一个文件夹，命名为 `待识别图片`，然后把你的图片都放进去。  **记住这个文件夹的路径**。

  * **输出文件夹:**  你需要准备一个 **空的文件夹**，用来存放识别结果。  程序会把识别出来的文字信息保存成 JSON 文件放到这里。  比如你可以新建一个文件夹，命名为 `OCR结果`。  **同样要记住这个文件夹的路径**。  **这个文件夹一开始必须是空的。**

**4. 修改代码中的路径**

打开你保存的 `.py` 代码文件 (比如 `image_ocr.py`)。  在代码中找到以下几行：

```python
config_path = "/media/tmzn/DATA5/ocr_paddle/config_paddle/OCR.yaml"
image_root_dir = "/media/tmzn/DATA5/music_picture/"
output_root_dir = "/media/tmzn/DATA5/ocr_paddle/output_music_picture_ocr_results"
log_and_error_dir = "/media/tmzn/DATA5/ocr_paddle/ocr_logs_and_errors"
```

你需要修改这几行代码，把路径 **替换成你电脑上实际的 配置文件路径, 图片文件夹路径 和 输出文件夹路径**。

* **`config_path = "/media/tmzn/DATA5/ocr_paddle/config_paddle/OCR.yaml"`:**  **重要:**  因为我们把 `OCR.yaml` 文件和 `image_ocr.py` 文件放在了同一个文件夹，所以这里 **直接改成下面这样就可以了**：

  ```python
  config_path = "OCR.yaml"
  ```
  这样程序就能在代码文件所在的文件夹里找到 `OCR.yaml` 配置文件了。  **如果你把 `OCR.yaml` 文件放在了其他地方，才需要写完整的路径。**

* **`image_root_dir = "/media/tmzn/DATA5/music_picture/"`:**  改成  `image_root_dir = "你的图片文件夹的路径"`  ，**把双引号里面的内容替换成你上面准备好的 图片文件夹的完整路径**。

* **`output_root_dir = "/media/tmzn/DATA5/ocr_paddle/output_music_picture_ocr_results"`:**  改成  `output_root_dir = "你的输出文件夹的路径"`  ， **把双引号里面的内容替换成你上面准备好的 输出文件夹的完整路径**。

* **`log_and_error_dir = "/media/tmzn/DATA5/ocr_paddle/ocr_logs_and_errors"`:** 这个是用来存放日志和错误图片的文件夹，你可以根据需要修改，或者保持默认。 如果要修改，改成 `log_and_error_dir = "你想要存放日志和错误的文件夹路径"`。

**路径的写法和注意事项**  与之前的 PDF 拆图工具说明中的 **第4步**  相同，请参考之前的说明。  修改完成后，**保存你的代码文件**。

**5. 运行代码，开始文字识别**

打开 **命令提示符** (Windows) 或者 **终端** (Mac/Linux)。

1.  **切换到代码文件所在的目录:**  使用 `cd` 命令。 确保你切换到的是 `image_ocr.py` 和 `OCR.yaml` 文件所在的文件夹。

2.  **运行代码:**  在命令提示符或终端中输入下面这条命令，然后按下回车键：

    ```bash
    python image_ocr.py
    ```

    或者 (如果 `python` 命令不行，尝试使用 `python3`)

    ```bash
    python3 image_ocr.py
    ```

    （请把 `image_ocr.py` 替换成你实际保存的代码文件名）。

    按下回车后，程序就开始运行了。  你会看到命令行窗口中显示识别进度，例如：

    ```
    [2025-02-12 17:00:00] OCR process started.
    [2025-02-12 17:00:01] Using 8 processes.
    [2025-02-12 17:00:01] Batch size: 64
    [2025-02-12 17:00:01] Total images to process: 100
    [2025-02-12 17:00:05] Processed 10/100 images... (0.400 seconds/image), ETA: 2025-02-12 17:00:40 (00:00:35), Errors: 0
    [2025-02-12 17:00:09] Processed 20/100 images... (0.420 seconds/image), ETA: 2025-02-12 17:00:42 (00:00:33), Errors: 0
    ...
    [2025-02-12 17:00:38] OCR results saved to: 你的输出文件夹的路径
    [2025-02-12 17:00:38] Total processing time: 37.50 seconds
    [2025-02-12 17:00:38] Average time per image: 0.375 seconds
    [2025-02-12 17:00:38] Total errors: 0
    ```

    程序会处理你 图片文件夹 中的所有图片文件，并显示处理进度、预计完成时间等等。

**6. 查看识别结果**

等待程序运行完成。  打开你之前设置的 **输出文件夹**。  你会看到，里面多了一些文件夹，这些文件夹的结构会和你的 **图片文件夹** 结构类似。  例如，如果你的图片文件夹是这样的:

```
待识别图片/
    文件夹1/
        image1.jpg
        image2.png
    文件夹2/
        image3.bmp
```

那么，在你的 **输出文件夹**  `OCR结果`  中，你会看到类似的结构:

```
OCR结果/
    文件夹1/
        image1_result.json
        image2_result.json
    文件夹2/
        image3_result.json
```

每个图片文件都会对应一个 `*_result.json` 文件。  打开这些 JSON 文件，就可以看到程序识别出来的文字信息了。  JSON 文件是一种文本格式，可以用文本编辑器或者专门的 JSON 查看器打开。  里面会包含识别出来的文字，以及文字在图片中的位置等等信息。

**7. 关于 "进程数" (num\_processes) 和 CPU/GPU 设置**

* **进程数 (num\_processes):**  代码中 `num_processes = max(1, cpu_count() - 16)`  这一行设置了程序使用的 “小帮手” (进程) 的数量。  和之前的 PDF 拆图工具类似，你可以根据你的电脑 CPU 核心数调整这个数字。  如果电脑配置好，可以适当增大，配置一般，可以减小。  默认设置通常就够用。

* **CPU/GPU 设置:**  代码默认情况下会 **尝试使用 GPU 进行加速** (如果你的电脑有 NVIDIA 显卡并且安装了 CUDA 环境)。  如果你想 **强制使用 CPU 进行识别**，可以修改代码中的 `use_cpu = False`  为  `use_cpu = True`。  如果设置为 `use_cpu = True`，程序会自动修改 `OCR.yaml` 配置文件，创建一个 CPU 版本的配置文件 ( `OCR_cpu.yaml` )  并使用它。  **通常建议使用 GPU 加速，速度会快很多。**  如果你的电脑没有 GPU，或者 GPU 显存不足，再考虑使用 CPU。

**常见问题和注意事项:**

  * **配置文件 `OCR.yaml`:**  **务必确保 `OCR.yaml` 文件和 `image_ocr.py` 在同一个文件夹下，并且 `config_path` 变量设置正确 (通常设置为 `"OCR.yaml"` 即可)。**  配置文件的内容会影响识别效果，更高级的用户可以尝试修改配置文件来优化识别结果。
  * **路径错误, 权限问题, 缺少依赖库, 文件名乱码, 内存占用:**  这些问题和注意事项与之前的 PDF 拆图工具类似，请参考之前的说明。
  * **识别精度:**  图片文字识别是一个复杂的技术，受到图片质量、文字清晰度、字体、语言等等多种因素的影响。  对于一些复杂或者低质量的图片，识别结果可能不尽如人意。  可以尝试调整 `OCR.yaml` 配置文件中的参数，或者尝试使用更高精度的识别模型来提高识别效果。

希望这份说明能够帮助你成功使用这个图片文字识别工具！ 如果在使用过程中遇到任何问题，欢迎再次提问。


好的，当然！  这是一份关于这个更新后的代码的通俗易懂的使用说明，它现在可以同时处理图片和 PDF 文件，并且依然对小白用户非常友好。

# 3.high_f2_pdf2.py

## 图片和 PDF 文字识别工具使用说明 

这个工具升级啦！ 现在它不仅可以识别 **图片中的文字**，还可以直接处理 **PDF 文件**，把 PDF 文件里的每一页都识别出来，并提取文字！  如果你既有图片，又有 PDF 文件需要进行文字识别，用这个工具就更方便了。  它依然可以快速处理大量文件，并充分利用你电脑的性能。

**1. 准备工作：安装工具 (与之前相同，但请再次确认)**

你需要安装的“小工具”和之前图片文字识别工具是一样的： **Python**, **PaddlePaddle**, **PaddleX**, **PyYAML**, **Pillow (PIL)**,  以及新增的 **pypdfium2** 和 **PyMuPDF**。

* **检查Python:**  同样先检查 Python 是否已安装。  打开 **命令提示符** (Windows) 或 **终端** (Mac/Linux)，输入 `python --version` 或 `python3 --version`。  已安装则跳过安装步骤，未安装则参考之前的说明安装。

* **安装必要的 Python 库:** 在 **命令提示符** 或 **终端** 中依次输入以下命令，**每条命令输入完都按回车键**：

    ```bash
    pip install paddlepaddle
    pip install paddlex
    pip install pyyaml
    pip install Pillow
    pip install pypdfium2
    pip install pymupdf
    ```

    或者 (如果 `pip` 不行，尝试 `pip3`)

    ```bash
    pip3 install paddlepaddle
    pip3 install paddlex
    pip3 install pyyaml
    pip3 install Pillow
    pip3 install pypdfium2
    pip3 install pymupdf
    ```

    这些命令会安装 PaddlePaddle, PaddleX, PyYAML, Pillow, **pypdfium2 (用于 PDF 处理)** 和 **PyMuPDF (另一个 PDF 处理库)**。  请确保所有库都安装成功，看到 "Successfully installed ..." 的提示。

**2. 下载代码和配置文件 (与之前完全相同)**

你需要把代码（从 `# import paddlex as pdx` 到 `if __name__ == "__main__":` 部分）复制保存为 `.py` 文件，例如 `ocr_tool.py`。  配置文件 `OCR.yaml` 也需要准备好，并放在和代码文件 **同一个文件夹**。  `OCR.yaml` 的内容可以使用默认配置，或者根据需要调整 (高级用户)。  **确保 `OCR.yaml` 和 `ocr_tool.py` 在同一目录下。**

**3. 准备 PDF 和图片文件夹，设置输出文件夹**

  * **PDF 和图片文件夹:**  现在你需要准备一个 **文件夹**，**这个文件夹里面可以同时放 PDF 文件 和 图片文件** (JPG, JPEG, PNG, BMP)。  程序会自动识别并处理 PDF 和图片。  例如，你可以新建一个文件夹 `我的文档`，然后把你要处理的所有 PDF 和图片都放进去。 **记住这个文件夹的路径**。

  * **输出文件夹:**  和之前一样，准备一个 **空的文件夹** 用于存放识别结果 (JSON 文件)。  例如 `OCR结果文件夹`。 **同样要记住这个文件夹的路径，并且确保一开始是空的。**

**4. 修改代码中的路径 (与之前的图片工具类似)**

打开你的 `.py` 代码文件 (比如 `ocr_tool.py`)。  找到以下几行代码：

```python
config_path = "/media/tmzn/DATA5/ocr_paddle/config_paddle/OCR.yaml"
image_root_dir = "/media/tmzn/DATA5/ocr_paddle/tmppdf"  #  PDF folder
output_root_dir = "/media/tmzn/DATA5/ocr_paddle/tmpicture_ocr_results2" # OCR results folder
log_and_error_dir = "/media/tmzn/DATA5/ocr_paddle/ocr_logs_and_errors2" # Log and error folder
```

和之前的说明一样，你需要 **修改这些路径为你电脑上实际的路径**。

* **`config_path = "/media/tmzn/DATA5/ocr_paddle/config_paddle/OCR.yaml"`:**  **改成 `config_path = "OCR.yaml"`** (假设 `OCR.yaml` 和代码文件在同一文件夹)。

* **`image_root_dir = "/media/tmzn/DATA5/ocr_paddle/tmppdf"`:**  **重要：**  现在 `image_root_dir`  变量的名字可能有点误导，虽然名字是 `image_root_dir`，但实际上它现在 **是你存放 PDF 文件和图片文件的那个总文件夹的路径**。  所以，**改成 `image_root_dir = "你的 PDF和图片总文件夹的路径"`**  ，  **把双引号里的内容替换成你准备好的 `我的文档` 文件夹 (或者你命名的总文件夹) 的完整路径**。

* **`output_root_dir = "/media/tmzn/DATA5/ocr_paddle/tmpicture_ocr_results2"`:**  改成  `output_root_dir = "你的输出文件夹的路径"`  ，  **替换成你准备好的 `OCR结果文件夹` (或你命名的输出文件夹) 的完整路径**。

* **`log_and_error_dir = "/media/tmzn/DATA5/ocr_paddle/ocr_logs_and_errors2"`:** 日志和错误文件夹路径，可以修改或保持默认。

**路径写法和注意事项** 仍然与之前的说明相同。  修改完成后，**保存代码文件**。

**5. 运行代码，开始识别 PDF 和图片文字**

打开 **命令提示符** (Windows) 或 **终端** (Mac/Linux)。

1.  **切换到代码文件目录:**  使用 `cd` 命令，切换到 `ocr_tool.py` 和 `OCR.yaml` 所在的文件夹。

2.  **运行代码:**  输入以下命令并回车：

    ```bash
    python ocr_tool.py
    ```

    或者 `python3 ocr_tool.py` (根据你的 Python 版本，以及你保存的代码文件名)。

    程序开始运行，你会看到命令行窗口显示处理进度。  **注意，现在程序会先处理 PDF 文件，把 PDF 每一页转成图片，然后再对这些图片以及你文件夹里原有的图片进行文字识别。**  进度信息会动态更新，显示已处理的页数/图片数，平均速度，预计完成时间等等。

**6. 查看识别结果 (与之前类似，但结果结构稍有变化)**

程序运行结束后，打开你的 **输出文件夹** (例如 `OCR结果文件夹`)。

*   **PDF 文件的结果:**  对于你输入的 PDF 文件，程序会在输出文件夹里 **为每个 PDF 文件创建一个单独的文件夹**，文件夹名字和 PDF 文件名相同。  在 PDF 文件夹里，你会看到：
    *   `temp_images` 文件夹： 里面是 PDF 每一页转换成的 PNG 图片 (默认情况下，程序运行结束后会 **自动删除** 这个 `temp_images` 文件夹，如果你想保留临时图片，需要修改代码中的 `delete_temp_images = True` 为 `False`)。
    *   `page_0001_result.json`, `page_0002_result.json`, ... 等文件： 这些是 PDF 每一页的文字识别结果，以 JSON 格式保存。
    *   `PDF文件名_result.json` 文件 (例如 `我的PDF文档_result.json`)：  这是一个 **汇总文件**，它包含了 **整个 PDF 文档所有页面的识别结果**，方便你一次性查看整个 PDF 的文字内容。

*   **图片文件的结果:**  对于你输入的图片文件，识别结果的 JSON 文件会 **直接保存在输出文件夹中**，文件名会以 `图片文件名_result.json` 的格式命名。  输出文件夹的目录结构会尽量保持和你的输入文件夹一致。

**7. 关于 "进程数" (num\_processes) 和 CPU/GPU 设置 (与之前完全相同)**

*   **进程数 (num\_processes):**  调整 `num_processes` 变量的值可以控制程序使用的 “小帮手” 数量，根据电脑配置调整，默认值通常够用。

*   **CPU/GPU 设置:**  `use_cpu = False` 默认使用 GPU 加速 (推荐)。  改成 `use_cpu = True`  强制使用 CPU。

**新增功能和注意事项:**

*   **PDF 文件处理:**  这个版本的工具可以 **直接处理 PDF 文件** 了！  你不需要手动把 PDF 转换成图片再识别，程序会自动完成 PDF 转换和识别的全过程。
*   **PDF 结果汇总:**  程序会为每个 PDF 文件生成一个 **汇总的 JSON 结果文件**，方便你查看整个 PDF 的识别内容。
*   **临时图片删除:**  PDF 转换成图片的临时文件，默认情况下会在识别完成后 **自动删除**，节省磁盘空间。  可以通过修改 `delete_temp_images` 变量来控制是否删除临时图片。
*   **其他注意事项:**  之前的 “常见问题和注意事项” (路径错误, 权限问题, 缺少依赖库, 文件名乱码, 内存占用, 识别精度等) 仍然适用。

希望这个升级后的工具能够更好地帮助你进行图片和 PDF 文字识别！  使用中有任何问题，欢迎随时提出。


# highocr4_f1_pdf_img.py



**主要更新与优化:**

**1. 增强的配置管理与灵活性:**
    * **用户可配置参数区域**: 脚本顶部新增了清晰的“用户可配置参数”区域，方便用户集中修改关键设置，如 PaddleX 缓存路径、输入源（支持多个PDF和图片文件夹）、输出目录、日志目录、临时文件删除选项、并发进程数等。
    * **`PADDLEX_HOME_OVERRIDE`**: 引入 `PADDLEX_HOME_OVERRIDE` 环境变量设置，允许用户指定 PaddleX 的主缓存目录，以解决默认路径（`~/.paddlex/temp`）可能导致的磁盘空间问题。脚本会尝试创建此目录，并提供了手动创建符号链接的建议。
    * **多输入源支持**: `INPUT_SOURCES` 配置允许用户指定多个不同路径和类型的输入（PDF 或图片文件夹），脚本会自动扫描和处理。
    * **细化的并发控制**:
        * `NUM_RENDER_PROCESSES_PER_PDF`: 控制单个PDF内部页面渲染的并发数（尽管当前版本中页面渲染在 `prepare_single_pdf_for_ocr` 中是串行的，但参数保留）。
        * `NUM_OCR_PROCESSES`: 主 OCR 任务的并发进程数。
        * `NUM_CONCURRENT_PDF_PREP_PROCESSES`: 新增参数，用于控制并行准备（页面提取和OCR任务列表生成）PDF 文件的进程数，优化了处理大量PDF文件时的效率。

**2. 显著的性能和效率提升:**
    * **多阶段处理流程**: `main` 函数重构为更清晰的多阶段流程：
        1.  扫描输入源，构建初始任务列表。
        2.  PDF 准备阶段：使用 `NUM_CONCURRENT_PDF_PREP_PROCESSES` 指定的进程数并行处理多个 PDF 文件（通过 `prepare_single_pdf_for_ocr` 函数）。每个 PDF 的页面提取现在在此函数内部串行执行，以避免之前嵌套 `Pool` 可能导致的问题。
        3.  OCR 执行阶段：将所有准备好的图片/页面任务提交给主 OCR `Pool`。
    * **优化的PDF处理逻辑 (`prepare_single_pdf_for_ocr`)**:
        * 代替了旧的 `pdf_to_images_multiprocess` 和部分 `pdf_path_generator` 的功能。
        * 该函数现在负责单个PDF的页面提取（串行化，移除了内部的 `Pool` 以增强稳定性）、检查已存在的JSON结果以跳过OCR，并返回需要OCR的图片路径列表。
        * 直接将待OCR的任务参数返回给主流程，由主OCR `Pool` 统一处理。
    * **更智能的缓存清理 (`clear_cache`)**:
        * 全面重写了缓存清理逻辑，优先管理 `PADDLEX_HOME_OVERRIDE` 指定的目录。
        * 尝试处理 `~/.paddlex/temp` 与 `PADDLEX_HOME_OVERRIDE`/temp 之间的符号链接关系，确保 PaddleX 使用正确的缓存路径。
        * 增加了更详细的日志输出和错误处理，并在可能挂起的地方（如 `shutil.rmtree` 大量小文件时）给出了用户提示。
    * **工作进程初始化优化 (`init_worker`)**:
        * 确保在每个工作进程中正确设置 `PADDLEX_HOME` 环境变量，并尝试创建相应的缓存目录结构，增强了在分布式环境下的稳定性。
    * **提前跳过已处理文件**: 在扫描图片输入源和 `prepare_single_pdf_for_ocr` 内部，都会检查目标JSON文件是否存在，如果存在则跳过后续的OCR处理，节省了不必要的计算。

**3. 增强的健壮性与错误处理:**
    * **`render_page` 优化**:
        * 现在会检查目标图片是否已存在，若存在则跳过渲染。
        * 更细致地处理 `fitz.open()` 和页面加载可能出现的错误，尝试在错误发生后关闭 PDF 文档。
    * **`process_image` 增强**:
        * 在工作进程内记录更详细的日志，包括工作进程PID。
        * 增加了对 `image_info` 参数格式和类型的严格校验。
        * 图片校验 (`Image.open().verify()`) 被恢复，以尽早发现损坏的图片。
        * 详细记录了图片处理各阶段（校验、OCR预测、JSON保存）的耗时。
        * 改进了对 PaddleX `predict()` 方法返回结果的处理，能更好地处理生成器、列表或单个结果对象，并记录了保存JSON文件的数量和相关错误。
    * **全局 `log_file_path`**: 许多函数现在接受 `log_file_path` 参数，确保日志记录到统一的文件。
    * **`RedirectStdout`**: 仍然用于抑制 PaddleX 预测过程中的标准输出，保持控制台清洁。

**4. 改进的用户体验与日志记录:**
    * **彩色输出与日志文件**: `colored_output` 函数现在可以选择是否同时输出到标准输出和日志文件 (`stdout_too` 参数)。
    * **更详细的进度显示**:
        * PDF 准备阶段会显示已处理的PDF文件数量。
        * `prepare_single_pdf_for_ocr` 内部记录了页面渲染的进度。
        * 主OCR任务的进度条现在显示已处理的任务数、百分比、平均速度、预计完成时间（ETA）、成功数和错误数，并且会动态更新。
    * **时间格式化**: `format_timedelta` 用于将总处理时间等格式化为 HH:MM:SS。
    * **日志信息**: 整个脚本的日志信息更加丰富，包括了启动信息、配置参数、各阶段耗时、错误详情等。

**5. 代码结构与可维护性:**
    * **模块化**: 将PDF的页面提取和OCR任务准备逻辑封装到 `prepare_single_pdf_for_ocr` 函数中，使得 `main` 函数的流程更清晰。
    * **参数传递**: 配置文件路径、批处理大小等通过参数传递给工作进程初始化函数和核心处理函数，而不是完全依赖全局变量。
    * **移除了部分旧函数/逻辑**: 旧的 `pdf_to_images_multiprocess` 函数虽然保留，但其核心功能已被新的PDF处理流程取代。 `pdf_path_generator` 的逻辑被整合到了新的扫描和准备阶段。

**潜在的更新说明标题:** "大规模重构：性能、配置与稳定性全面升级的OCR处理流程"

**更新说明摘要示例:**
此版本对OCR处理脚本进行了重大重构和优化。主要改进包括：引入了灵活的用户配置参数（支持多输入源和`PADDLEX_HOME`覆盖），显著提升了处理大量PDF文件时的并行准备效率，优化了缓存管理机制，增强了错误处理和日志记录的详细程度。新的多阶段处理流程使得任务管理更加清晰，同时改进了进度报告和用户体验。这些更改旨在提高脚本的整体性能、可配置性、健壮性和易用性。



# 高性能 PaddleX OCR 脚本 (highocr4_f1_pdf_img.py)

本项目是一个基于 PaddleX 实现的高性能光学字符识别（OCR）Python 脚本。它能够批量处理指定的PDF文件和图片文件夹，提取其中的文本信息，并将结果保存为 JSON 文件。脚本设计了多级并行处理机制以提高效率，并提供了丰富的配置选项。

## 主要功能

* **支持多种输入源**：可同时处理 PDF 文件和包含图片的文件夹。
* **PDF 页面提取**：自动将 PDF 文件按页面转换为图片（PNG格式）进行 OCR。
* **多级并行处理**：
    * 并行准备多个 PDF 文件（页面提取和待 OCR 列表生成）。
    * 并行进行 OCR 任务（使用独立的进程池）。
    * （注意：单个PDF内的页面渲染当前版本改为串行，以解决潜在的嵌套 Pool 问题）。
* **结果缓存/跳过**：如果某个图片/页面的 OCR 结果 JSON 文件已存在，则跳过该文件的处理，避免重复工作。
* **PaddleX 缓存管理**：
    * 允许用户指定 `PADDLEX_HOME` 环境变量的覆盖路径，以控制 PaddleX 模型和临时文件的存储位置。
    * 脚本启动时会尝试清理指定的 PaddleX 临时缓存目录。
    * 强烈建议用户在特定情况下手动创建符号链接以确保缓存路径正确。
* **灵活配置**：大部分关键参数（如并发数、输入输出路径、模型配置等）均可在脚本顶部进行配置。
* **CPU/GPU 支持**：可配置 OCR 任务使用 CPU 或 GPU。
* **详细日志记录**：记录脚本运行过程中的关键信息、错误和统计数据。
* **错误处理**：处理失败的图片会被复制到指定的错误图片目录。
* **临时文件管理**：可配置是否在处理完 PDF 后删除从 PDF 转换产生的临时图片。
* **进度显示**：在控制台实时显示 OCR 处理进度、平均速度和预计剩余时间。
* **彩色控制台输出**：使用不同颜色标记不同类型的日志信息，便于阅读。

## 环境依赖

* Python 3.x
* PaddlePaddle (`paddlepaddle` 或 `paddlepaddle-gpu`，根据你的硬件选择)
* PaddleX (`paddlex`)
* PyMuPDF (`fitz`)：用于 PDF 文件解析和页面渲染。
* Pillow (`PIL`)：用于图像基本操作和校验。
* PyYAML (`yaml`)：用于加载和修改 PaddleX 配置文件。
* NumPy (`numpy`)
* pypdfium2 (虽然 `fitz` 是主要渲染工具，但此库也被导入)

**安装建议**：
建议在虚拟环境中安装依赖。
```bash
pip install paddlepaddle  # 或者 paddlepaddle-gpu
pip install paddlex
pip install PyMuPDF Pillow PyYAML numpy pypdfium2
````

## 配置文件和目录结构

### 1\. PaddleX 模型配置文件

  * 脚本通过 `CONFIG_PATH_PADDLE` 参数指定 PaddleX OCR 模型的 YAML 配置文件路径。
  * 示例配置 `OCR.yaml` 需要用户自行准备，并确保其与所使用的 PaddleX 版本和模型兼容。

### 2\. 输入与输出

  * **输入源 (`INPUT_SOURCES`)**：
      * 一个列表，每个元素是一个字典，定义一个输入源。
      * 字典包含 `path` (绝对路径) 和 `type` ('pdf' 或 'image')。
      * 脚本会递归扫描指定路径下的所有符合类型的文件。
  * **OCR 结果输出根目录 (`OUTPUT_ROOT_DIR`)**：
      * 所有 OCR 结果（JSON 文件）和从 PDF 转换的临时图片都会存放在此目录下。
      * 输出目录会保持与输入源相对路径一致的结构。
      * 例如，如果输入图片为 `/media/tmzn/DATA4/splitdict/汉语/类别A/img1.jpg`，输出根目录为 `/media/tmzn/DATA5/ocr_paddle/词典pdf_ocr_result`，则结果JSON会保存在 `/media/tmzn/DATA5/ocr_paddle/词典pdf_ocr_result/类别A/img1_result.json`。
      * 对于 PDF 文件，例如 `/media/tmzn/DATA5/ocr_paddle/词典pdf/book1.pdf`，其每页的 JSON 结果会保存在 `/media/tmzn/DATA5/ocr_paddle/词典pdf_ocr_result/词典pdf/book1/page_XXXX_result.json`，临时图片（如果未删除）会保存在 `/media/tmzn/DATA5/ocr_paddle/词典pdf_ocr_result/词典pdf/book1/temp_images_from_pdf/page_XXXX.png`。
  * **日志和错误图片目录 (`LOG_AND_ERROR_DIR_BASE`)**：
      * `ocr_log.txt`：详细的运行日志。
      * `error_images/`：存放处理失败的原始图片。

## 参数配置 (用户可配置)

在脚本的 `highocr4_f1_pdf_img.py` 文件顶部，可以找到以下用户可配置参数：

  * `PADDLEX_HOME_OVERRIDE`: (字符串) PaddleX 缓存主目录。用于重定向 PaddleX 的缓存和临时文件，防止占满默认用户主目录磁盘。
      * **重要提示**：如果 PaddleX 仍然将临时文件写入默认的 `~/.paddlex/temp/` 并导致磁盘空间不足，请务必按照脚本注释中的建议，手动将 `~/.paddlex/temp` 符号链接到此 `PADDLEX_HOME_OVERRIDE` 下的 `temp` 目录。例如：`ln -sfn /media/tmzn/DATA5/paddlex_cache_home/temp ~/.paddlex/temp`。
  * `CONFIG_PATH_PADDLE`: (字符串) PaddleX OCR 模型配置文件 (例如 `OCR.yaml`) 的绝对路径。
  * `INPUT_SOURCES`: (列表) 输入源配置。每个元素是一个字典，包含：
      * `path`: (字符串) 输入文件夹的绝对路径。
      * `type`: (字符串) `'pdf'` 或 `'image'`。
  * `OUTPUT_ROOT_DIR`: (字符串) OCR 结果统一输出的根目录。
  * `LOG_AND_ERROR_DIR_BASE`: (字符串) 日志文件和错误图片存放的根目录。
  * `DELETE_TEMP_IMAGES_AFTER_PDF_PROCESSING`: (布尔值) 是否删除 PDF 转换产生的临时图片 (`True` 删除, `False` 保留)。
  * `NUM_RENDER_PROCESSES_PER_PDF`: (整数) 单个PDF内页面渲染时的并发进程数。*注意：当前版本中，由于 `prepare_single_pdf_for_ocr` 内部页面渲染改为串行，此参数主要影响串行渲染的逻辑，但保留以便将来可能恢复嵌套并行。*
  * `NUM_OCR_PROCESSES`: (整数) 主 OCR 任务的并发进程数（即同时处理多少张图片/页面）。
  * `OCR_BATCH_SIZE`: (整数) PaddleX OCR 模型的批处理大小。增大此值可能提高 GPU 利用率，但需注意显存。
  * `NUM_CONCURRENT_PDF_PREP_PROCESSES`: (整数) 并行准备 PDF 文件的进程数。决定同时有多少个 PDF 文件可以被并行地进行页面提取和待 OCR 列表的生成。
  * `USE_CPU_FOR_OCR`: (布尔值) 是否强制使用 CPU 进行 OCR (`True` 表示使用 CPU，`False` 表示根据配置文件，通常是 GPU)。

## 使用方法

1.  **安装依赖**：确保所有必要的 Python 包已安装（参见 [环境依赖](https://www.google.com/search?q=%23%E7%8E%AF%E5%A2%83%E4%BE%9D%E8%B5%96)）。
2.  **准备 PaddleX 模型**：
      * 获取或训练一个 PaddleX OCR 模型。
      * 准备对应的 `OCR.yaml` 配置文件。
3.  **配置脚本参数**：
      * 打开 `highocr4_f1_pdf_img.py` 文件。
      * 根据你的环境和需求，修改脚本顶部的用户可配置参数（路径、并发数等）。
      * 特别注意 `PADDLEX_HOME_OVERRIDE`、`CONFIG_PATH_PADDLE`、`INPUT_SOURCES` 和 `OUTPUT_ROOT_DIR` 的设置。
4.  **运行脚本**：
    ```bash
    python highocr4_f1_pdf_img.py
    ```
5.  **查看结果**：
      * OCR 结果 (JSON 文件) 会保存在 `OUTPUT_ROOT_DIR` 下对应的子目录中。
      * 运行日志会保存在 `LOG_AND_ERROR_DIR_BASE/ocr_log.txt`。
      * 处理失败的图片会保存在 `LOG_AND_ERROR_DIR_BASE/error_images/`。

## 工作流程概述

1.  **初始化**：
      * 设置 `PADDLEX_HOME` 环境变量（如果 `PADDLEX_HOME_OVERRIDE` 已配置）。
      * 创建必要的输出和日志目录。
      * 清理 PaddleX 缓存目录 (`clear_cache()`)。
      * 记录开始时间。
2.  **扫描输入源**：
      * 遍历 `INPUT_SOURCES` 中定义的每个路径。
      * 对于 PDF 文件，构建 PDF 准备任务列表。
      * 对于图片文件，检查其对应的 JSON 结果是否已存在。如果不存在，则直接构建 OCR 任务列表。
3.  **PDF 准备阶段** (使用 `NUM_CONCURRENT_PDF_PREP_PROCESSES` 个进程并行处理不同的 PDF 文件)：
      * 对于每个 PDF 文件，调用 `prepare_single_pdf_for_ocr` 函数：
          * 创建该 PDF 对应的临时图片输出目录和 OCR 结果输出目录。
          * 获取 PDF 总页数。
          * **串行地**（在 `prepare_single_pdf_for_ocr` 内部）为每一页调用 `render_page`：
              * 将 PDF 页面渲染为 PNG 图片，保存到临时图片目录。
              * 如果页面图片已存在，则跳过渲染。
          * 检查渲染后的图片对应的 OCR 结果 JSON 文件是否已存在。
          * 如果 JSON 不存在，则将该图片的路径添加到待 OCR 任务列表。
          * 返回待 OCR 的图片路径列表、此 PDF 中跳过的 OCR 任务数、临时图片文件夹路径以及 OCR 结果输出目录。
4.  **收集 OCR 任务**：
      * 汇总所有直接来自图片输入源和来自 PDF 准备阶段的待 OCR 图片任务。
5.  **OCR 执行阶段** (使用 `NUM_OCR_PROCESSES` 个工作进程并行处理图片)：
      * 初始化 OCR 工作进程池 (`ocr_pool`)，每个工作进程加载 PaddleX 模型 (`init_worker`)。
      * 将所有待 OCR 的图片任务提交到 `ocr_pool`。
      * 每个工作进程执行 `process_image` 函数：
          * 校验图片是否有效。
          * 调用 PaddleX `pipeline.predict()` 进行 OCR。此步骤的 PaddleX 内部输出会被重定向以保持控制台清洁。
          * 将 OCR 识别结果保存为 JSON 文件。
          * 如果处理过程中发生错误，将原始图片复制到 `error_images` 目录。
      * 主进程监控 OCR 任务的完成情况，并实时更新控制台进度条。
6.  **清理**：
      * 如果 `DELETE_TEMP_IMAGES_AFTER_PDF_PROCESSING` 为 `True`，则删除所有在 PDF 处理过程中产生的临时图片文件夹。
7.  **总结与报告**：
      * 输出总处理时间、成功/失败任务数、跳过任务数等统计信息。

## 注意事项

  * **PaddleX 缓存 (`PADDLEX_HOME_OVERRIDE` 和符号链接)**：
      * PaddleX 在运行时会下载模型、生成临时文件等，默认情况下这些文件存储在 `~/.paddlex/`。如果主目录磁盘空间有限，这可能导致问题。
      * `PADDLEX_HOME_OVERRIDE` 参数允许你指定一个新的基础目录。脚本会尝试通过设置 `PADDLEX_HOME` 环境变量来让 PaddleX 使用这个新路径。
      * **关键**：某些 PaddleX 版本或特定操作可能仍会尝试写入 `~/.paddlex/temp`。如果遇到此问题，最可靠的解决方案是**手动创建符号链接**，将 `~/.paddlex/temp` 指向 `PADDLEX_HOME_OVERRIDE/temp`。例如：
        ```bash
        # 假设 PADDLEX_HOME_OVERRIDE = "/mnt/large_disk/paddlex_cache"
        mkdir -p /mnt/large_disk/paddlex_cache/temp
        rm -rf ~/.paddlex/temp  # 如果已存在，先删除或备份
        ln -sfn /mnt/large_disk/paddlex_cache/temp ~/.paddlex/temp
        ```
      * 脚本中的 `clear_cache()` 函数会尝试清理 `PADDLEX_HOME_OVERRIDE/temp` 和 `~/.paddlex/temp`。如果 `~/.paddlex/temp` 是一个指向 `PADDLEX_HOME_OVERRIDE/temp` 的符号链接，它会优先管理目标真实目录，并尝试保持符号链接的有效性。
  * **PyMuPDF (`fitz`) 与多进程**：`fitz.open()` 在多进程环境中，尤其是在子进程中再次创建进程池时，可能存在不稳定性或导致进程挂起。当前版本已将单个 PDF 内的页面渲染改为串行，以避免此类问题。主要的并行性体现在同时处理多个不同的 PDF 文件以及同时进行多个 OCR 任务。
  * **配置文件兼容性**：确保 `CONFIG_PATH_PADDLE` 指向的 `OCR.yaml` 与你安装的 PaddleX 和 PaddlePaddle 版本兼容。
  * **显存/内存**：较大的 `OCR_BATCH_SIZE` 和较多的 `NUM_OCR_PROCESSES`（尤其是在 GPU 模式下）会消耗更多显存和内存。请根据硬件资源进行调整。
  * **文件路径**：脚本中所有路径配置（如 `CONFIG_PATH_PADDLE`, `INPUT_SOURCES` 中的 `path`, `OUTPUT_ROOT_DIR` 等）都应使用绝对路径，以避免潜在的相对路径问题。
  * **错误排查**：
      * 首先检查 `LOG_AND_ERROR_DIR_BASE/ocr_log.txt` 中的日志信息。
      * 查看 `LOG_AND_ERROR_DIR_BASE/error_images/` 目录中是否有处理失败的图片。
      * 检查 PaddleX 缓存目录和 `~/.paddlex/temp` 是否有异常。

## 未来可能的改进

  * 更细致的进度条，分别显示 PDF 准备进度和 OCR 进度。
  * 更完善的错误重试机制。
  * 通过命令行参数传递配置，而非直接修改脚本。
  * 支持更多图片和文档格式。




# del_10min_cache.py
# 自动清理过期文件脚本 (Auto Clean Expired Files)

这是一个 Python 脚本，用于自动监控指定目录，并使用多线程删除超过设定时间未被修改的文件。此脚本设计为持续运行，定期扫描并清理文件。

## 功能特性

* **自动删除**: 自动删除指定目录下超过预设时间（例如10分钟）未修改的文件。
* **递归扫描**: 扫描指定目录及其所有子目录下的文件。
* **多线程删除**: 使用多个线程并行删除文件，以提高效率，尤其是在处理大量小文件时。
* **持续监控**: 脚本会持续运行，并按设定的时间间隔重复扫描和清理操作。
* **可配置参数**:
    * 目标监控目录
    * 文件过期时间（分钟）
    * 扫描间隔时间（秒）
    * 删除操作的最大线程数
* **命令行输出**: 在前台运行时，会打印详细的扫描和删除日志。
* **优雅退出**: 支持通过 `Ctrl+C` 来中断程序运行。

## 环境要求

* Python 3.x
* 操作系统：Linux (已在此环境测试，理论上也兼容 macOS 和 Windows，但路径格式可能需要调整)

## 配置

在运行脚本之前，你需要在脚本 (`auto_delete_cache.py` 或你命名的文件) 中修改以下常量：

```python
# 要监控和清理的目录
TARGET_DIR = "/media/tmzn/DATA5/paddlex_cache_home"  # 修改为你的目标目录
# 文件过期时间（分钟）
EXPIRATION_MINUTES = 10
# 扫描间隔时间（秒）
SCAN_INTERVAL_SECONDS = 60
# 删除操作的最大线程数
MAX_THREADS = 5  # 根据你的CPU核心数和IO性能调整
