# paddle_change
把paddle改的更好用，最大限度利用硬件资源。
# NOTICE
**！！！**
**最好在ubuntu22.04+CUDA=11.8 Python 3.9.21的情况下**

**这个东西可能对你的CPU GPU 内存 磁盘造成严重负载（大概就是总有一个到达瓶颈）。**

**请小心使用，记得看温度，使用btop和nvitip做好监测，防止烧毁硬件。**

**同时对磁盘的4K读写及寿命要求极高，很容易出现谈笑之间写了一个T的情况。如果是一般消费级磁盘请谨慎使用。（谁用消费级磁盘处理大量数据啊？）**

# 主要功能
## 0. pdf拆图
请使用pdf_to_png.py完成

## 1. 大量图片进行ocr处理
请使用highocr3_f2.py完成。
## 2. 制作可以搜索的pdf

请使用pdf_creator_with_text_layer5.py完成。
## 3. 直接pdf进行ocr
请使用highocr3_f2_pdf.py完成

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

