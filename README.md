# RecursiveScholarCrawler - 递归式学术文献网络爬虫

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.8+-brightgreen.svg)](https://www.python.org/)

📚 **项目简介**  
RecursiveScholarCrawler 是一个为科研人员设计的强大工具，它能从一篇或多篇种子论文开始，**自动化地构建和下载一个完整的学术引用网络**。本项目通过先进的大型语言模型（LLM）精准提取参考文献，利用多源API查询可靠的DOI信息，并通过健壮的下载模块获取PDF全文，最终形成一个结构化的、可供分析的本地学术文献库。

---

🌟 **核心功能**

-   🔄 **全自动引用网络构建**: 从根论文出发，递归解析其参考文献，并持续向下探索，构建深度的引用关系网。
-   🤖 **LLM 赋能的精准解析**: 借助大型语言模型（如 DeepSeek, Llama, Qwen 等）的能力，高精度地从PDF中提取标题、作者、摘要及结构化的参考文献列表。
-   🔗 **强大的DOI查找与验证**: 综合利用 CrossRef 等学术API，为缺失DOI的文献进行智能查找与相似度匹配，确保下载目标的准确性。
-   📥 **多镜像健壮下载**: 通过遍历多个 Sci-Hub 镜像源并自动重试，大幅提高PDF全文的下载成功率，并内置PDF有效性验证，避免下载错误页面。
-   📊 **可视化进度与统计**: 实时记录已处理论文数、已下载文献数、失败率等关键指标，任务进度一目了然。
-   💾 **完善的断点续传**: 所有任务状态（包括处理进度、下载队列、失败记录）都会被持久化。即使程序意外中断，也能从上次的断点完美恢复，无需从头开始。
-   🧠 **内存与性能优化**: 针对长时间、大规模爬取任务进行了内存管理优化，通过批处理和智能队列管理，确保在处理数万篇论文时依然能稳定运行。
-   🔧 **独立的恢复工具**: 提供一个独立的状态恢复脚本，即使在严重崩溃导致队列文件损坏时，也能从下载日志中一键恢复所有失败的任务。

---

🛠️ **安装与配置**

1.  **克隆本仓库**:
    ```bash
    git clone https://github.com/your-username/RecursiveScholarCrawler.git
    cd RecursiveScholarCrawler
    ```

2.  **创建虚拟环境 (推荐)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

---

🚀 **快速开始**

### 1. 准备工作
-   将你的种子PDF论文放入一个目录，例如 `papers/`。
-   获取你的LLM服务的API `base_url`, `api_key` 和 `model_name`。

### 2. 启动主程序

执行以下命令开始你的第一次爬取任务：

```bash
python parser.py \
    --input-dir ./papers \
    --output-dir ./output \
    --api-base https://your-api-base-url/v1 \
    --api-key YOUR_API_KEY \
    --model your-llm-model-name
```

### 3. (可选) 当任务中断后恢复

如果程序意外停止，`download_queue.json` 文件可能损坏。此时，先运行恢复脚本：

```bash
# 从下载日志中重建下载队列
python recover_queue.py --output-dir ./output
```

恢复成功后，再次运行主程序命令即可从断点处继续。

---

📌 **命令行参数详解**

| 参数                    | 说明                                                                                              | 默认值 |
| ----------------------- | ------------------------------------------------------------------------------------------------- | ------ |
| `--input-dir`           | **[必需]** 包含初始PDF文件的目录。                                                                | `None` |
| `--output-dir`          | **[必需]** 存储所有输出文件（日志、状态、下载的PDF）的目录。                                        | `None` |
| `--api-base`            | **[必需]** LLM API的基础URL。                                                                     | `None` |
| `--model`               | **[必需]** 要使用的LLM模型名称。                                                                  | `None` |
| `--api-key`             | LLM API密钥。                                                                                     | `EMPTY`|
| `--subdir`              | 只处理`input-dir`下的特定子目录。输出也会在`output-dir`下创建同名子目录。                        | `None` |
| `--max-depth`           | 最大递归深度。`-1`表示无限制，`0`表示只处理初始论文。                                               | `-1`   |
| `--max-workers`         | PDF解析和LLM请求的最大并行工作线程数。                                                            | `4`    |
| `--text-ratio`          | 优化PDF文本提取，格式为`"前部%,后部%"` (例如`10,30`)。有助于处理大文件并聚焦参考文献。         | `None` |
| `--download-delay`      | 每次下载尝试之间的基础延迟时间（秒），以避免IP被封。                                              | `3`    |
| `--batch-size`          | 下载批处理大小，每批处理完后会进入更长的等待时间。                                                | `20`   |
| `--batch-delay`         | 每批下载任务之间的长延迟时间（秒）。                                                              | `300`  |
| `--save-interval`       | 自动保存所有状态文件的间隔时间（秒）。                                                            | `30`   |
| `--retry-downloads-only`| 一个独立的模式：只运行一次对所有历史失败下载的重试，然后退出。                                    | `False`|
| `--no-retry-on-start`   | 禁用主程序启动时默认的自动重试失败下载功能。                                                      | `False`|
| `--temperature`, `--top-p` | LLM生成参数，用于微调解析质量。                                                                   | `None` |

---

📂 **输出文件结构**

在指定的 `--output-dir` 中，你会看到以下文件和目录：
```
output/
├── downloads/                  # 所有成功下载的PDF文件都存放在这里
├── paper_details.json          # [摘要] 已处理论文的元数据摘要（用于快速检查）
├── crawler_stats.json          # [统计] 任务的总体统计数据
├── progress.json               # [状态] 记录哪些文件已被处理、失败或待处理
├── download_queue.json         # [队列] 当前待下载论文的动态队列
├── download_results.json       # [日志] 所有下载尝试的详细结果记录（成功与失败）
├── processing_state.json       # [状态] 实时记录当前正在被处理的文件，用于崩溃恢复
└── ... (更多日志和以论文名命名的详细JSON文件)
```

---

🔧 **故障排除**

**问题: 程序启动后很快就显示“All tasks appear to be completed!”，但实际上还有很多论文没下载。**  
**解决方案:** 这通常意味着 `download_queue.json` 文件已损坏或被清空。请停止主程序，运行 `python recover_queue.py --output-dir ./your-output-dir` 来恢复队列，然后再重新启动主程序。

**问题: 下载时出现大量 `403 Forbidden` 错误。**  
**解决方案:** 你的请求过于频繁，已被Sci-Hub暂时封禁。请停止程序，并使用更“温柔”的参数重新启动，例如：`--download-delay 10 --batch-size 5 --batch-delay 600`。

**问题: LLM提取的参考文献不准确或不完整。**  
**解决方案:**
1.  首先尝试使用 `--text-ratio 10,40`。这会强制LLM关注论文的开头（标题、摘要）和结尾（参考文献），通常效果最好。
2.  如果问题依旧，可以微调LLM参数，例如尝试一个略高的 `--temperature` (如`0.5`)。
3.  确保你使用的LLM模型本身具备较强的JSON抽取和遵循指令的能力。

**问题: 内存使用量过高或程序崩溃。**  
**解决方案:**
1.  减少并行工作线程数：`--max-workers 2`。
2.  如果初始PDF文件特别大，请务必使用 `--text-ratio` 参数来减少送给LLM的文本量。

---

📝 **免责声明**
-   本工具仅供个人学术研究和教育目的使用。
-   用户在使用本工具时，必须严格遵守目标网站（如Sci-Hub、CrossRef）的使用条款和所有相关的版权法律法规。
-   开发者对用户因使用本工具而可能产生的任何法律责任或后果概不负责。请负责任地使用本工具。

---

📄 **许可证**

本项目采用 [Apache License 2.0](LICENSE) 授权。