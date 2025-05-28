# RecursiveScholarCrawler - 递归学术文献爬虫

![版本](https://img.shields.io/badge/版本-1.0.0-blue)
![许可证](https://img.shields.io/badge/许可证-Apache--2.0-green)

## 📚 项目简介

RecursiveScholarCrawler 是一个强大的学术文献爬虫工具，能够自动化提取学术论文的元数据（标题、作者、摘要）和参考文献，并递归下载这些参考文献形成完整的引用网络。本项目利用大型语言模型（LLM）进行准确的引用信息提取，通过DOI查询实现高效的论文检索，并提供完整的状态跟踪和断点续传功能。

### 🌟 核心功能

- 🔄 **递归爬取**: 从根论文开始，自动递归爬取其所有参考文献及其进一步的引用
- 🤖 **LLM智能提取**: 利用大模型精准提取论文元数据和参考文献信息
- 📋 **自动DOI解析**: 智能查找和匹配论文DOI标识符
- 📥 **自动下载**: 通过Sci-Hub等渠道获取全文PDF
- 📊 **完整统计**: 详细记录处理进度和成功率
- 🔁 **断点续传**: 支持中断后从断点处继续任务

## 🛠️ 安装方法

1. 克隆仓库:

git clone https://github.com/yourusername/RecursiveScholarCrawler.git
cd RecursiveScholarCrawler

2. 安装依赖:

pip install -r requirements.txt

## 🚀 使用方法

### 基本用法

python main.py --input-dir ./papers --output-dir ./output --api-base https://your-api-base --api-key YOUR_API_KEY --model model-name

### 高级选项

# 递归深度限制为2级
python main.py --input-dir ./papers --output-dir ./output --api-base https://your-api-base --api-key YOUR_API_KEY --model model-name --max-depth 2

# 使用10个并行工作线程
python main.py --input-dir ./papers --output-dir ./output --api-base https://your-api-base --api-key YOUR_API_KEY --model model-name --max-workers 10

# 只处理论文的前10%和后30%内容（更高效地提取摘要和参考文献）
python main.py --input-dir ./papers --output-dir ./output --api-base https://your-api-base --api-key YOUR_API_KEY --model model-name --text-ratio 10,30

### 参数说明

| 参数 | 说明 |
|------|------|
| `--input-dir` | 包含输入PDF文件的目录 |
| `--output-dir` | 存储输出文件的目录 |
| `--api-base` | LLM API的基础URL |
| `--api-key` | LLM API密钥 |
| `--model` | 要使用的LLM模型名称 |
| `--max-depth` | 最大递归深度，-1表示无限制 |
| `--max-workers` | 最大并行工作线程数 |
| `--text-ratio` | 文本提取比例，格式为"前部%,后部%" |
| `--temperature` | LLM温度参数 |
| `--top-p` | LLM top-p参数 |
| `--top-k` | LLM top-k参数 |

## 📂 输出文件

- `paper_details.json`: 每篇论文的详细信息，包括元数据和引用
- `crawler_stats.json`: 爬取进度和统计信息
- `downloads/`: 下载的PDF文件存储目录
- `paper_crawler.log`: 详细的日志记录

## 📝 注意事项

- 本工具仅用于学术研究目的
- 请遵守相关网站的使用条款和版权法规
- 大规模爬取时请合理控制频率，避免对服务器造成负担

## 🔧 故障排除

**问题**: 无法连接到API服务器  
**解决方案**: 检查API基础URL和密钥是否正确，确保网络连接正常

**问题**: PDF解析失败  
**解决方案**: 确保PDF文件未加密且格式正确，尝试使用`--text-ratio`参数处理复杂格式

**问题**: 参考文献提取不完整  
**解决方案**: 调整LLM参数如temperature，或使用`--text-ratio`参数确保参考文献部分被包含在处理文本中

## 📄 许可证

Apache License 2.0
