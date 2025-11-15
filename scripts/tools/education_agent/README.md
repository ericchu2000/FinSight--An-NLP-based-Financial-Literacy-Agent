# 金融教育内容生成器

本项目是一个基于大型语言模型（LLM）的金融教育内容生成工具。它接收一个包含股票预测信息的JSON文件，然后自动生成一段通俗易懂的金融知识普及内容，旨在帮助金融初学者理解AI的决策逻辑。同时，它能识别关键的金融概念并进行解释，验证金融知识的准确性，提供个性化的教育内容。

## ✨ 功能特性

- **智能金融概念解释**：自动识别预测依据中的关键金融指标（如MACD, RSI, KDJ等）并进行解释。
- **个性化教育**：根据用户与AI的预测差异，生成有针对性的教育内容。
- **多格式输入兼容**：能够处理多种不同结构和字段命名的JSON输入文件。
- **强大的错误处理**：对输入文件格式、内容和API调用进行校验和错误处理。
- **安全保障**：内置多层防御机制，防止Prompt注入，并过滤不当的投资建议。
- **事实验证**：通过金融知识库验证预测中引用的金融知识的准确性。

## 🚀 快速开始

### 1. 环境准备

确保您的环境中已安装 Python 3.6 或更高版本。

### 2. 安装依赖

本项目依赖于 `requests` 库来调用API。如果您的环境中没有安装，请运行：

```bash
pip install requests
```

### 3. 配置API密钥

为了能成功调用百炼（Bailian）的API，您需要配置API密钥。推荐使用环境变量的方式进行配置，这样最安全。

**Windows (PowerShell):**
```powershell
$env:BAILIAN_API_KEY="你的-api-key-sk-xxxx"
```

**Windows (CMD):**
```cmd
set BAILIAN_API_KEY=你的-api-key-sk-xxxx
```

**Linux / macOS:**
```bash
export BAILIAN_API_KEY="你的-api-key-sk-xxxx"
```

> **备用方案**：如果您只是想快速测试，也可以直接在 `financial_educator.py` 文件中修改 `API_KEY` 的默认值。但不推荐在生产环境中使用此方法。

## 💻 4. 核心模块介绍   

### financial_educator.py

主程序入口文件，负责协调其他模块并完成金融教育内容的生成。主要功能包括：

- **加载配置**：读取prompt模板和金融概念数据
- **处理输入**：解析和验证预测数据JSON文件
- **内容生成**：调用百炼API生成教育内容
- **错误处理**：提供多级错误处理和重试机制
- **投资建议过滤**：过滤可能的直接投资建议
- **命令行接口**：提供灵活的命令行参数选项

### modules.py

包含百炼API客户端的封装，负责与大型语言模型通信。主要功能包括：

- **BailianClient类**：封装了API调用逻辑和错误处理
- **请求构建**：构建符合百炼API要求的请求格式
- **响应解析**：解析API返回的响应并提取有用信息
- **辅助函数**：提供prompt构建和代码提取等功能

### input_processor.py

输入数据处理器，负责净化、标准化和增强输入数据。主要功能包括：

- **输入净化**：防止Prompt注入攻击
- **格式标准化**：统一各种不同格式的输入结构
- **金融概念识别**：识别需要解释的关键金融指标
- **数据增强**：添加概念定义和类比信息

### error_handler.py

错误处理器，负责检测和报告各种错误情况。主要功能包括：

- **JSON文件验证**：检查文件格式和解析错误
- **数据结构验证**：确保必要字段存在
- **错误报告**：提供详细的错误信息和修复建议
- **用户友好提示**：转换技术错误为易于理解的提示

### factcheck.py

金融知识检索和事实验证模块，负责验证金融知识的准确性。主要功能包括：

- **知识库加载**：从JSONL文件加载金融知识
- **索引构建**：为标题和内容建立检索索引
- **知识检索**：基于关键词搜索相关金融知识
- **事实验证**：验证预测中引用的金融知识


### 5. 运行程序

配置完成后，您就可以运行主程序了。

**基本运行：**

使用默认的测试文件 (`test_prediction.json`) 生成内容并输出到控制台。

```bash
python financial_educator.py
```

**指定输入文件：**

使用 `-f` 或 `--file` 参数来指定一个你自己的JSON输入文件。

```bash
python financial_educator.py -f err_test_01_both_up.json
```

**保存输出到文件：**

使用 `-o` 或 `--output` 参数将生成的Markdown内容保存到文件中。

```bash
python financial_educator.py -o result.md
```

**显示处理后的数据结构：**

使用 `-s` 或 `--show-processed` 参数可以在生成内容前，先在控制台打印出程序内部处理和标准化的数据结构，便于调试。

```bash
python financial_educator.py -s
```

**启用或禁用事实验证功能：**

使用 `--factcheck` 参数启用事实验证功能，或者 `--no-factcheck` 参数禁用此功能。

```bash
python financial_educator.py --factcheck
```

## 📁 项目文件结构

```
.
├── financial_educator.py       # 主程序入口
├── modules.py                  # 封装了API客户端
├── input_processor.py          # 输入数据处理器（标准化、净化）
├── error_handler.py            # 错误处理器（文件、JSON校验）
├── factcheck.py               # 金融知识检索和事实验证模块
├── prompt_template.txt         # 核心Prompt模板
├── financial_concepts.json     # 金融概念知识库
├── Financial_knowledge.jsonl   # 金融知识库（用于事实验证）
├── test_prediction.json        # 默认测试文件
├── err_test_*.json             # 用于多样本测试的系列文件
└── README.md                   # 本文档
```

## 🛠️ 如何贡献

如果您有任何改进建议或发现了Bug，欢迎提交Issue或Pull Request。

1. **扩展金融概念**：可以直接修改 `financial_concepts.json`，添加更多金融指标的定义、类比和解释模板。
2. **优化Prompt**：可以调整 `prompt_template.txt` 中的指令，以获得更好的生成效果。
3. **增强代码逻辑**：改进 `input_processor.py` 的兼容性或 `error_handler.py` 的健壮性。
4. **扩充知识库**：向 `Financial_knowledge.jsonl` 添加更多金融知识条目，提升事实验证能力。
