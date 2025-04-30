# Persona-Echo

[English](README.md) | [中文](README_zh.md)

Persona-Echo 是一个会话式 AI 系统，旨在通过对大型语言模型（LLMs）进行聊天历史数据的微调来模仿特定人物角色。它结合了记忆提取、检索增强生成（RAG）和 LoRA 微调技术，创建个性化的聊天体验，模拟特定个体的风格、语调和知识。

## 概述

该系统处理聊天历史数据，从对话中提取有意义的记忆，并通过 RAG 和微调来增强大型语言模型的能力。这使模型能够：

1. 以反映该人物的沟通风格方式回应
2. 精确引用共享记忆和过去的对话
3. 保持对关系历史的上下文感知

## 特性

- **聊天历史处理**：将原始聊天历史数据转换为结构化训练样例
- **记忆提取**：使用 LLMs 从对话中总结和提取关键记忆
- **基于记忆的 RAG 系统**：在对话过程中检索相关记忆
- **LoRA 微调**：高效地将大型模型（如 Qwen 2.5）适应到特定角色模式
- **对话界面**：以自然方式与角色模型交互

## 系统架构

```
Persona-Echo
├── data/              # 数据存储
│   ├── sample_data/   # 原始聊天历史
│   ├── cleaned_data/  # 处理后的聊天数据
│   ├── memory_data/   # 提取的记忆
│   └── extra_sticker/ # 额外表情数据
├── configs/           # 配置文件
├── models/            # 模型存储
│   ├── base_model/    # 基础 LLM
│   └── finetuned_lora/# LoRA 适配器
├── src/               # 源代码
│   ├── data/          # 数据处理模块
│   ├── memory/        # 记忆提取和 RAG 系统
│   ├── lora/          # LoRA 微调流程
│   └── config/        # 配置处理
└── scripts/           # 执行脚本
```

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/hrwu1/Persona-Echo.git
cd Persona-Echo
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 下载所需模型（首次运行时自动处理）

## 使用方法

该项目组织为具有多个阶段的流程，可以通过提供的 Jupyter 笔记本运行：

1. **数据处理**：准备训练用的聊天数据
```python
from src.data.data_process import DataProcessor
data_processor = DataProcessor()
data_processor.run()
```

2. **记忆提取**：从对话中提取关键记忆
```python
from src.memory.memory_extractor_api import ChatMemoryExtractor
extractor = ChatMemoryExtractor()
extractor.extract_memories()
```

3. **RAG 系统设置**：将记忆导入检索系统
```python
from src.memory.rag_system import RAGSystem
rag_demo = RAGSystem()
rag_demo.ingest_memories_csv()
```

4. **LoRA 微调**：将基础模型适应到特定角色
```python
from src.lora.finetune_lora import LoRAFinetuner
finetuner = LoRAFinetuner()
finetuner.run_finetuning_pipeline()
```

5. **交互式聊天**：与角色适应模型对话
```python
from src.lora.query_rag import PersonaChat
chat = PersonaChat()
chat.run_chat_loop()
```

为方便起见，所有这些步骤都包含在 `scripts/main.ipynb` 笔记本中。

## 配置

系统参数可以在 `configs/config.yaml` 中自定义：

- **模型选择**：选择基础 LLM
- **训练参数**：调整 LoRA 设置
- **数据处理**：配置聊天历史处理
- **记忆设置**：调整记忆提取和检索

## 要求

- Python 3.11+
- PyTorch 2.3+
- CUDA 12.1+（用于 GPU 加速）
- 16GB+ GPU 内存（用于运行 Qwen 模型）
- 完整依赖列表见 `requirements.txt`

## 许可

本项目根据所包含的 LICENSE 文件条款授权。

## 致谢

- 本项目使用 Qwen 2.5 模型作为基础 LLM
- RAG 系统使用 ChromaDB 和 SentenceTransformers 构建
- LoRA 训练使用 PEFT 库实现

## 贡献

欢迎贡献、错误报告和功能请求！随时打开 issue 或提交 pull request。 