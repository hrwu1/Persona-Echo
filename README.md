# Persona-Echo

[English](README.md) | [中文](README_zh.md)

Persona-Echo is a conversational AI system designed to imitate a persona through fine-tuning large language models (LLMs) on chat history data. It combines memory extraction, retrieval-augmented generation (RAG), and LoRA fine-tuning to create a personalized chat experience that mimics the style, tone, and knowledge of a specific individual.

## Overview

The system processes chat history data, extracts meaningful memories from conversations, and uses these memories to enhance the capabilities of a large language model through both RAG and fine-tuning. This enables the model to:

1. Respond in a manner that reflects the person's communication style
2. Reference shared memories and past conversations with precision
3. Maintain contextual awareness of the relationship history

## Features

- **Chat History Processing**: Converts raw chat history data into structured training examples
- **Memory Extraction**: Uses LLMs to summarize and extract key memories from conversations
- **Memory-Based RAG System**: Retrieves relevant memories during conversations
- **LoRA Fine-tuning**: Efficiently adapts large models (like Qwen 2.5) to persona-specific patterns
- **Conversational Interface**: Interact with the persona model in a natural way

## System Architecture

```
Persona-Echo
├── data/              # Data storage
│   ├── sample_data/   # Raw chat history
│   ├── cleaned_data/  # Processed chat data
│   ├── memory_data/   # Extracted memories
│   └── extra_sticker/ # Additional sticker data
├── configs/           # Configuration files
├── models/            # Model storage
│   ├── base_model/    # Base LLM
│   └── finetuned_lora/# LoRA adapters
├── src/               # Source code
│   ├── data/          # Data processing modules
│   ├── memory/        # Memory extraction and RAG system
│   ├── lora/          # LoRA fine-tuning pipeline
│   └── config/        # Configuration handling
└── scripts/           # Execution scripts
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hrwu1/Persona-Echo.git
cd Persona-Echo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models (automatically handled during first run)

## Usage

The project is organized as a pipeline with several stages that can be run through the provided Jupyter notebook:

1. **Data Processing**: Prepare chat data for training
```python
from src.data.data_process import DataProcessor
data_processor = DataProcessor()
data_processor.run()
```

2. **Memory Extraction**: Extract key memories from conversations
```python
from src.memory.memory_extractor_api import ChatMemoryExtractor
extractor = ChatMemoryExtractor()
extractor.extract_memories()
```

3. **RAG System Setup**: Ingest memories into the retrieval system
```python
from src.memory.rag_system import RAGSystem
rag_demo = RAGSystem()
rag_demo.ingest_memories_csv()
```

4. **LoRA Fine-tuning**: Adapt the base model to the persona
```python
from src.lora.finetune_lora import LoRAFinetuner
finetuner = LoRAFinetuner()
finetuner.run_finetuning_pipeline()
```

5. **Interactive Chat**: Talk with the persona-adapted model
```python
from src.lora.query_rag import PersonaChat
chat = PersonaChat()
chat.run_chat_loop()
```

For convenience, all these steps are included in the `scripts/main.ipynb` notebook.

## Configuration

System parameters can be customized in `configs/config.yaml`:

- **Model selection**: Choose the base LLM
- **Training parameters**: Adjust LoRA settings
- **Data processing**: Configure chat history handling
- **Memory settings**: Adjust memory extraction and retrieval

## Requirements

- Python 3.11+
- PyTorch 2.3+
- CUDA 12.1+ (for GPU acceleration)
- 16GB+ GPU memory (for running Qwen models)
- See `requirements.txt` for a complete list of dependencies

## License

This project is licensed under the terms of the included LICENSE file.

## Acknowledgments

- This project uses the Qwen 2.5 model as the base LLM
- The RAG system is built using ChromaDB and SentenceTransformers
- LoRA training is implemented using the PEFT library

## Contributing

Contributions, bug reports, and feature requests are welcome! Feel free to open an issue or submit a pull request. 