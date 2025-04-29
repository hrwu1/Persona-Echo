import os
import sys
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
#from awq import AutoAWQForCausalLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config.config_loader import load_config

class LoRAFinetuner:
    def __init__(self):
        # Load Configurations
        self.config = load_config()
        self.base_dir = self.config["BASE_DIR"]
        self.model_name = self.config["settings"]["model_name"]
        self.output_path = os.path.join(self.base_dir, self.config["paths"]["processed_data"])
        self.jsonl_file = os.path.join(self.output_path, "conversations.jsonl")
        self.lora_r = self.config["lora_training"]["lora_r"]
        self.lora_alpha = self.config["lora_training"]["lora_alpha"]
        self.lora_dropout = self.config["lora_training"]["lora_dropout"]
        self.lora_path = os.path.join(self.base_dir, self.config["lora_training"]["lora_path"])
        self.lora_epochs = self.config["lora_training"]["lora_epochs"]
        self.lora_learning_rate = self.config["lora_training"]["lora_learning_rate"]
        self.lora_batch_size = self.config["lora_training"]["lora_batch_size"]
        self.lora_gradient_accumulation_steps = self.config["lora_training"]["lora_gradient_accumulation_steps"]
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.model_with_lora = None
        self.dataset = None
        
    def load_model_and_tokenizer(self):
        """Load the model and tokenizer with memory-efficient settings"""
        #quantization_config = BitsAndBytesConfig(
        #    load_in_4bit=True,
        #)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            cache_dir=os.path.join(self.base_dir, "models/base_model")
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=os.path.join(self.base_dir, "models/base_model"),
            device_map="auto",         # Directly load model weights to available GPU memory if possible
            torch_dtype=torch.float16,  # Use half precision to reduce memory footprint
            low_cpu_mem_usage=True      # Helps avoid huge intermediate CPU memory usage during model loading
            #quantization_config=quantization_config
        )
        
        print("Model loaded with device map and low CPU memory usage.")
        return self.model, self.tokenizer
    
    def apply_lora(self):
        """Apply LoRA configuration to the model"""
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            target_modules=["q_proj", "v_proj"],  # "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"
            task_type="CAUSAL_LM",
        )
        
        self.model_with_lora = get_peft_model(self.model, lora_config)
        self.model_with_lora.print_trainable_parameters()
        return self.model_with_lora
    
    def load_chat_dataset(self, input_file=None):
        """
        Reads a JSONL chat dataset and formats it.
        A system prompt is prepended to every conversation.
        Each conversation is then split into samples that end with a single assistant reply.
        """
        if input_file is None:
            input_file = self.jsonl_file
            
        formatted_samples = []
        system_prompt = {
            "role": "system", 
            "content": "You are a helpful agent."
        }

        with open(input_file, "r", encoding="utf-8") as infile:
            for line in infile:
                try:
                    chat_entry = json.loads(line.strip())
                    if "conversation" not in chat_entry:
                        continue
                    # Prepend the system prompt to the conversation
                    conversation = [system_prompt] + chat_entry["conversation"]
                    # Split the conversation into single-turn samples.
                    samples = self._split_conversation(conversation)
                    formatted_samples.extend(samples)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line}")
        return formatted_samples
    
    def _split_conversation(self, conversation):
        """
        Given a conversation (a list of messages with roles), split it into multiple samples.
        Each sample ends immediately after an assistant message.
        """
        samples = []
        current_context = [conversation[0]]  # always start with system prompt
        for msg in conversation[1:]:
            current_context.append(msg)
            if msg["role"] == "assistant":
                samples.append({"conversation": list(current_context)})
        return samples
    
    def tokenize_single_response(self, example, max_length=256):
        """
        Tokenizes a conversation sample using the model's chat template such that:
          - All context (system prompt, user messages, earlier assistant turns) is included in the input
            but their tokens are masked (labels set to -100).
          - Only the final assistant reply is used as the target (labels match token IDs).
        """
        # Extract the conversation from the example
        messages = example["conversation"]
        
        if not messages or messages[-1]["role"] != "assistant":
            return None  # Should never happen if splitting is done correctly
        
        # Split the messages into context and target
        context_messages = messages[:-1]
        target_message = messages[-1]
        
        # Apply chat template to the entire conversation
        full_text = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Apply chat template to just the context messages
        context_text = ""
        if context_messages:
            context_text = self.tokenizer.apply_chat_template(
                context_messages,
                add_generation_prompt=True,
                tokenize=False
            )
        
        # Tokenize the full conversation
        full_tokens = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        
        # Tokenize just the context
        context_tokens = []
        if context_text:
            context_tokens = self.tokenizer(context_text, add_special_tokens=False)["input_ids"]
        
        # Determine which tokens belong to the target by finding the difference
        # (This assumes the target tokens are at the end, after the context)
        target_tokens = full_tokens[len(context_tokens):]
        
        # Create the input_ids and labels
        all_input_ids = full_tokens
        
        # Create labels: -100 for context tokens, actual token IDs for target tokens
        all_labels = [-100] * len(context_tokens) + target_tokens
        
        # Pad and/or truncate the sequences to max_length
        model_inputs = self.tokenizer.prepare_for_model(
            all_input_ids,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Ensure labels are padded/truncated identically
        if len(all_labels) < max_length:
            all_labels = all_labels + ([-100] * (max_length - len(all_labels)))
        else:
            all_labels = all_labels[:max_length]
        
        model_inputs["labels"] = all_labels
        return model_inputs
    
    def prepare_dataset(self, max_length=128):
        """Prepare and tokenize the dataset for training"""
        # Load chat samples
        chat_samples = self.load_chat_dataset()
        
        # Convert to a Hugging Face Dataset
        hf_dataset = Dataset.from_list(chat_samples)
        
        # Tokenize and filter out any samples that failed
        tokenized_dataset = hf_dataset.map(
            lambda x: self.tokenize_single_response(x, max_length=max_length),
            batched=False
        ).filter(lambda x: x is not None)
        
        self.dataset = tokenized_dataset
        print(f"Prepared dataset with {len(tokenized_dataset)} samples")
        return tokenized_dataset
    
    def train(self):
        """Train the model with the prepared dataset"""
        # Define training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=self.lora_batch_size,
            gradient_accumulation_steps=self.lora_gradient_accumulation_steps,
            learning_rate=self.lora_learning_rate,
            num_train_epochs=self.lora_epochs,
            logging_steps=100,
            save_strategy="epoch",
            output_dir=self.lora_path,
            save_total_limit=1,
            fp16=True,
            optim="adamw_torch_fused",
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model_with_lora,
            args=training_args,
            train_dataset=self.dataset
        )
        
        # Start training
        trainer.train()
        return trainer

    def run_finetuning_pipeline(self, max_length=128):
        """Run the full finetuning pipeline"""
        self.load_model_and_tokenizer()
        self.apply_lora()
        self.prepare_dataset(max_length=max_length)
        return self.train()


# Example usage
if __name__ == "__main__":
    finetuner = LoRAFinetuner()
    finetuner.run_finetuning_pipeline()
