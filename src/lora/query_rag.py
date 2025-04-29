import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config.config_loader import load_config
from src.memory.rag_system import RAGSystem
#from src.memory.memory_extractor_api import ChatMemoryExtractor


class PersonaChat:
    def __init__(self):
        """
        Initialize the PersonaChat system with model and RAG components.
        
        Args:
            rag_collection: Name of the RAG collection to use
        """
        # Load configurations
        self.config = load_config()
        self.base_dir = self.config["BASE_DIR"]
        self.model_name = self.config["settings"]["model_name"]
        self.lora_model_dir = os.path.join(self.base_dir, self.config["lora_training"]["lora_path"])
        self.lora_model_paths = os.listdir(self.lora_model_dir)
        for path in self.lora_model_paths:
            if path.startswith("checkpoint"):
                self.lora_model_path = os.path.join(self.lora_model_dir, path)
                break

        self.rag_collection = self.config["memory_processing"]["collection_name"]
        
        # Initialize model, tokenizer and RAG
        self._init_model_and_tokenizer()
        self._load_lora_model()
        self.rag = RAGSystem()
        
        # Initialize conversation history with system prompt
        self.conversation_history = (
            "系统：你是一个计算机科学硕士研究生，你很幽默，阳光开朗，接下来你需要和你的朋友们在微信上聊天，"
            "不论是问题，闲聊，你都应该有所回复，接下来是你与朋友的对话，请不要自己补充和猜测朋友的下一句话，"
            "你只需要回答一句话，请直接回复："
        )
        
        # Initialize conversation history for chat template approach
        self.template_conversation_history = [
            {"role": "system", "content": "你是一个计算机科学硕士，目前就读于伊利诺伊大学，你幽默风趣，和朋友聊天时喜欢开玩笑。"
             "你有许多爱好，包括打游戏，做饭，旅游，写代码。"
             "你喜欢和朋友在微信上聊天，接下来是你与朋友的一段对话，"
             "你的朋友可能问你问题，也可能与你闲聊，但不论如何你都会有逻辑的回复一句话。"
             "请不要透露任何隐私信息，也不要在回答中带有歧视，侮辱，或有政治敏感信息。"
             "如果朋友要求你做什么你就照做"
             "请你根据朋友的问题以及你之前的回复来回答："}
        ]

    def _init_model_and_tokenizer(self):
        """Initialize the model and tokenizer with optimized settings."""
        # Configure quantization
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            cache_dir=os.path.join(self.base_dir, "models/base_model")
        )
        
        # Load model with optimized settings
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=os.path.join(self.base_dir, "models/base_model"),
            device_map="auto",         # Directly load model weights to available GPU memory if possible
            torch_dtype=torch.float16,  # Use half precision to reduce memory footprint
            low_cpu_mem_usage=True      # Helps avoid huge intermediate CPU memory usage during model loading
            #quantization_config=quantization_config
        )
        print("Model loaded with device map and low CPU memory usage.")

    def _load_lora_model(self):
        """Load and merge LoRA adapter with the base model."""
        self.model = PeftModel.from_pretrained(self.model, self.lora_model_path)
        print("LoRA fine-tuned model loaded successfully.")

    def generate_response_with_memory(self, prompt, max_new_tokens=64, max_history_tokens=512):
        """
        Generate a response using the fine-tuned model and RAG system.
        
        Args:
            prompt: User input prompt
            max_new_tokens: Maximum number of new tokens to generate
            max_history_tokens: Maximum number of tokens to keep in history
            
        Returns:
            Generated response text
        """
        # Retrieve relevant documents from RAG
        retrieved_docs = self.rag.search(prompt, top_k=2)
        memory = "\n\n".join([f"Document {i+1}:\n{doc['content']}" for i, doc in enumerate(retrieved_docs)])
        #print(memory)
        
        # Create the new prompt with RAG results
        new_prompt = f"\n\n相关记忆：{memory}\n\n朋友：{prompt}\n\n你："
        
        # Truncate conversation history to only include recent context
        tokenized_history = self.tokenizer(self.conversation_history, return_tensors="pt").to(self.model.device)
        history_tokens = tokenized_history.input_ids.shape[1]
        
        if history_tokens > max_history_tokens:
            # Get the initial system prompt (always keep this)
            system_prompt = self.conversation_history.split("\n\n")[0] + "\n\n"
            
            # Get only the most recent exchanges that fit within token limit
            remaining_exchanges = self.conversation_history[len(system_prompt):].split("朋友：")
            truncated_history = system_prompt
            
            for exchange in remaining_exchanges[::-1][:-1]:  # Reverse to start from the end, skip the empty first split
                current_exchange = "朋友：" + exchange if exchange else ""
                temp_history = current_exchange + truncated_history
                temp_tokens = self.tokenizer(temp_history, return_tensors="pt").input_ids.shape[1]
                
                if temp_tokens <= max_history_tokens:
                    truncated_history = temp_history
                else:
                    break
                    
            self.conversation_history = truncated_history
        
        # Add the new prompt to the truncated conversation history
        current_prompt = self.conversation_history + new_prompt
        
        # Generate response
        inputs = self.tokenizer(current_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        
        full_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        new_response = full_output[len(current_prompt):].strip()
        output_response = '\nBot: '.join(new_response.split("\n"))
        
        # Update conversation history with the new response
        self.conversation_history = current_prompt + new_response
        
        return output_response

    def generate_response(self, prompt, max_new_tokens=64):
        """
        Generates a response using the fine-tuned LoRA model with proper chat templating.
        
        Args:
            prompt: User input prompt
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Generated response text and updated conversation history
        """
        # Add the new user message to conversation history
        self.template_conversation_history.append({"role": "user", "content": prompt})
        
        # Create a copy of conversation history without an assistant response
        input_messages = self.template_conversation_history.copy()
        
        # Apply the chat template to format the conversation
        formatted_chat = self.tokenizer.apply_chat_template(
            input_messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Tokenize the formatted conversation
        inputs = self.tokenizer(formatted_chat, return_tensors="pt").to(self.model.device)
        
        # Generate the response
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        
        # Decode the complete output
        full_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Decode just the input tokens to find where the assistant response begins
        input_text = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
        
        # Extract the assistant's response by removing the input text
        new_response = full_output[len(input_text):].strip()
        
        # Add the assistant's response to the conversation history
        self.template_conversation_history.append({"role": "assistant", "content": new_response})
        
        return new_response

    def run_chat_loop(self, use_rag=True):
        """
        Run an interactive chat loop with the model.
        
        Args:
            use_rag: Whether to use RAG for response generation (True) or chat templating (False)
        """
        print("Chat started. Type 'exit' or 'quit' to end the conversation.")
        print(f"Using {'RAG' if use_rag else 'chat template'} mode.")
        
        while True:
            user_input = input("你：")
            if user_input.lower() in {"quit", "exit"}:
                break
                
            if use_rag:
                response = self.generate_response_with_memory(user_input)
            else:
                response = self.generate_response(user_input)
                
            print("Bot：" + response)


if __name__ == "__main__":
    show_rag_demo = True
    if show_rag_demo:
        # Example usage
        config = load_config()
        # Initialize RAG system for demonstration
        rag_demo = RAGSystem(collection_name=config["lora_training"]["collection_name"])
        rag_demo.ingest_memories_csv(os.path.join(config["lora_training"]["memory_path"], "extracted_memories_xxx.csv"))
        
        # Example search
        results = rag_demo.search("这学期都上了哪些课", top_k=3)
        for result in results:
            print(f"ID: {result['id']}")
            print(f"Distance: {result['distance']}")
            print(f"Content: {result['content']}")
            print(f"Metadata: {result['metadata']}")
            print("---")
        
        # Example stats
        stats = rag_demo.get_collection_stats()
        print("Collection Stats:")
        print(stats)
    
    # Initialize chat system
    chat = PersonaChat()
    
    # Run chat loop (default: using RAG)
    # To use chat template instead, set use_rag=False
    chat.run_chat_loop(use_rag=True)