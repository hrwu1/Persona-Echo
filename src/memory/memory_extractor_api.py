import pandas as pd
import datetime
import json
from typing import List, Dict, Any, Optional, Tuple
import os
import sys
import traceback
import time
import openai
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config.config_loader import load_config


class ChatMemoryExtractor:
    def __init__(
        self,
        openai_api_key: str = None,  # OpenAI API key
        model: str = "deepseek-reasoner",  # Default model to use
        batch_size: int = 1,  # Default batch size of 1
        time_window: int = 3600 * 24,  # Group messages within 24 hours as a conversation
        min_messages: int = 2,  # Minimum messages to consider for memory extraction
        memory_output_file: str = "extracted_memories.csv",
        save_batch_size: int = 1,  # Save after 1 batch
        checkpoint_file: str = "extraction_checkpoint.json",  # File to track progress
        max_tokens: int = 4096  # Maximum tokens for response generation
    ):
        self.batch_size = batch_size
        self.time_window = time_window
        self.min_messages = min_messages
        self.memory_output_file = memory_output_file
        self.save_batch_size = save_batch_size
        self.checkpoint_file = checkpoint_file
        self.max_tokens = max_tokens
        self.model = model
        
        self.config = load_config()
        openai_api_key = self.config["memory_processing"]["api_key"]
        # Set up OpenAI client
        if openai_api_key:
            self.client = openai.OpenAI(api_key=openai_api_key, base_url="https://api.deepseek.com")
        else:
            # Use environment variable OPENAI_API_KEY if not explicitly provided
            #self.client = openai.OpenAI()
            self.client = openai.OpenAI(base_url="https://api.deepseek.com")
        
        print(f"Using OpenAI model: {model}")
        
    def load_chat_data(self, csv_path: str) -> pd.DataFrame:
        """Load chat data from CSV and perform initial preprocessing."""
        df = pd.read_csv(csv_path)
        
        # Ensure we have the expected columns
        required_cols = ['sender', 'msg', 'send_time']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV")
        
        # Convert timestamps to datetime
        df['datetime'] = pd.to_datetime(df['send_time'], unit='s')
        
        # Sort by time
        df = df.sort_values('send_time')
        
        # Convert sender to string labels for clarity
        df['sender'] = df['sender'].astype(str)
        
        return df
    
    def segment_conversations(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Segment the chat history into separate conversations based on time gaps.
        Returns a list of dataframes, each representing a conversation.
        """
        conversations = []
        current_convo = []
        last_time = None
        
        for _, row in df.iterrows():
            if last_time is None or (row['send_time'] - last_time) > self.time_window:
                # If this is the first message or there's a significant time gap,
                # start a new conversation
                if current_convo and len(current_convo) >= self.min_messages:
                    conversations.append(pd.DataFrame(current_convo))
                current_convo = [row]
            else:
                # Continue the current conversation
                current_convo.append(row)
            
            last_time = row['send_time']
        
        # Add the last conversation if it exists and meets the minimum message count
        if current_convo and len(current_convo) >= self.min_messages:
            conversations.append(pd.DataFrame(current_convo))
        
        return conversations
    
    def format_conversation_for_llm(self, conversation: pd.DataFrame) -> str:
        """Format a conversation segment for input to the LLM."""
        formatted_msgs = []
        
        for _, row in conversation.iterrows():
            sender = f"用户{row['sender']}"
            time_str = row['datetime'].strftime("%Y-%m-%d %H:%M:%S")
            formatted_msgs.append(f"{sender} [{time_str}]: {row['msg']}")
        
        # Include the date range of the conversation
        start_date = conversation['datetime'].min().strftime("%Y-%m-%d")
        end_date = conversation['datetime'].max().strftime("%Y-%m-%d")
        date_range = f"{start_date}"
        if start_date != end_date:
            date_range += f" 到 {end_date}"
        
        formatted_text = f"以下是{date_range}的对话:\n\n" + "\n".join(formatted_msgs)
        return formatted_text
    
    def generate_prompt(self, conversation_text: str) -> str:
        """Generate a prompt for the LLM to extract memories."""
        return f"原始对话：{conversation_text}\n提炼结果："

    def extract_memories_from_batch(self, conversation_batch: List[pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Process a batch of conversations to extract memories using OpenAI API.
        Returns a list of dictionaries containing the extracted memories and metadata.
        Handles API errors with retries.
        """
        results = []
        prompts = []
        
        # Prepare all prompts in the batch
        for convo in conversation_batch:
            convo_text = self.format_conversation_for_llm(convo)
            prompt = self.generate_prompt(convo_text)
            prompts.append((prompt, convo))
        
        # Process the prompts with OpenAI API
        for i, (prompt, convo) in enumerate(prompts):
            memory_text = ""
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    # Call OpenAI API
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": 
                             """你是一个高效的信息提炼助手，请从用户提供的微信聊天记录中提取有价值的长期记忆信息。包括事实、事件、观点、约定、决策、时间节点、人名等关键信息，尽量避免无关闲聊或重复信息。请遵循以下原则：

核心关注点：

专属回忆（对话双方提到的共同经历，如约饭，出去玩）
日常生活（单方面分享的日常经历）
想法/观点（针对一些事的观点、看法）
情感表达（提到的对事情或人的情感、态度）
重要事件（对其中一人或双方有很大影响的事件，如升学，找工作）
待办事项/任务（提到需要做的事，含截止日期）

处理要求：

用中文以自然语言形式输出总结
保持原有时序逻辑，重要时间节点需保留
关键数字/日期必须原文呈现
合并重复信息，保留最新版本
区分工作/学业和生活/社交
对模糊表述进行合理性推测时需标注[推测]

过滤标准：

忽略日常寒暄、表情包、无实质内容对话
不保留第三方隐私信息（如他人身份证号、银行卡）

示例处理1：
原始对话：用户1 [2024-10-22 20:12:47]: 周三组会改到周五下午3点，需要准备5页PPT，小王说他有模板可以分享
提炼结果：记忆：[待办事项/任务] 周五15:00组会（原定周三），需准备5页PPT，小王可提供模板（待联系确认）

示例处理2：
原始对话：用户0 [2024-11-27 19:32:06]: 我们410有作业吗
用户1 [2024-11-27 19:32:34]: 没有吧应该，12.4有个hw3 due，很早之前发的
应该没新的了
正在boston玩哈哈哈哈哈
看到消息大惊
用户0 [2024-11-27 19:33:07]: 草
我周围1/3去芝加哥
1/3去纽约
1/3去波士顿
笑死我了
我找找
有说tech report的事情吗
用户1 [2024-11-27 19:33:28]: 那也快了
canvas上410一堆due
反正都是十二月初
提炼结果：记忆：[待办事项/任务] 410 HW3截止日期12月4日
记忆：[日常生活] 周围同学近期分赴芝加哥、纽约、波士顿（提及比例1/3）
记忆：[日常生活] 对方当前正在波士顿游玩
记忆：[待办事项/任务] 410课程十二月初有多项任务待完成（需查看Canvas确认具体内容）

现在请处理以下聊天记录，请将每个记忆点作为单独的一行，以"记忆："开头，不论如何不要输出任何其他内容："""
                            },
                            {"role": "user", "content": prompt}
                        ],
                        stream=False,
                        max_tokens=self.max_tokens,
                        temperature=0.5  # Lower temperature for more focused outputs
                    )
                    
                    # Get the generated text
                    memory_text = response.choices[0].message.content.strip()
                    break  # Break out of retry loop if successful
                    
                except Exception as e:
                    retry_count += 1
                    print(f"Error calling OpenAI API (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        # Exponential backoff
                        sleep_time = 2 ** retry_count + 1
                        print(f"Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                    else:
                        print(f"Failed after {max_retries} attempts, skipping this conversation")
                        memory_text = "处理错误：API调用失败"
            
            # Parse the memories from the LLM output
            memories = []
            for line in memory_text.split('\n'):
                line = line.strip()
                if line.startswith("MEMORY:"):
                    memories.append(line[len("MEMORY:"):].strip())
                elif line.startswith("记忆:"):
                    memories.append(line[len("记忆:"):].strip())
                elif line.startswith("记忆："):
                    memories.append(line[len("记忆："):].strip())
            
            # Create result with metadata
            start_time = convo['send_time'].min()
            end_time = convo['send_time'].max()
            
            result = {
                "start_time": start_time,
                "end_time": end_time,
                "start_datetime": datetime.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
                "end_datetime": datetime.datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S"),
                "num_messages": len(convo),
                "participants": convo['sender'].unique().tolist(),
                "memories": memories,
                "raw_llm_output": memory_text
            }
            
            results.append(result)
        
        return results
    
    def save_memories_batch_to_csv(self, results: List[Dict[str, Any]], is_first_batch: bool = False) -> None:
        """Save a batch of extracted memories to CSV file."""
        memory_rows = []
        
        for result in results:
            start_time = result['start_time']
            end_time = result['end_time']
            
            for memory in result['memories']:
                memory_rows.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'start_datetime': result['start_datetime'],
                    'end_datetime': result['end_datetime'],
                    'memory': memory,
                    'participants': ', '.join(map(str, result['participants']))
                })
        
        # Create DataFrame from memory rows
        if memory_rows:
            memory_df = pd.DataFrame(memory_rows)
            
            # Determine mode: write or append
            mode = 'w' if is_first_batch else 'a'
            header = True if is_first_batch else False
            
            # Save to CSV
            memory_df.to_csv(self.memory_output_file, mode=mode, header=header, index=False)
            print(f"Saved {len(memory_rows)} memories to {self.memory_output_file} (mode: {mode})")
        else:
            print("No memories to save in this batch")
    
    def load_checkpoint(self) -> Dict[str, Any]:
        """Load the checkpoint file if it exists."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                print(f"Loaded checkpoint: processed {checkpoint.get('conversations_processed', 0)} conversations")
                return checkpoint
            except Exception as e:
                print(f"Error loading checkpoint file: {e}")
                return {"conversations_processed": 0, "first_batch": True, "total_memories": 0}
        else:
            return {"conversations_processed": 0, "first_batch": True, "total_memories": 0}
    
    def save_checkpoint(self, conversations_processed: int, first_batch: bool, total_memories: int) -> None:
        """Save the current progress to the checkpoint file."""
        checkpoint = {
            "conversations_processed": conversations_processed,
            "first_batch": first_batch,
            "total_memories": total_memories,
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            print(f"Saved checkpoint: processed {conversations_processed} conversations")
        except Exception as e:
            print(f"Error saving checkpoint file: {e}")
    
    def check_existing_output(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if the output file exists and has content.
        Returns a tuple of (exists, metadata) where metadata contains
        information about the already processed conversations.
        """
        if os.path.exists(self.memory_output_file):
            try:
                df = pd.read_csv(self.memory_output_file)
                if len(df) > 0:
                    # Extract the latest end_time from the existing data
                    latest_end_time = df['end_time'].max() if 'end_time' in df.columns else 0
                    # Count entries
                    entry_count = len(df)
                    # Get unique conversation time ranges
                    unique_convos = df[['start_time', 'end_time']].drop_duplicates() if 'start_time' in df.columns and 'end_time' in df.columns else pd.DataFrame()
                    convo_count = len(unique_convos)
                    
                    return True, {
                        "latest_end_time": latest_end_time,
                        "entry_count": entry_count,
                        "conversation_count": convo_count,
                        "unique_conversations": unique_convos.to_dict('records') if not unique_convos.empty else []
                    }
            except Exception as e:
                print(f"Error reading existing output file: {e}")
                return False, {}
        return False, {}
    
    def process_all_conversations(self, conversations: List[pd.DataFrame]) -> Dict[str, Any]:
        """Process all conversations in batches, save results incrementally, and return summary stats."""
        # Load checkpoint if exists
        checkpoint = self.load_checkpoint()
        conversations_processed = checkpoint.get("conversations_processed", 0)
        first_batch = checkpoint.get("first_batch", True)
        total_memories = checkpoint.get("total_memories", 0)
        
        # Skip already processed conversations
        if conversations_processed > 0:
            print(f"Resuming from checkpoint: skipping first {conversations_processed} conversations")
            if conversations_processed >= len(conversations):
                print("All conversations have been processed according to checkpoint")
                return {
                    "total_conversations": len(conversations),
                    "total_memories_extracted": total_memories,
                    "total_results_saved": conversations_processed,
                    "memory_output_file": self.memory_output_file,
                    "status": "completed_from_checkpoint"
                }
        
        total_results_saved = conversations_processed
        pending_results = []
        
        # Process in batches, starting from where we left off
        for i in tqdm(range(conversations_processed, len(conversations), self.batch_size), 
                     desc="Processing conversation batches"):
            batch = conversations[i:i+self.batch_size]
            
            try:
                batch_results = self.extract_memories_from_batch(batch)
                
                # Add to pending results
                pending_results.extend(batch_results)
                
                # Count memories
                batch_memories = sum(len(result['memories']) for result in batch_results)
                total_memories += batch_memories
                
                # Update conversations processed
                conversations_processed = i + len(batch)
                
                # Save if we've accumulated enough results or this is the last batch
                if len(pending_results) >= self.save_batch_size or i + self.batch_size >= len(conversations):
                    self.save_memories_batch_to_csv(pending_results, is_first_batch=first_batch)
                    total_results_saved += len(pending_results)
                    
                    # Update checkpoint after successful save
                    self.save_checkpoint(conversations_processed, False, total_memories)
                    
                    # Clear pending results after saving
                    pending_results = []
                    first_batch = False
                
            except Exception as e:
                print(f"Error processing batch at index {i}: {e}")
                print(traceback.format_exc())
                
                # Try to save any successfully processed results so far
                if pending_results:
                    try:
                        self.save_memories_batch_to_csv(pending_results, is_first_batch=first_batch)
                        total_results_saved += len(pending_results)
                        first_batch = False
                    except Exception as save_error:
                        print(f"Failed to save partial results: {save_error}")
                
                # Update checkpoint even if there was an error
                self.save_checkpoint(i, first_batch, total_memories)
        
        return {
            "total_conversations": len(conversations),
            "total_memories_extracted": total_memories,
            "conversations_processed": conversations_processed,
            "total_results_saved": total_results_saved,
            "memory_output_file": self.memory_output_file,
            "status": "completed" if conversations_processed >= len(conversations) else "partial"
        }
    
    def filter_already_processed_conversations(self, conversations: List[pd.DataFrame], metadata: Dict[str, Any]) -> List[pd.DataFrame]:
        """
        Filter out conversations that have already been processed based on their time ranges.
        
        Args:
            conversations: List of conversation dataframes
            metadata: Metadata about already processed conversations
            
        Returns:
            List of conversations that haven't been processed yet
        """
        if not metadata:
            return conversations
            
        # Extract the latest timestamp from previously processed data
        latest_end_time = metadata.get("latest_end_time", 0)
        
        # Get the unique conversation time ranges that have already been processed
        processed_convos = metadata.get("unique_conversations", [])
        processed_time_ranges = set()
        
        for convo in processed_convos:
            # Create a tuple of (start_time, end_time) for fast lookup
            if 'start_time' in convo and 'end_time' in convo:
                processed_time_ranges.add((convo['start_time'], convo['end_time']))
        
        # Filter conversations that haven't been processed yet
        filtered_conversations = []
        
        for convo in conversations:
            start_time = convo['send_time'].min()
            end_time = convo['send_time'].max()
            
            # Skip if this exact conversation has already been processed
            if (start_time, end_time) in processed_time_ranges:
                continue
                
            # Skip if all messages in this conversation are older than the latest processed timestamp
            # This is a safety check to avoid missing conversations when we don't have exact time ranges
            if end_time <= latest_end_time and processed_time_ranges:
                continue
                
            filtered_conversations.append(convo)
        
        skipped_count = len(conversations) - len(filtered_conversations)
        if skipped_count > 0:
            print(f"Skipped {skipped_count} already processed conversations")
            
        return filtered_conversations
        
    def run(self, csv_path: str) -> Dict[str, Any]:
        """
        Run the full memory extraction pipeline with resume capability.
        If an existing output file exists, will resume from where it left off.
        """
        # Check if output file exists and we need to resume
        resuming, metadata = self.check_existing_output()
        
        if resuming:
            print(f"Found existing output file {self.memory_output_file} with {metadata.get('entry_count', 0)} entries")
            print(f"Will resume extraction from where we left off")
        
        # Load and preprocess data
        df = self.load_chat_data(csv_path)
        print(f"Loaded {len(df)} messages")
        
        # Segment into conversations
        all_conversations = self.segment_conversations(df)
        print(f"Segmented into {len(all_conversations)} total conversations")
        
        # Filter out already processed conversations if resuming
        if resuming:
            conversations = self.filter_already_processed_conversations(all_conversations, metadata)
            print(f"After filtering already processed conversations: {len(conversations)} conversations remaining")
        else:
            conversations = all_conversations
        
        if not conversations:
            print("No new conversations to process")
            return {
                "total_conversations": len(all_conversations),
                "total_messages": len(df),
                "new_conversations": 0,
                "resumed": resuming,
                "status": "completed",
                "memory_output_file": self.memory_output_file
            }
        
        # Process conversations with checkpoint/resume capability
        stats = self.process_all_conversations(conversations)
        
        # Add additional stats
        stats["total_messages"] = len(df)
        stats["total_conversations"] = len(all_conversations)
        stats["new_conversations"] = len(conversations)
        stats["resumed"] = resuming
        
        # Clean up checkpoint file if successfully completed
        if stats.get("status") == "completed" and os.path.exists(self.checkpoint_file):
            try:
                os.remove(self.checkpoint_file)
                print("Removed checkpoint file after successful completion")
            except:
                print("Note: Could not remove checkpoint file")
        
        return stats

    def extract_memories(self):
        config = load_config()
        base_dir = config["BASE_DIR"]

        # Load paths
        material_path = os.path.join(base_dir, config["memory_processing"]["material_path"])
        memory_path = os.path.join(base_dir, config["memory_processing"]["memory_path"])

        # Load settings
        api_key = config["memory_processing"]["api_key"]
        memory_model = config["memory_processing"]["model"]
        
        for file in os.listdir(material_path):
            if file.endswith(".csv"):
                print(f"Processing {file}")
                extractor = ChatMemoryExtractor(
                    openai_api_key=api_key,  # Uncomment if not using environment variable
                    model=memory_model,           # Use DeepSeek-Reasoner for best results
                    batch_size=1,            # Process 1 conversations at a time
                    time_window=3600 * 24,    # Group messages within 24 hours
                    save_batch_size=1,         # Save to CSV after processing 1 conversations
                    memory_output_file=os.path.join(memory_path, f"extracted_memories_{file.split('.')[0]}.csv"),  # Output file for extracted memories
                    checkpoint_file=os.path.join(memory_path, f"extraction_checkpoint_{file.split('.')[0]}.json")
                )
            
                results = extractor.run(os.path.join(material_path, file))
                
                # Print summary statistics
                print("\nExtraction Summary:")
                print(f"Total messages processed: {results['total_messages']}")
                print(f"Total conversations: {results['total_conversations']}")
                print(f"New conversations processed: {results['new_conversations']}")
                print(f"Total memories extracted: {results.get('total_memories_extracted', 'N/A')}")
                print(f"Output file: {results['memory_output_file']}")
                print(f"Status: {results['status']}")

if __name__ == "__main__":
    extractor = ChatMemoryExtractor()
    extractor.extract_memories()
