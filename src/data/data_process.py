import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config.config_loader import load_config
from src.data.chat_history_processor import ChatHistoryProcessor


class DataProcessor:
    def __init__(self):
        """Initialize the DataProcessor with configuration."""
        self.config = load_config()
        self.base_dir = self.config["BASE_DIR"]
        
        # Load paths
        self.chat_history_path = os.path.join(self.base_dir, self.config["paths"]["chat_history"])
        self.processed_data_path = os.path.join(self.base_dir, self.config["paths"]["processed_data"])
        self.extra_sticker_path = os.path.join(self.base_dir, self.config["paths"]["extra_sticker"])
        self.sensitive_words_path = os.path.join(self.base_dir, self.config["paths"]["sensitive_words"])
        
        # Load settings
        self.method = self.config["settings"]["method"]
        self.delete_tmp_jsonl = self.config["settings"]["delete_tmp_jsonl"]
        self.save_intermediate_csv = self.config["settings"]["save_intermediate_csv"]
        self.include_official_sticker = self.config["settings"]["include_official_sticker"]
        self.include_customized_sticker = self.config["settings"]["include_customized_sticker"]
        self.tokenize_emoji = self.config["settings"]["tokenize_emoji"]
        
        # Load processing parameters
        self.combine_interval = self.config["data_processing"]["combine_interval"]
        self.split_interval = self.config["data_processing"]["split_interval"]
        self.turn_num = self.config["data_processing"]["turn_num"]
        self.stride = self.config["data_processing"]["stride"]
        self.newline_token = self.config["data_processing"]["newline_token"]
        self.eos_token = self.config["data_processing"]["eos_token"]
    
    def run(self):
        """Run the chat history processing."""
        processor = ChatHistoryProcessor(
            chat_history_path=self.chat_history_path,
            processed_data_path=self.processed_data_path,
            method=self.method,
            delete_tmp_jsonl=self.delete_tmp_jsonl,
            save_intermediate_csv=self.save_intermediate_csv,
            combine_interval=self.combine_interval,
            split_interval=self.split_interval,
            turn_num=self.turn_num,
            stride=self.stride,
            newline_token=self.newline_token,
            sensitive_words_path=self.sensitive_words_path,
            eos_token=self.eos_token,
            extra_sticker_path=self.extra_sticker_path,
            include_official_sticker=self.include_official_sticker,
            include_customized_sticker=self.include_customized_sticker,
            tokenize_emoji=self.tokenize_emoji
        )
        processor.process()


if __name__ == "__main__":
    processor = DataProcessor()
    processor.run()
