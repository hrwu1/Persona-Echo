import os
import glob
import shutil
import json

from tqdm import tqdm

from src.data.lora_processor import LoraProcessor


class ChatHistoryProcessor:
    def __init__(
            self,
            chat_history_path,
            processed_data_path,
            method,
            delete_tmp_jsonl=False,
            save_intermediate_csv=False,
            combine_interval=600,
            split_interval=4 * 3600,
            turn_num=2,
            stride=2,
            newline_token="<newline>",
            eos_token="<eos>",
            sensitive_words_path=None,
            extra_sticker_path=None,
            include_official_sticker=False,
            include_customized_sticker=False,
            tokenize_emoji=False
    ):
        """
        主要聊天记录处理类
        :param chat_history_path: 原始聊天记录路径
        :param processed_data_path: 处理后数据输出路径
        :param method: 微调任务（当前仅支持lora）
        :param delete_tmp_jsonl: 是否保存中间步jsonl（单人聊天记录的jsonl，由lora_processor生成）
        :param save_intermediate_csv: 保存中间步csv路径，None则不保存（清洗后的单人聊天记录csv，由single_chat_processor生成，没有后续处理）
        :param combine_interval: 单人连续消息的合并间隔（如果为600，则合并10分钟内连续单人消息）
        :param split_interval: 静默时间间隔（如果为4 * 3600，则切分间隔为4h的聊天记录）
        :param turn_num: 对话轮次（单人单次对话为一个turn）
        :param stride: 对话跨度（新数据生成跨多少轮对话，与turn_num的差值最好为2的倍数）
        :param newline_token: 换行符token
        :param eos_token: 中止符token（ChatGLM为<eop>）
        :param sensitive_words_path: 敏感词路径
        :param extra_sticker_path: 自定义表情包路径
        :param include_official_sticker: 是否包含官方表情包
        :param include_customized_sticker: 是否包含自定义表情包
        :param tokenize_emoji: 是否tokenize emoji
        """
        self.chat_history_path = chat_history_path
        self.processed_data_path = processed_data_path
        self.method = method
        self.delete_tmp_jsonl = delete_tmp_jsonl
        self.save_intermediate_csv = save_intermediate_csv
        self.combine_interval = combine_interval
        self.split_interval = split_interval
        self.turn_num = turn_num
        self.stride = stride
        self.newline_token = newline_token
        self.eos_token = eos_token
        self.sensitive_words_path = sensitive_words_path
        self.extra_sticker_path = extra_sticker_path
        self.include_official_sticker = include_official_sticker
        self.include_customized_sticker = include_customized_sticker
        self.tokenize_emoji = tokenize_emoji

        self.clear_output_dir(self.processed_data_path)

    @staticmethod
    def clear_output_dir(path):
        """
        Clear dir or create it if it doesn't exist
        """
        if os.path.exists(path):
            if os.listdir(path):
                shutil.rmtree(path)
                os.makedirs(path)
        else:
            os.makedirs(path)

    def process(self):
        """
        Main process function
        """
        self.process_all_chat()
        print("Finish processing all chat")
        self.merge_jsonl_files()

    def process_all_chat(self):
        """
        Process all contacts' chat
        """
        contact_folder_names = os.listdir(self.chat_history_path)

        for contact_folder_name in tqdm(contact_folder_names):
            if self.method == "LoRA" or "Lora" or "lora":
                processor = LoraProcessor(
                    chat_history_path=self.chat_history_path,
                    processed_data_path=self.processed_data_path,
                    contact_folder_name=contact_folder_name,
                    save_intermediate_csv=self.save_intermediate_csv,
                    combine_interval=self.combine_interval,
                    split_interval=self.split_interval,
                    turn_num=self.turn_num,
                    stride=self.stride,
                    newline_token=self.newline_token,
                    eos_token=self.eos_token,
                    sensitive_words_path=self.sensitive_words_path,
                    extra_sticker_path=self.extra_sticker_path,
                    include_official_sticker=self.include_official_sticker,
                    include_customized_sticker=self.include_customized_sticker,
                    tokenize_emoji=self.tokenize_emoji
                )
                processor.process()

    def merge_jsonl_files(self):
        """
        Merge all jsonl files
        """
        jsonl_dir = os.path.join(self.processed_data_path, "jsonl")
        all_data = []
        jsonl_files = glob.glob(os.path.join(jsonl_dir, "*.jsonl"))

        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    all_data.append(json.loads(line))
        processed_data_path = os.path.join(self.processed_data_path, "conversations.jsonl")
        with open(processed_data_path, "w", encoding="utf-8") as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Finish processing and export to: {processed_data_path} (length: {len(all_data)})")
        if self.delete_tmp_jsonl:
            try:
                shutil.rmtree(jsonl_dir)
            except Exception as e:
                print(f"Failed to delete {jsonl_dir}: {e}")
