import os
import json

from src.data.basic_processor import BasicProcessor


class LoraProcessor(BasicProcessor):
    """
    Process for LoRA
    """
    def __init__(
            self,
            chat_history_path,
            processed_data_path,
            contact_folder_name,
            save_intermediate_csv,
            combine_interval,
            split_interval,
            turn_num,
            stride,
            newline_token,
            eos_token,
            sensitive_words_path,
            extra_sticker_path,
            include_official_sticker,
            include_customized_sticker,
            tokenize_emoji
    ):
        super().__init__(
            chat_history_path,
            processed_data_path,
            contact_folder_name,
            save_intermediate_csv,
            combine_interval,
            newline_token,
            sensitive_words_path,
            extra_sticker_path,
            include_official_sticker,
            include_customized_sticker,
            tokenize_emoji
        )
        self.split_interval = split_interval
        self.turn_num = turn_num
        self.stride = stride
        self.eos_token = eos_token

        self.conv_json_list = []

    def further_process(self, intermediate_data):
        """
        For LoRA
        """
        segments_sender = self.split_conv_by_sender(intermediate_data)
        for seg_sender in segments_sender:
            if not seg_sender:
                continue
            sub_data_sender = intermediate_data.loc[seg_sender[0]:seg_sender[1]]
            if sub_data_sender.empty:
                continue
            segments_time = self.split_conv_by_time(sub_data_sender)

            for seg_time in segments_time:
                if not seg_time:
                    continue
                sub_data_time = intermediate_data.loc[seg_time[0]:seg_time[1]]
                if sub_data_time.empty:
                    continue
                self.split_conv_for_lora(sub_data_time)
        if self.conv_json_list:
            self.save_jsonl()

    @staticmethod
    def split_conv_by_sender(df):
        """
        Split chat history by (continuous) sender
        """
        if df is None or df.empty:
            return []

        index_list = df.index.tolist()
        segments = []
        start_idx = 0
        for i in range(1, len(df)):
            if df.iloc[i]["sender"] == df.iloc[i - 1]["sender"]:
                segments.append((index_list[start_idx], index_list[i - 1]))
                start_idx = i
        segments.append((index_list[start_idx], index_list[-1]))
        return segments

    def split_conv_by_time(self, df):
        """
        Split chat history by time
        """
        if df.empty:
            return []

        index_list = df.index.tolist()
        segments = []
        start_idx = 0

        for i in range(1, len(df)):
            time_diff = df.iloc[i]["send_time"] - df.iloc[i - 1]["send_time"]
            if time_diff > self.split_interval:
                segments.append((index_list[start_idx], index_list[i - 1]))
                start_idx = i

        segments.append((index_list[start_idx], index_list[-1]))
        return segments

    def split_conv_for_lora(self, df):
        """
        Split chat history for lora (ChatGLM version)
        """
        if df.empty:
            return []

        df = df.sort_values(by="send_time").reset_index(drop=True)
        if len(df) <= 1:
            return

        start_idx = 0
        while start_idx < len(df) and df.iloc[start_idx]["sender"] == 1:
            start_idx += 1

        for i in range(start_idx, len(df) - self.turn_num + 1, self.stride):
            dialog_slice = df.iloc[i:i + self.turn_num]
            if dialog_slice.iloc[0]["sender"] == 1:
                continue
            if dialog_slice.iloc[-1]["sender"] != 1:
                continue
            messages = []
            for j, row in dialog_slice.iterrows():
                role = "assistant" if row["sender"] == 1 else "user"
                content = row["msg"]
                if j == len(dialog_slice) - 1 and row["sender"] == 1:
                    content += self.eos_token
                messages.append({"role": role, "content": content})
            self.conv_json_list.append({"conversation": messages})

    def save_jsonl(self):
        """
        Convert list of json to jsonl and save
        """
        file_folder = os.path.join(self.processed_data_path, "jsonl")
        os.makedirs(file_folder, exist_ok=True)
        with open(os.path.join(file_folder, f"{self.contact_folder_name}.jsonl"), "w", encoding="utf-8") as f:
            for conv_json in self.conv_json_list:
                f.write(json.dumps(conv_json, ensure_ascii=False) + "\n")
