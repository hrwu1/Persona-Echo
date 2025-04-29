import os
import re
import json
import pandas as pd

from src.data.multimedia_process import match_sticker, match_emoji


class BasicProcessor:
    """
    This class is deprecated and split into two classes: TextProcessor and MetadataProcessor.
    """
    def __init__(
            self,
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
    ):
        self.chat_history_path = chat_history_path
        self.processed_data_path = processed_data_path
        self.contact_folder_name = contact_folder_name
        self.save_intermediate_csv = save_intermediate_csv
        self.combine_interval = combine_interval
        self.newline_token = newline_token
        self.sensitive_words_path = sensitive_words_path
        self.include_official_sticker = include_official_sticker
        self.include_customized_sticker = include_customized_sticker
        self.extra_sticker_path = extra_sticker_path
        self.tokenize_emoji = tokenize_emoji

        self.folder_path = str(os.path.join(self.chat_history_path, self.contact_folder_name))
        self.token2emoji = self.load_token_json(os.path.join(self.processed_data_path, "token2emoji.json"))
        self.friend_name = self.extract_friend_name()

        # Sensitive words
        self.sensitive_words = self.load_sensitive_words(self.sensitive_words_path)
        # Sticker
        self.extra_sticker_map = self.load_token_json(os.path.join(self.extra_sticker_path, "extra_sticker.json"))

    @staticmethod
    def load_token_json(json_file_path):
        """
        Load token json
        """
        if not os.path.exists(json_file_path):
            return {}
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    def load_sensitive_words(self, file_path):
        """
        Load sensitive words from a JSON file.
        The file should contain a list of dictionaries, each with keys "word" and "type".
        For example: [{"word": "XXXX", "type": "name"}, {"word": "1234567890", "type": "phone"}]
        """
        if not os.path.exists(file_path):
            return []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except:
            return []

    def extract_friend_name(self):
        """
        Extract friend name from folder name
        """
        match = re.match(r"(.+?)\(([^()]*)\)$", self.contact_folder_name)
        if match and "@chatroom" not in match.group(2):  # Skip chatroom
            return match.group(1)
        return None

    def mask_sensitive_word(self, text):
        """
        Mask sensitive words in text using the loaded sensitive words list.
        Each occurrence of a sensitive word is replaced with a mask token that includes its type.
        For example, if the word is "Haoran" and its type is "name", it is replaced with "[MASK:name]".
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        masked_text = text
        for entry in self.sensitive_words:
            word = entry.get("word", "")
            type_label = entry.get("type", "")
            if word:
                # Use regex for case-insensitive replacement
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                mask_token = f"[MASK:{type_label}]" if type_label else "[MASK]"
                masked_text = pattern.sub(mask_token, masked_text)
        return masked_text

    def generate_intermediate_csv(self):
        """
        Generate cleaned chat history
        """
        if not self.friend_name:
            return None

        def process_row(r):
            """
            Process each row
            """
            msg = ""
            row_msg = r["StrContent"]
            if r["Type"] == 1:
                # First mask sensitive words then process emojis
                msg = self.mask_sensitive_word(row_msg)
                if self.tokenize_emoji:
                    msg = match_emoji(msg, self.token2emoji)
            elif r["Type"] == 47 and (self.include_customized_sticker or self.include_official_sticker):
                msg = match_sticker(
                    text=row_msg,
                    processed_data_path=self.processed_data_path,
                    folder_path=self.folder_path,
                    extra_sticker_map=self.extra_sticker_map,
                    extra_sticker_path=self.extra_sticker_path,
                    include_official_sticker=self.include_official_sticker,
                    include_customized_sticker=self.include_customized_sticker,
                )
            return r["IsSender"], msg, int(r["CreateTime"])

        csv_path = os.path.join(self.folder_path, f"{self.friend_name}.csv")
        data = pd.read_csv(csv_path)
        output = []

        pre_sender, pre_msg, pre_time = None, "", None

        for index, row in data.iterrows():
            crt_sender, crt_msg, crt_time = process_row(row)
            if not crt_msg:
                continue
            crt_msg = str(crt_msg)
            if crt_sender == pre_sender:
                if pre_time is not None and (crt_time - pre_time) <= self.combine_interval:
                    pre_msg = (pre_msg + self.newline_token + crt_msg).strip()
                    pre_time = crt_time
                else:
                    if pre_msg:
                        output.append([pre_sender, pre_msg, pre_time])
                    pre_msg, pre_time = crt_msg, crt_time
            else:
                if pre_msg:
                    output.append([pre_sender, pre_msg, pre_time])
                pre_sender, pre_msg, pre_time = crt_sender, crt_msg, crt_time

        if pre_msg:
            output.append([pre_sender, pre_msg, pre_time])

        output_df = pd.DataFrame(output, columns=["sender", "msg", "send_time"])
        if self.save_intermediate_csv:
            output_csv_path = os.path.join(self.processed_data_path, "csv")
            os.makedirs(output_csv_path, exist_ok=True)
            output_df.to_csv(os.path.join(output_csv_path, f"{self.contact_folder_name}.csv"), index=False)

        with open(os.path.join(self.processed_data_path, "token2emoji.json"), "w", encoding="utf-8") as f:
            json.dump(self.token2emoji, f, ensure_ascii=False, indent=4)

        return output_df

    def further_process(self, intermediate_data):
        """
        For further processing
        """
        pass

    def process(self):
        """
        Main processing function
        """
        output_df = self.generate_intermediate_csv()
        self.further_process(output_df)
