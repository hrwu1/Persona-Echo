import os
import re
import shutil
import emoji
import pandas as pd
from configs.constants import WECHAT_EMOJI_STR


def match_sticker(
        text,
        processed_data_path,
        folder_path,
        extra_sticker_map,
        extra_sticker_path,
        include_official_sticker,
        include_customized_sticker
):
    """
    Tokenize a sticker from the conversation and return the corresponding <sticker_x> tag.
    """

    def copy_sticker(source_path, md5, save_dir):
        ext = os.path.splitext(source_path)[1]
        sticker_filename = f"sticker_{md5}{ext}"
        save_path = os.path.join(save_dir, sticker_filename)
        if not os.path.exists(save_path):
            shutil.copy(source_path, save_path)
        return f"<sticker_{md5}>"

    if not isinstance(text, str):
        text = str(text) if not pd.isna(text) else ""
    save_dir = os.path.join(processed_data_path, "sticker")
    os.makedirs(save_dir, exist_ok=True)

    md5_pattern = re.compile(r'<emoji[^>]*?\smd5\s*=\s*"([^"]+)"')
    md5_matches = md5_pattern.findall(text)
    if not md5_matches:
        return None
    md5 = md5_matches[0]
    extensions = [".png", ".gif", ".jpg", ".jpeg"]

    # Official stickers
    if include_official_sticker:
        for ext in extensions:
            file_path = os.path.join(folder_path, "emoji", f"{md5}{ext}")
            if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
                return copy_sticker(file_path, md5, save_dir)

    # Customized stickers
    if not extra_sticker_map and not include_customized_sticker:
        return None
    if md5 in extra_sticker_map:
        extra_md5 = extra_sticker_map[md5]
        for ext in extensions:
            extra_file_path = os.path.join(extra_sticker_path, f"{extra_md5}{ext}")
            if os.path.isfile(extra_file_path) and os.path.getsize(extra_file_path) > 0:
                return copy_sticker(extra_file_path, extra_md5, save_dir)

    return None


def match_emoji(text, token2emoji):
    """
    Tokenize both WeChat and system emojis, updating the token2emoji mapping.
    """
    if not isinstance(text, str):
        text = str(text) if not pd.isna(text) else ""
    wechat_emojis = re.findall(r"\[(.*?)]", text)
    for wechat_emoji in wechat_emojis:
        if wechat_emoji in WECHAT_EMOJI_STR:
            token = f"<wechat_{wechat_emoji}>"
            token2emoji[token] = f"[{wechat_emoji}]"
            text = text.replace(f"[{wechat_emoji}]", token)

    new_text = []
    for char in text:
        if emoji.is_emoji(char):
            emoji_name = emoji.demojize(char).strip(":")
            token = f"<system_{emoji_name}>"
            token2emoji[token] = char
            new_text.append(token)
        else:
            new_text.append(char)
    return "".join(new_text)
