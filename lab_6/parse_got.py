import json
import os
import re
import pickle


def clean_text(text):
    if not text:
        return None

    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^]]*\]', '', text)
    text = text.lower()
    text = re.sub(r"\b'(\w+)", r" '\1", text)
    
    contractions = {
        r"\b're\b": " are",
        r"\b's\b": " is",
        r"\b'm\b": " am",
        r"\b've\b": " have",
        r"\b'll\b": " will",
        r"\b'd\b": " would",
        r"\bn't\b": " not",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text and len(text) > 2 else None


def extract_pairs_from_season(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        season_data = json.load(f)
    
    pairs = []
    for episode_name, subtitles_dict in season_data.items():

        sorted_keys = sorted(subtitles_dict.keys(), key=int)
        lines = []
        for k in sorted_keys:
            raw = subtitles_dict[k]
            cleaned = clean_text(raw)
            if cleaned:
                lines.append(cleaned)
        
        # Создаём пары: (текущая реплика, следующая)
        for i in range(len(lines) - 1):
            src = lines[i]
            tgt = lines[i + 1]
            # Фильтруем слишком короткие или странные
            if len(src) > 2 and len(tgt) > 2:
                pairs.append((src, tgt))
    return pairs

# Собираем все пары со всех сезонов
all_pairs = []
data_dir = "game-of-thrones-srt"

for filename in os.listdir(data_dir):
    if filename.endswith(".json"):
        full_path = os.path.join(data_dir, filename)
        print(f"Processing {filename}...")
        pairs = extract_pairs_from_season(full_path)
        all_pairs.extend(pairs)

print(f"Total dialogue pairs: {len(all_pairs)}")



with open("got_pairs.pkl", "wb") as f:
    pickle.dump(all_pairs, f)
