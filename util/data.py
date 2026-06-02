import sys
sys.path.append('..')
from util.util import load_jsonl
from torch.utils.data import Dataset


# Dataset for Stage 2 training and test
class SelectQADataset(Dataset):
    def __init__(self, data_path):
        self.texts = load_jsonl(data_path)
        print('Loaded Dataset')
        print(f'Size of dataset: {len(self.texts)}')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]


# Dataset for Stage 1 training
class SelectTrainDataset(Dataset):
    def __init__(self, data_path):
        self.texts = load_jsonl(data_path)
        print('Loaded Dataset')
        print(f'Size of dataset: {len(self.texts)}')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]


# Dataset for Stage 3 joint alignment training.
# Supports optional difficulty-level filtering (design doc §4.1):
#   - If the data has a 'difficulty' field, samples with difficulty < min_difficulty are dropped.
#   - min_difficulty=0 (default) keeps all samples.
#   - Per the design doc, for Stage 3 use min_difficulty=3 to keep only
#     high-difficulty samples (Difficulty Level 3 & 4) where joint alignment
#     has the greatest effect.
# Data format: same as Stage 2 — each record has 'question', 'documents', 'answer'.
# The 'answer' field may be a string (Stage 2 style) or a list (NQ eval style);
# the collator handles both.
class Stage3Dataset(Dataset):
    def __init__(self, data_path, min_difficulty: int = 0):
        all_texts = load_jsonl(data_path)
        if min_difficulty > 0:
            before = len(all_texts)
            self.texts = [
                item for item in all_texts
                if item.get('difficulty', min_difficulty) >= min_difficulty
            ]
            print(f'Difficulty filter (>= {min_difficulty}): {before} → {len(self.texts)} samples')
        else:
            self.texts = all_texts
        print(f'Stage3Dataset size: {len(self.texts)}')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]


