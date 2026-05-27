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


    
