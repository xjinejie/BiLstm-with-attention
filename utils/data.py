from collections import Counter
import torch
from torch.utils.data import Dataset
import json
# 根据输入的文本列表构建词表
class Vocab:
    def __init__(self, tokens, max_size, min_freq): # 根据tokens列表构建词表
        self.pad_token = '<PAD>' # padding token，id=0
        self.unk_token = '<UNK>' # unknown token，id=1
        all_tokens = []
        for text in tokens:
            all_tokens.extend(text.lower().split())
        token_counts = Counter(all_tokens) # 统计词频，key: token, value: count
        # 过滤低频词并按频率排序
        valid_tokens = [t for t, c in token_counts.items() if c >= min_freq]
        valid_tokens = sorted(valid_tokens, key=lambda x: token_counts[x], reverse=True)[:max_size]
        
        self.token2id = {self.pad_token: 0, self.unk_token: 1}
        for idx, token in enumerate(valid_tokens):
            self.token2id[token] = idx + 2
            
    def __len__(self):
        return len(self.token2id)
    
    def convert_text_to_ids(self, text, max_len): # 将文本转换为ID序列
        tokens = text.lower().split()
        ids = [self.token2id.get(t, self.token2id[self.unk_token]) for t in tokens]
        if len(ids) > max_len:
            return ids[:max_len]
        else:
            return ids + [self.token2id[self.pad_token]] * (max_len - len(ids))


# 自定义模型的输入数据和标签
class YelpDataset(Dataset): # 自定义数据集
    def __init__(self, data, vocab, max_len):
        self.data = data # list of (text, label)
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx): # 定义模型输入样本和标签
        text, label = self.data[idx]
        input_ids = self.vocab.convert_text_to_ids(text, self.max_len)
        # 假设label是 1-5 星，转为 0-4
        label = int(label) - 1 
        return torch.tensor(input_ids), torch.tensor(label)
    

# 从数据集文件中加载数据
def load_data(path, limit=None): # 读取JSON数据，形成 (text, label) 列表
    """读取JSON数据。limit用于调试时限制数据量"""
    data = []
    print(f"Loading data from {path}...")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit: break
                item = json.loads(line)
                # 假设字段名为 'text' 和 'stars'
                if 'text' in item and 'stars' in item:
                    data.append((item['text'], item['stars']))
    except FileNotFoundError:
        print(f"Error: File {path} not found.")
    return data