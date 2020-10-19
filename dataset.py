import torch
from torch.utils.data import Dataset


class CustomTextDataset(Dataset):
    def __init__(self, texts, labels, dictionary):
        # Unknown Token is index 1 (<UNK>)
        self.x = [[dictionary.get(token, 1) for token in token_list] for token_list in texts]
        self.y = labels

    def __len__(self):
        """Return the data length"""
        return len(self.x)

    def __getitem__(self, idx):
        """Return one item on the index"""
        return self.x[idx], self.y[idx]


def collate_fn(data, args, pad_idx=0):
    """Padding"""
    texts, labels = zip(*data)
    texts = [s + [pad_idx] * (args.max_len - len(s)) if len(s) < args.max_len else s[:args.max_len] for s in texts]
    return torch.LongTensor(texts), torch.LongTensor(labels)