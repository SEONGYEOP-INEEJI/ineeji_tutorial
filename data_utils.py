import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


# 데이터 로딩
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data


# 전처리 (NaN 값 제거)
def preprocess_data(data):
    data = data.dropna()
    return data


def scale(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data, scaler


# PyTorch Dataset
class RegressionDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = torch.tensor(data.values, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self):
        # window_size를 고려하여 길이를 조정합니다.
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size, :-1]  # window_size 만큼의 이전 데이터
        y = self.data[idx + self.window_size, -1:]  # 현재의 타겟 데이터
        return x, y


# 데이터셋 분할
def train_valid_test_split(dataset, train_ratio=0.7, valid_ratio=0.15):
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    valid_size = int(total_size * valid_ratio)
    test_size = total_size - train_size - valid_size

    train_dataset = dataset[:train_size]
    valid_dataset = dataset[train_size : train_size + valid_size]
    test_dataset = dataset[train_size + valid_size :]

    return train_dataset, valid_dataset, test_dataset
