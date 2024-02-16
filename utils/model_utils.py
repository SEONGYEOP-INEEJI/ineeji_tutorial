import torch
from torch import nn, optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


# 손실 함수를 반환하는 함수
def get_loss():
    return nn.MSELoss()


# 최적화 도구를 반환하는 함수
def get_opt(model,lr):
    return optim.Adam(model.parameters(), lr=lr)


# 모델 학습 함수
def train_model(model, criterion, optimizer, dataloader):
    model.train()  # 모델을 학습 모드로 설정
    for inputs, targets in dataloader:  # 데이터 로더에서 배치 단위로 데이터를 가져옴
        outputs = model(inputs)  # 모델에 입력을 전달하여 출력을 계산
        loss = criterion(outputs, targets)  # 출력과 타깃 사이의 손실을 계산
        optimizer.zero_grad()  # 이전에 계산된 기울기를 제거
        loss.backward()  # 손실에 대한 기울기를 계산
        optimizer.step()  # 모델의 파라미터를 업데이트


# 모델 평가 함수
def evaluate_model(model, dataloader, criterion):
    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():  # 기울기 계산을 비활성화
        total_loss = 0
        for inputs, targets in dataloader:  # 데이터 로더에서 배치 단위로 데이터를 가져옴
            outputs = model(inputs)  # 모델에 입력을 전달하여 출력을 계산
            loss = criterion(outputs, targets)  # 출력과 타깃 사이의 손실을 계산
            total_loss += loss.item()  # 전체 손실을 계산
    return total_loss / len(dataloader)  # 평균 손실을 반환


# 모델의 성능을 평가하고 결과를 출력하는 함수
def evaluate_and_print_metrics(model, test_dataset):
    test_inputs = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    test_labels = torch.stack([test_dataset[i][1] for i in range(len(test_dataset))])
    test_predictions = model(test_inputs).detach().numpy()
    test_labels = test_labels.numpy()
    mse = mean_squared_error(test_labels, test_predictions)
    mae = mean_absolute_error(test_labels, test_predictions)
    print(f"Test MSE: {mse}, Test MAE: {mae}")


# 학습 과정을 시각화하는 함수
def plot_learning_curve(train_losses, valid_losses):
    plt.plot(train_losses, label="Training loss")
    plt.plot(valid_losses, label="Validation loss")
    plt.legend()
    plt.show()


# 모델을 저장하는 함수
def save_model(model, path):
    torch.save(model.state_dict(), path)
