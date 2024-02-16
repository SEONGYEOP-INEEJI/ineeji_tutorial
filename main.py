import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 사용자 정의 모듈 임포트
from config import model_conf, test_bed, train_conf, train_conf_testbed
from data_utils import (
    load_data,
    preprocess_data,
    train_valid_test_split,
    RegressionDataset,
)
from early_stopping import EarlyStopping
from model_utils import (
    get_loss,
    get_opt,
    train_model,
    evaluate_model,
    plot_learning_curve,
    save_model,
    evaluate_and_print_metrics,
)
from models import NLINEAR

# prepare data
if __name__ == "__main__":
    # 모델 설정 불러오기
    if test_bed:
        _model_conf, _train_conf = model_conf["NLINEAR"], train_conf_testbed
    else:
        _model_conf, _train_conf = model_conf["NLINEAR"], train_conf
    window_size, forecast_size, individual = (
        _model_conf["window_size"],
        _model_conf["forecast_size"],
        _model_conf["individual"],
    )

    # 데이터 불러오기 및 전처리
    data = load_data("california_housing.csv")
    data = preprocess_data(data)
    train_dataset, valid_dataset, test_dataset = train_valid_test_split(data)

    # 데이터셋 생성
    train_dataset = RegressionDataset(train_dataset, window_size=window_size)
    valid_dataset = RegressionDataset(valid_dataset, window_size=window_size)
    test_dataset = RegressionDataset(test_dataset, window_size=window_size)

    # 데이터 로더 생성
    train_dataloader = DataLoader(
        train_dataset, batch_size=_train_conf["batch_size"], shuffle=True
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=_train_conf["batch_size"])
    test_dataloader = DataLoader(test_dataset, batch_size=_train_conf["batch_size"])

    # 모델 및 최적화 도구 준비
    model = NLINEAR(window_size, forecast_size, individual, data.shape[1] - 1)
    optimizer = get_opt(model, lr=_train_conf["learning_rate"])
    criterion = get_loss()

    # 학습률 스케줄러와 EarlyStopping 설정
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=10)
    early_stopping = EarlyStopping(patience=_train_conf["patience"], verbose=True)

    train_losses = []
    valid_losses = []

    # 학습 시작
    for epoch in range(_train_conf["epochs"]):
        train_loss = train_model(model, criterion, optimizer, train_dataloader)
        valid_loss = evaluate_model(model, valid_dataloader, criterion)

        # 학습 및 검증 손실 저장
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        scheduler.step(valid_loss)
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Validation Loss: {valid_loss}")

    # 모델 로드
    model.load_state_dict(torch.load("checkpoint.pt"))

    # 학습 곡선 그리기
    plot_learning_curve(train_losses, valid_losses)

    # 테스트 세트에서 성능 평가
    evaluate_and_print_metrics(model, test_dataset)

    # 모델 저장
    save_model(model, "model.pth")
