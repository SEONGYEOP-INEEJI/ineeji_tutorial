# 테스트 베드 설정. True로 설정하면 테스트 베드 설정을 사용하고, False로 설정하면 일반 설정을 사용합니다.
test_bed = True

data_conf = {
    'path' : 'data/california_housing.csv',
}
# 모델 설정. NLINEAR 모델에 대한 설정을 담고 있습니다.
model_conf = {
    "NLINEAR": {
        "window_size": 10,  # 윈도우 크기. 이전 몇 개의 데이터를 입력으로 사용할지 결정합니다.
        "forecast_size": 1,  # 예측 크기. 몇 개의 데이터를 예측할지 결정합니다.
        "individual": False,  # 개별 예측 여부. True로 설정하면 각 예측이 독립적이게 됩니다.
    }
}

# 학습 설정. 배치 크기, 에포크 수, 학습률, 조기 종료를 위한 인내심 등을 설정합니다.
train_conf = {
    "batch_size": 32,  # 배치 크기. 한 번에 처리할 데이터의 수를 결정합니다.
    "epochs": 100,  # 에포크 수. 전체 데이터셋에 대한 학습 반복 횟수를 결정합니다.
    "learning_rate": 0.001,  # 학습률. 파라미터 업데이트 속도를 결정합니다.
    "patience": 20,  # 조기 종료를 위한 인내심. 검증 손실이 이 횟수만큼 개선되지 않으면 학습이 종료됩니다.
    'checkpoint_path': 'models/checkpoint.pth',  # 체크포인트 경로. 학습 중간 중간 모델을 저장할 경로를 설정합니다.
    'model_path': 'models/model.pth'  # 모델 경로. 학습이 끝난 후 최종 모델을 저장할 경로를 설정합니다.
}

# 테스트 베드를 위한 학습 설정. 일반 학습 설정과 비슷하지만 에포크 수가 1로 설정되어 있습니다.
train_conf_testbed = {
    "batch_size": 32,  # 배치 크기. 한 번에 처리할 데이터의 수를 결정합니다.
    "epochs": 1,  # 에포크 수. 전체 데이터셋에 대한 학습 반복 횟수를 결정합니다.
    "learning_rate": 0.001,  # 학습률. 파라미터 업데이트 속도를 결정합니다.
    "patience": 20,  # 조기 종료를 위한 인내심. 검증 손실이 이 횟수만큼 개선되지 않으면 학습이 종료됩니다.
    'checkpoint_path': 'models/checkpoint.pth',  # 체크포인트 경로. 학습 중간 중간 모델을 저장할 경로를 설정합니다.
    'model_path': 'models/model.pth'  # 모델 경로. 학습이 끝난 후 최종 모델을 저장할 경로를 설정합니다.
}