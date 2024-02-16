from sklearn.datasets import fetch_california_housing
from config import model_conf, test_bed, train_conf, train_conf_testbed, data_conf
data = fetch_california_housing(as_frame=True)
data_frame = data.frame

# 데이터를 CSV 파일로 저장
data_frame.to_csv(data_conf['path'], index=False)