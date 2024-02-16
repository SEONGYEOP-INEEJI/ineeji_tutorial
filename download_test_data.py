from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
data_frame = data.frame

# 데이터를 CSV 파일로 저장
data_frame.to_csv('california_housing.csv', index=False)