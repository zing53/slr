import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

def train_knn():
    # 读取提取的特征CSV文件
    csv_path = "./data/processed_hand_landmarks.csv"
    data = pd.read_csv(csv_path)

    X = data.iloc[:, 1:21].values  # 特征列(d_1 to d_20)
    # X = np.array(X, dtype='float')
    y = data.iloc[:, 21].values  # 标签列(label)

    # 归一化特征数据到 [0,1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_normalized = scaler.fit_transform(X)

    # 将归一化器保存到文件，以便在推理时使用
    joblib.dump(scaler, './model/distance_scaler.pkl')

    k_values = range(1,20,2)
    accuracy_scores = []
    time_records = []

    # 5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        
        start_time = time.time()
        scores = cross_val_score(knn, X_normalized, y, cv=skf, scoring='accuracy')
        end_time = time.time()
        avg_time = (end_time - start_time) / skf.get_n_splits()
        
        accuracy_scores.append(scores.mean())
        time_records.append(avg_time)

    for k, acc, avg_t in zip(k_values, accuracy_scores, time_records):
        print(f"K={k}: 平均准确率={acc:.4f}, 平均验证时间={avg_t:.4f} 秒")


    # 选择最佳的k值训练模型并保存
    best_k = k_values[np.argmax(accuracy_scores)]
    knn_slr = KNeighborsClassifier(n_neighbors=best_k)
    knn_slr.fit(X_normalized, y)
    joblib.dump(knn_slr, './model/knn_slr_model.pkl')
    print(f'Best K: {best_k} - 模型已保存至knn_slr_model.pkl和distance_scaler.pkl')

train_knn()