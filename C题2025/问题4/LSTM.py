import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def load_and_preprocess_data(file_path):
    """
    加载数据并进行预处理
    """
    data = pd.read_csv(file_path)
    feature_columns = [
        '孕妇BMI', '原始读段数', '在参考基因组上比对的比例', '重复读段的比例', 
        '唯一比对的读段数', 'GC含量', '13号染色体的Z值', '18号染色体的Z值', 
        '21号染色体的Z值', 'X染色体的Z值', 'X染色体浓度', 
        '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量', 
        '被过滤掉读段数的比例'
    ]
    target_column = '染色体的非整倍体'
    data = data.dropna(subset=feature_columns + [target_column])
    X = data[feature_columns].values
    y = data[target_column].values
    print(f"数据分布: 0 - {np.sum(y == 0)}, 1 - {np.sum(y == 1)}")
    return X, y, feature_columns

def prepare_lstm_data(X, y):
    """
    准备LSTM模型所需的数据格式
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    return X_reshaped, y, scaler

def build_lstm_model(input_shape):
    """
    构建LSTM模型
    """
    model = Sequential()
    model.add(LSTM(32, input_shape=input_shape, return_sequences=True, kernel_regularizer='l2'))
    model.add(Dropout(0.4))
    model.add(LSTM(16, return_sequences=False, kernel_regularizer='l2'))
    model.add(Dropout(0.4))
    model.add(Dense(8, activation='relu', kernel_regularizer='l2'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_training_history(history):
    """
    绘制训练历史
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='训练准确率')
    ax1.plot(history.history['val_accuracy'], label='验证准确率')
    ax1.set_title('模型准确率')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax2.plot(history.history['loss'], label='训练损失')
    ax2.plot(history.history['val_loss'], label='验证损失')
    ax2.set_title('模型损失')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    plt.tight_layout()
    plt.show()

def main():
    file_path = '附件_wash_nv.csv'
    try:
        # 1. 加载和预处理数据
        X, y, feature_columns = load_and_preprocess_data(file_path)
        X_reshaped, y, scaler = prepare_lstm_data(X, y)

        # 2. 使用K折交叉验证
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold = 1
        all_f1_scores = []

        for train_index, test_index in skf.split(X_reshaped, y):
            print(f"\n=== 第 {fold} 折交叉验证 ===")
            X_train, X_test = X_reshaped[train_index], X_reshaped[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 3. 构建模型
            model = build_lstm_model((1, X_train.shape[2]))
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

            # 4. 训练模型
            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=4,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1
            )

            # 5. 评估模型
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
            f1 = f1_score(y_test, y_pred)
            all_f1_scores.append(f1)

            print(f"第 {fold} 折 F1分数: {f1:.4f}")
            fold += 1

        print(f"\n平均F1分数: {np.mean(all_f1_scores):.4f}")

        # 6. 绘制最后一折的训练历史
        plot_training_history(history)

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()