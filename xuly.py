import pandas as pd
import numpy as np
import json
import joblib
from collections import deque
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

def load_data(source):
    with open(source, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data

def extract_features(data):
    return [
        np.mean(data),  # Giá trị trung bình
        np.std(data),  # Độ lệch chuẩn
        np.sqrt(np.mean(np.square(data))),  # Căn bậc hai trung bình bình phương (RMS)
        np.max(data),  # Biên độ lớn nhất
        np.min(data),  # Biên độ nhỏ nhất
        np.median(data),  # Trung vị
        np.sum(np.diff(np.sign(data)) != 0),  # Số lần đổi dấu
        skew(data),  # Độ lệch
        kurtosis(data),  # Độ nhọn
        np.percentile(data, 25),  # Q1
        np.percentile(data, 75),  # Q3
        np.correlate(data, data, mode='full')[len(data)-1]  # Tự tương quan
    ]


# Tạo các cửa sổ thời gian
def create_windows(data, window_size=50, overlap=0.5):
    step = int(window_size * (1 - overlap))
    data_len = data.shape[0]
    return np.array([data[i:i + window_size] for i in range(0, data_len - window_size + 1, step)])

# Lấy thời gian bắt đầu và kết thúc của các cửa sổ
def get_start_end_time(data, window_size=50, overlap=0.5):
    step = int(window_size * (1 - overlap))
    data_len = data.shape[0]
    start_times = data[0 : data_len - window_size + 1 : step]
    end_times = data[window_size - 1 : data_len : step]
    return start_times, end_times

                
def prepare_data_for_lstm(buffer, sequence_length=3):
    # Chuyển đổi dữ liệu thành định dạng phù hợp cho LSTM (queue to list)
    buffer_list = list(buffer)
    sequences = []
    
    if len(buffer_list) < sequence_length:
        print("Buffer length is less than sequence length. Returning empty array.")
        return np.array(sequences)
    
    for i in range(len(buffer) - sequence_length + 1):
        seq = np.stack(buffer_list[i:sequence_length + i], axis=0)
        sequences.append(seq)
    return np.array(sequences)

def normalize_data(data):
    scaler = joblib.load('scaler.pkl')
    print("mean :", scaler.mean_)
    data = scaler.transform(data)
    return data

def process_data(data, window_size=50, overlap=0.5):
    # print('data:', data)
    session_data = data

    # Chuyển list of dict thành DataFrame
    df = pd.json_normalize(session_data)

    # Lấy các cột cảm biến
    acc_x = df["acceleration.x"].astype(float).values
    acc_y = df["acceleration.y"].astype(float).values
    acc_z = df["acceleration.z"].astype(float).values
    vec_x = df["rotation.x"].astype(float).values
    vec_y = df["rotation.y"].astype(float).values
    vec_z = df["rotation.z"].astype(float).values
    timestamp = pd.to_datetime(df["timestamp"]).values
    # print("Timestamp:", timestamp)

    # Chia dữ liệu thành các cửa sổ (5 giây)
    acc_x_win = create_windows(acc_x, window_size, overlap)
    acc_y_win = create_windows(acc_y, window_size, overlap)
    acc_z_win = create_windows(acc_z, window_size, overlap)
    vec_x_win = create_windows(vec_x, window_size, overlap)
    vec_y_win = create_windows(vec_y, window_size, overlap)
    vec_z_win = create_windows(vec_z, window_size, overlap)
    # start_times, end_times = get_start_end_time(timestamp, window_size, overlap)
    start_times, end_times = timestamp[0], timestamp[-1]

    # Trích xuất đặc trưng từ mỗi cửa sổ
    all_window_features = []
    for k in range(acc_x_win.shape[0]):  # Lặp qua từng cửa sổ
        window_features = (
            extract_features(acc_x_win[k]) +
            extract_features(acc_y_win[k]) +
            extract_features(acc_z_win[k]) +
            extract_features(vec_x_win[k]) +
            extract_features(vec_y_win[k]) +
            extract_features(vec_z_win[k])
        )
        all_window_features.append(window_features)
        
    normalized_features = normalize_data(np.array(all_window_features))
    
    for feature in normalized_features:
        feature_buffer.append(feature)
    
    X = prepare_data_for_lstm(feature_buffer, sequence_length = 3)

    return X, start_times, end_times

feature_buffer = deque(maxlen=3)

def main(): 
    window_size = 50
    overlap = 0.5
    
    # Giả lập dữ liệu từ ESP32
    data = load_data("mock_data.json")
    # Xử lý dữ liệu và tạo các cửa sổ
    X = process_data(data, window_size, overlap)
    
    # load model
    model = load_model("LSTM_model.h5")
    
    # Dự đoán hành vi
    prediction = model.predict(X)
    predicted_class = int(np.argmax(prediction[0]))
    print("Dự đoán hành vi:", predicted_class)

#  main
if __name__ == "__main__":
    main()

