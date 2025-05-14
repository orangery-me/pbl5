import firebase_admin
import numpy as np
from firebase_admin import credentials, db
from xuly import process_data, feature_buffer
from tensorflow.keras.models import load_model
from datetime import datetime

cred = credentials.Certificate("demo.json")  
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://fir-2402-default-rtdb.asia-southeast1.firebasedatabase.app/'  
})

# load model
model = load_model("LSTM_model.h5")
new_sub_data = []
existing_key = None
user_id = "user_abc"
today = datetime.now().strftime('%d/%m/%Y')  # Ví dụ: "2025-05-14"

def convert_np_datetime64_to_str(np_datetime):
    py_datetime = np_datetime.astype('M8[ms]').astype(datetime)
    return py_datetime.isoformat()

# def send_result_to_firebase(activity, start_time, end_time, key):
#     result_ref = db.reference("result")
    
#     if isinstance(start_time, np.datetime64):
#         start_time = convert_np_datetime64_to_str(start_time)
#     if isinstance(end_time, np.datetime64):
#         end_time = convert_np_datetime64_to_str(end_time)
    
#     result_data = {
#         "activity": activity,
#         "start_time": start_time,
#         "end_time": end_time
#     }

#     # Gửi kết quả lên Firebase, dùng start_time làm key
#     result_ref.child(str(key)).set(result_data)
    

def on_firebase_data(event):
    data = event.data
    path = event.path
    key = path.strip("/")
    
    if key is None:
        return
    
    if data is None:
        return
    
    # Kiểm tra xem dữ liệu có phải là dict hay không
    if isinstance(data, list):
        handle_data(data, key)
        
    elif isinstance(data, dict):
        for _, value in data.items():
            if (isinstance(key, str)):
                handle_data(value, key)
    else:
        print("Received data is not in a correct format", data)


def handle_data(raw_data, key):
    
    try:
        
        X, start_time, end_time = process_data(raw_data)
         
        #  X is an array of sequences
        #  Duyệt qua từng sequence trong X
        for i in range(X.shape[0]):
            sequence = X[i]
            reshaped_sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
        
            # Dự đoán hành vi
            prediction = model.predict(reshaped_sequence)
            predicted_class = int(np.argmax(prediction[0]))
            print("Dự đoán hành vi: ", predicted_class, "Start time:", start_time, "End time:", end_time)
            
            send_result_to_firebase(predicted_class, start_time, end_time, key)

    except Exception as e:
        print("Error:", e)

def send_result_to_firebase(activity, start_time, end_time, key):
    result_ref = db.reference("activity_records")
    all_records = result_ref.get()
    
    if isinstance(start_time, np.datetime64):
        start_time = convert_np_datetime64_to_str(start_time)
    if isinstance(end_time, np.datetime64):
        end_time = convert_np_datetime64_to_str(end_time)
        
    result_data = {
        "activity": activity,
        "start_time": start_time,
        "end_time": end_time
    }
        
    if all_records:
        # Kiểm tra trong all_records có tồn tại existing key không
        if existing_key in all_records:
            return
        else:
            for _, value in all_records.items():
                if value.get('user_id') == user_id and value.get('date') == today:
                    existing_key = value.get('key')
                    break
            
    if existing_key:
        records_ref = result_ref.child(f"{existing_key}/records")
        current_records = records_ref.get() or []
        current_records.append(result_data)
    else:
        new_object = {
            "user_id": user_id,
            "date": today,
            "records": [result_data]
        }
        records_ref = result_ref.push(new_object)
        existing_key = records_ref.key

def main(): 
    data_ref = db.reference("dataCollection") 
    data_ref.listen(on_firebase_data)
    
    
if __name__ == "__main__":
    main()
    
