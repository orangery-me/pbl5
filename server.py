import firebase_admin
import numpy as np
from firebase_admin import credentials, db
from xuly import process_data, feature_buffer
from tensorflow.keras.models import load_model
from datetime import datetime

cred = credentials.Certificate("healthy_app.json")  
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://healthyapp-15f68-default-rtdb.firebaseio.com/'  
})

# load model
model = load_model("LSTM_model.h5")
new_sub_data = []
existing_key = None
user_id = "user_1029357990"
today = datetime.now().strftime('%d/%m/%Y') 

def convert_np_datetime64_to_str(np_datetime):
    py_datetime = np_datetime.astype('M8[ms]').astype(datetime)
    return py_datetime.isoformat()

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
            predicted_class = int(np.argmax(prediction[0])) + 1
            print("Dự đoán hành vi: ", predicted_class, "Start time:", start_time, "End time:", end_time)
            
            send_result_to_firebase(predicted_class, start_time, end_time, key)

    except Exception as e:
        print("Error:", e)

def send_result_to_firebase(activity, start_time, end_time, key):
    # global existing_key
    result_ref = db.reference("activity_records")
    all_records = result_ref.get()
    
    if isinstance(start_time, np.datetime64):
        start_time = convert_np_datetime64_to_str(start_time)
    if isinstance(end_time, np.datetime64):
        end_time = convert_np_datetime64_to_str(end_time)
        
    result_data = {
        "activityType": activity,
        "start_time": start_time,
        "end_time": end_time
    }
    
    existing_key = None
    # if not all_records or not existing_key in all_records:
        # print("Key not found in all records")
    if all_records:
        for new_key, value in all_records.items():
            print("Checking key:", new_key)
            if value.get('user_id') == user_id and value.get('date') == today:
                existing_key = new_key
                break
            
    if existing_key:
        records_ref = result_ref.child(f"{existing_key}/records")
        current_records = records_ref.get() 
        current_records.append(result_data)
        records_ref.set(current_records)
    else:
        new_object = {
            "user_id": user_id,
            "date": today,
            "records": [result_data]
        }
        start_time_dt = datetime.fromisoformat(start_time)
        new_key = start_time_dt.strftime("%Y%m%d")
        records_ref = result_ref.child(new_key)
        records_ref.set(new_object)
        existing_key = records_ref.key
        
def on_location_change(event):
    data = event.data
    path = event.path
    key = path.strip("/")
    print(f"[INFO] Location change detected: key = {key}, data = {data}")
    
    if path == "/":
        return
    
    print(f"[INFO] New location change detected: key = {key}, data = {data}")

    # Gửi notification ở đây
    # send_notification_to_device(data)  # Bạn cần định nghĩa hàm này
    

data_ref = db.reference("dataCollection") 
data_ref.listen(on_firebase_data)

# listen to location document and push notification if location changes
location_ref = db.reference("location")
location_ref.listen(on_location_change)
    
