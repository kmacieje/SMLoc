import paho.mqtt.client as mqtt
import json
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

BROKER_IP = "127.0.0.1"
BROKER_PORT = 1883
TOPIC = "ftm/raw_measurements"

CALIBRATION_FUNCTIONS = {
    "ap1": lambda x: 0.469624797627423 * x - 20.4477178365299,
    "ap2": lambda x: 35.0644039737743 * np.exp(0.00267882112577182 * x),
    "ap3": lambda x: 0.468084129872179 * x - 59.8722207484994,
    "ap4": lambda x: 0.489266941326667 * x - 66.7191748377351,
    "ap5": lambda x: 0.556245953645641 * x - 96.3411865166566
}

scaler = joblib.load('ftm_xy_scaler.pkl')
dnn_model = load_model('ftm_xy_dnn.keras')

def calibrate_and_extract_features(ap_name, measurements_list):
    cal_func = CALIBRATION_FUNCTIONS.get(ap_name, lambda x: x)
    valid_calibrated_distances = []

    for m in measurements_list:
        raw_dist = m.get("distance_cm")
        if raw_dist != 0:
            valid_calibrated_distances.append(cal_func(raw_dist))

    if not valid_calibrated_distances:
        return None, None

    feature_1_mean = np.mean(valid_calibrated_distances)
    feature_2_mean_sq = feature_1_mean ** 2
    return round(feature_1_mean, 6), round(feature_2_mean_sq, 6)

def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode('utf-8'))
    point_id = data.get('point_id')
    x, y = data['coords']['x'], data['coords']['y']

    print(f"\nPoint #{point_id} (x: {x}, y: {y})")

    ml_input_vector = {}
    for ap_name, measurements in data.get('measurements', {}).items():
        f1_mean, f2_mean_sq = calibrate_and_extract_features(ap_name, measurements)

        if f1_mean is not None:
            ml_input_vector[f"{ap_name}_mean"] = f1_mean
            ml_input_vector[f"{ap_name}_mean_sq"] = f2_mean_sq
            print(f"[{ap_name}] mean: {f1_mean} cm | mean^2: {f2_mean_sq}")
        else:
            print(f"[{ap_name}] dropped. all measurements invalid")
            ml_input_vector[f"{ap_name}_mean"] = -999
            ml_input_vector[f"{ap_name}_mean_sq"] = -999

    ordered_features = []
    for i in range(1, 6):
        ordered_features.append(ml_input_vector.get(f"ap{i}_mean", -999))
    for i in range(1, 6):
        ordered_features.append(ml_input_vector.get(f"ap{i}_mean_sq", -999))

    input_array = np.array(ordered_features).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = dnn_model.predict(scaled_input, verbose=0)
    pred_x, pred_y = prediction[0][0], prediction[0][1]

    print(f"feature vector: {ordered_features}")
    print(f"DNN prediction: x = {pred_x:.6f} | y = {pred_y:.6f}")

    error_distance = np.sqrt((x - pred_x) ** 2 + (y - pred_y) ** 2)
    print(f"distance error: {error_distance:.6f} m")

if __name__ == "__main__":
    client = mqtt.Client()
    client.on_message = on_message
    client.connect(BROKER_IP, BROKER_PORT, 60)
    print("Listening for FTM data...")
    client.subscribe(TOPIC)
    client.loop_forever()