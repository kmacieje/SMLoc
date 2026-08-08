# SMLoc Indoor Localization System

Indoor positioning system based on IEEE 802.11 FTM (fine time measurement) ranging from multiple access points, powered by a DNN (deep neural network) machine learning model that estimates the 2D position (x,y) on the indoor floor plan. 

## ⚙ High Level Workflow

1. A client device (station), placed at the measurement point, runs an FTM ranging loop, querying the nearby access points 50 times each.
2. The data is packed into JSON and published to the MQTT topic `ftm/raw_measurements`.
3. A subscriber running on a Windows workstation subscribes to the topic, parses the JSON payload, applies calibration, and builds and normalizes the input feature vector.
4. The station's position is estimated via trained DNN regression model.

## 🌐 Topology

The testbed consists of 6 Intel Joule 570x compute modules, each equipped with an Intel Dual Band Wireless-AC 8260 network adapter:

- 5x configured as **Access Points (AP1-AP5)**
- 1x configured as the **Station (STA)**

## 🏢 Test Environment

Measurements were collected in an office boardroom, **5.35 m x 19.25 m**, giving a floor area of approximately **102 m²**.
Five access points are placed at fixed positions, each at a height of 1 m above the floor. The station was moved between measurement points on a tripod.

## 📁 Repository structure

| File | Description |
|---|---|
| `feature_selection.ipynb` | Google Colab notebook containing the code for two-step feature selection pipeline: (1) correlation analysis to remove redundant features, (2) permutation importance to rank and select the most predictive features |
| `trilateration_baseline.py` | Classic NLS (non-linear least squares) trilateration using the 4 closest APs; code adapted from [1] |
| `dnn_baseline.py` | Baseline DNN model for x/y position regression |
| `smloc_dnn_train.py` | SMLoc DNN training script with offline evaluation on separate test data |
| `ftm_xy_dnn.keras` | SMLoc Keras model definition |
| `ftm_xy_scaler.pkl` | SMLoc scaler fitted on the training data of the SMLoc DNN model |
| `ftm_publisher.py` | Runs FTM ranging requests on the station and publishes raw measurements over MQTT |
| `ftm_subscriber.py` | Subscribes to MQTT, extracts features from raw FTM data, and runs live DNN inference |
| `office_measurements.csv` | Raw, unprocessed reference point measurements collected in the office |
| `office_dataset_mean_std.csv` | Computed features (mean and standard deviation) per AP, derived from raw measurements |
| `office_features.csv` | Final feature set after feature selection: mean and squared mean per AP, used as SMLoc DNN input |

## 📦 Requirements

- Python 3.12.0
- Linux with `hostapd` and `iw`
- MQTT broker (e.g. Mosquitto)

| Library | Version |
|---|---|
| pandas | 2.3.3 |
| numpy | 2.3.4 |
| scikit-learn | 1.8.0 |
| tensorflow | 2.21.0 |
| keras | 3.14.0 |
| optuna | 4.8.0 |
| shap | 0.51.0 |
| matplotlib | 3.10.9 |

Install dependencies:

```bash
pip install -r requirements.txt
```
## 🚀 Setup

### Real-time positioning
On the station (Linux device with FTM support):

```bash
python ftm_publisher.py
```

On the receiving machine (parses incoming measurements and runs live DNN inference):

```bash
python ftm_subscriber.py
```


## 📚 References

[1] S. Huilla, "Smartphone-based Indoor Positioning Using Wi-Fi Fine Timing Measurement Protocol," Master's thesis, University of Turku, Department of Future Technologies, 2019.


