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
- 1x configured as the **Station (STA)**, moved between measurement points

## 🏢 Test Environment

Measurements were collected in an office boardroom, **5.35 m x 19.25 m**.

