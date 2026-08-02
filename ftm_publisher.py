import paho.mqtt.client as mqtt
import subprocess
import json
import time
import re
import sys

BROKER_IP = "192.168.1.100"  # workstation IP address
BROKER_PORT = 1883
TOPIC = "ftm/raw_measurements"

INTERFACE = "wlp1s0"
ITERATIONS = 50

AP_LIST = ["AP1", "AP2", "AP3", "AP4", "AP5"]

FTM_REGEX = re.compile(r"Target:\s*([\w:]+),\s*status:\s*(\d+),\s*rtt:\s*(-?\d+)\s*psec,\s*distance:\s*(-?\d+)\s*cm")

def get_ftm_measurements(conf_file):
    measurements = []

    for i in range(ITERATIONS):
        sys.stdout.write("\r  Progress: {} / {}".format(i + 1, ITERATIONS))
        sys.stdout.flush()

        cmd = ["iw", INTERFACE, "measurement", "ftm_request", conf_file]
        result = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = result.communicate()
        output = stdout.decode('utf-8')

        match = FTM_REGEX.search(output)
        if match:
            mac = match.group(1)
            status = int(match.group(2))
            rtt = int(match.group(3))
            dist = int(match.group(4))

            measurements.append({
                "status": status,
                "rtt_psec": rtt,
                "distance_cm": dist
            })
        else:
            measurements.append({"status": -1, "rtt_psec": 0, "distance_cm": 0})

        time.sleep(0.1)

    print()
    return measurements

def main():
    client = mqtt.Client()
    client.connect(BROKER_IP, BROKER_PORT, 60)
    client.loop_start()

    point_id = 1

    while True:
        print("Measueremnt point #{}".format(point_id))
        x_str = input("Enter x coordinate (or 'q' to quit): ")
        if x_str.lower() == 'q':
            break
        y_str = input("Enter y coordinate: ")

        payload = {
            "timestamp": time.time(),
            "point_id": point_id,
            "coords": {"x": float(x_str), "y": float(y_str)},
            "measurements": {}
        }

        for ap_name in AP_LIST:
            print("Querying {}".format(ap_name))
            ap_data = get_ftm_measurements(ap_name)
            payload["measurements"][ap_name] = ap_data

        json_data = json.dumps(payload)
        client.publish(TOPIC, json_data)
        print("Data sent for point ({}, {})\n".format(x_str, y_str))

        point_id += 1

    client.loop_stop()
    client.disconnect()

if __name__ == "__main__":
    main()