import json
import paho.mqtt.client as mqtt
import time

class NodeRedLogger:
    def __init__(self, run_id):
        self.run_id = run_id
        self.client = mqtt.Client()
        self.client.connect("localhost", 1880)

    def send(self, pose, readings):
        payload = {
            "t": time.time(),
            "x": pose[0],
            "y": pose[1],
            "yaw": pose[2],
            "sensors": readings
        }
        self.client.publish(f"vacuum/{self.run_id}/telemetry", json.dumps(payload))