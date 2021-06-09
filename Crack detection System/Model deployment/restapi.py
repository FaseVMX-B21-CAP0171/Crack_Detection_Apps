"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io
import os
import base64

from flask.json import jsonify

import torch
from PIL import Image
from flask import Flask, request
from datetime import datetime

app = Flask(__name__)

DETECTION_URL = "/v1/object-detection/yolov5s"

@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.json.get("image"):
        image_b64 = request.json["image"]
        image_file = base64.b64decode(image_b64)
        image_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + 'image_input'

        image_file.save(r'./inference/images/'+image_name)

        # img = Image.open(io.BytesIO(image_bytes))

        # results = model(img, size=640)  # reduce size=320 for faster inference

        # !python detect.py --weights ./weights/best_yolov5x_21_1820.pt --conf 0.01 --source inference/images/image_file  --img 1024 # --view-img
        os.system("detect.py --weights ./weights/best_yolov5x_21_1820.pt --conf 0.01 --source inference/images/"+image_name+" --img 1024 # --view-img")

        #get the result directory
        result_dir = ...

        #encode the image
        img = Image.open(r"./"+result_dir)
        img_b64 = base64.b64encode(img)

        array = {
            'img': img_b64
        }

        return jsonify(array)

        # return results.pandas().xyxy[0].to_json(orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True)  # force_reload to recache
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat




