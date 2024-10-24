import os

from ultralytics import YOLO

model = YOLO("yolov10n.yaml").load("runs/detect/train/weights/best.pt")  # build from YAML and transfer weights
model.model.model[-1].export_for_intellif = True  # remove post-process ops
results = model.export(format="onnx", opset=12)

os.system("onnxsim yolov10n.onnx yolov10n.onnx")
os.system("mv yolov10n.onnx yolov10n_personveh.onnx")
