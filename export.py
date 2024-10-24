import sys
import os

sys.path.append("../..")
from ultralytics import YOLO

model = YOLO("yolov8n-obb.yaml")  # build a new model from YAML
model.model.model[-1].export_for_xc01 = True  # remove post-process ops
results = model.export(format="onnx", opset=12)


os.system("onnxsim yolov8n-obb.onnx yolov8n-obb.onnx")
os.system("sudo sh docker_build.sh")

os.system("cp outputs/nnp310/yolov8n-obb_nnp310_combine.ty ./")
os.system("python3 auto_test.py")
