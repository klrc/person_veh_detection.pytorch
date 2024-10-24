from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n-obb.yaml")  # build a new model from YAML
# model = YOLO("yolov8n-obb.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolov10n.yaml").load("yolov10n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="fullhan-pedveh.yaml", epochs=100, imgsz=640)
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
