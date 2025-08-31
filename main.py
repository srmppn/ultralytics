from ultralytics import YOLO

if __name__ == '__main__':
    yolo = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")
    yolo.train(data="sandbox/train_model/data.yaml")