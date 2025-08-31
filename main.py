from ultralytics import YOLO

if __name__ == '__main__':
    yolo = YOLO("research/yolov8s_trained_with_drone.pt")
    yolo.train(data="sandbox/train_model/data.yaml")