from ultralytics import YOLO

if __name__ == '__main__':
    yolo = YOLO("model")
    yolo.train()