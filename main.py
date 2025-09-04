from ultralytics import YOLO

import cv2
import torch

if __name__ == '__main__':
    yolo = YOLO("research/keep/weights/best.pt").load("sandbox/yolov8n.pt")
    yolo.train(data="sandbox/train_model/data.yaml", epochs=1, imgsz=640)

    # results = yolo.predict("sandbox/train_model/valid/images/img000001.jpg", conf=0.6, iou=0.5)
    # annotated_frame = results[0].plot()
    #
    # cv2.imshow("Detections", annotated_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
