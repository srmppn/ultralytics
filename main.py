from ultralytics import YOLO

import cv2

if __name__ == '__main__':
    yolo = YOLO("ultralytics/cfg/models/v8/yolov8.yaml").load("sandbox/yolov8n.pt")
    yolo.train(data="sandbox/train_model_3/data.yaml", epochs=100, imgsz=640)
    # results = yolo.predict("sandbox/test_3.png")
    # annotated_frame = results[0].plot()
    # cv2.imshow("Detections", annotated_frame)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()