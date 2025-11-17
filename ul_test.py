from ultralytics import YOLO
import cv2

model = YOLO("yolov8s-pose.pt")

img = cv2.imread("data/sexton.avif")

results = model(img, conf=0.2)
keypoints = results[0].keypoints
print(keypoints.xy) 

# results[0].show()
results[0].show().skeleton()
