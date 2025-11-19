from ultralytics import YOLO
import cv2
import numpy as np
import json
from typing import List, Dict, Any

# imgSource = "data/UnadillaGallery_ML512-085.jpg"
imgSource: str = "data/sexton.avif"
img = cv2.imread(imgSource)

# model = YOLO("yolov8s-pose.pt") # Faster but less accurate model
model = YOLO("yolov8m-pose.pt")
results = model(img, conf=0.2)
r = results[0]

# Keypoints (shape: [num_people, num_keypoints, 3])
kpts = r.keypoints.xy
print("Keypoints shape:", kpts.shape) 

rider = kpts[0]   # assume single rider
if len(kpts) == 1:
    print(rider)
elif len(kpts) > 1:
    print("Multiple people found!")
else:
    print("No person found!")

points: List[List[float]] = rider.tolist()  # convert numpy → python list
output: Dict[str, Any] = {
    "image": imgSource,
    "keypoints": points
}

with open("keypoints.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved keypoints!")


# r.show()

