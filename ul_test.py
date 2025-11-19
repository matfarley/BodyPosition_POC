from ultralytics import YOLO
import cv2
import numpy as np
import json
from typing import List, Dict, Any
from src.angles import hip_angle, knee_angle, elbow_angle, torso_angle

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

filename = "keypoints.json"
with open(filename, "w") as f:
    json.dump(output, f, indent=2)

print("Saved keypoints! as", filename)

# r.show()

# Do stuff with angles --------------
with open(filename) as f:
    data = json.load(f)

loaded_kp = data["keypoints"]

# Full COCO keypoint mapping (0-16)
NOSE           = loaded_kp[0]
LEFT_EYE       = loaded_kp[1]
RIGHT_EYE      = loaded_kp[2]
LEFT_EAR       = loaded_kp[3]
RIGHT_EAR      = loaded_kp[4]
LEFT_SHOULDER  = loaded_kp[5]
RIGHT_SHOULDER = loaded_kp[6]
LEFT_ELBOW     = loaded_kp[7]
RIGHT_ELBOW    = loaded_kp[8]
LEFT_WRIST     = loaded_kp[9]
RIGHT_WRIST    = loaded_kp[10]
LEFT_HIP       = loaded_kp[11]
RIGHT_HIP      = loaded_kp[12]
LEFT_KNEE      = loaded_kp[13]
RIGHT_KNEE     = loaded_kp[14]
LEFT_ANKLE     = loaded_kp[15]
RIGHT_ANKLE    = loaded_kp[16]

print("Hip angle:", hip_angle(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE))
print("Knee angle:", knee_angle(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE))
print("Elbow angle:", elbow_angle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST))
print("Torso angle:", torso_angle(LEFT_SHOULDER, LEFT_HIP))

