import cv2
import ultralytics as ul
import torch

print("OpenCV version:", cv2.__version__)
print("ultralytics version:", ul.__version__)
print("Torch version:", torch.__version__)

# Quick torch test
x = torch.rand(3, 3)
print("Torch tensor OK:\n", x)
