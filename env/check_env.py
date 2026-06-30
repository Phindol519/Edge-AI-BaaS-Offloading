import torch
import cv2
print(f"CUDA Available: {torch.cuda.is_available()}") # 반드시 True여야 함
print(f"OpenCV Version: {cv2.__version__}")
