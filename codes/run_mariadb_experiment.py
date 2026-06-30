import cv2
import torch
import time
import threading
import csv
import requests
from datetime import datetime

# ==========================================
# 1. 설정 및 경로
# ==========================================
YOLO_DIR = '/home/jinwoo/yolov5'
WEIGHTS_PATH = '/home/jinwoo/yolov5/yolov5s.pt'
VIDEO_PATH = '/home/jinwoo/test_video.mp4'
SPRING_URL = "http://localhost:8081/api/detections"

def send_to_spring(label, conf, fps):
    payload = {"label": label, "confidence": float(conf), "fps": float(fps)}
    try:
        requests.post(SPRING_URL, json=payload, timeout=1)
    except:
        pass

# ==========================================
# 2. 모델 로딩 (가라 해결법)
# ==========================================
print("🚀 [MariaDB Mode] YOLOv5 모델 로딩 중...")
model = torch.hub.load(YOLO_DIR, 'custom', path=WEIGHTS_PATH, source='local')

cap = cv2.VideoCapture(VIDEO_PATH)
experiment_logs = []
prev_time = time.time()

print(f"🔥 실험 시작! (CSV 포맷: 이미지와 동일하게 맞춤)")

# ==========================================
# 3. 메인 추론 루프
# ==========================================
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        start_infer = time.time()
        results = model(frame)
        df = results.pandas().xyxy[0]

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        latency_ms = (curr_time - start_infer) * 1000

        if not df.empty:
            label = df.iloc[0]['name']
            conf = df.iloc[0]['confidence']
            
            threading.Thread(target=send_to_spring, args=(label, conf, fps)).start()

            # [수정] 이미지 포맷과 동일하게 시:분:초.밀리초 형식으로 저장
            formatted_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            experiment_logs.append([formatted_time, 'MariaDB_Local', round(latency_ms, 2), round(fps, 2)])

        # 줄줄이 로그 보고 싶으시면 end='\r'을 뺍니다.
        print(f"🔥 [MariaDB] FPS: {fps:.1f} | 인식: {df.iloc[0]['name'] if not df.empty else 'None'}")

except KeyboardInterrupt:
    print("\n🛑 중단됨")

finally:
    with open('result_mariadb.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Version', 'Latency_ms', 'FPS'])
        writer.writerows(experiment_logs)
    cap.release()
    print("\n🏁 실험 종료! 'result_mariadb.csv' 확인 바랍니다.")
