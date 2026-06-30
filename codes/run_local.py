# -*- coding: utf-8 -*-
import torch
import cv2
import requests
import time
import os
import csv

# --- 수파베이스(버전 B) 때와 동일하게 설정 ---
video_file = '/home/jinwoo/test_video.mp4' 
SERVER_URL = "http://localhost:8081/api/send" # 저장소만 로컬로!
CSV_FILE = "result_local.csv"

if not os.path.exists(video_file):
    print(f"❌ 파일을 찾을 수 없습니다: {video_file}")
    exit()

print("🚀 모델 로딩 중 (CUDA 사용/Local Source)...")
# 핀돌님이 말씀하신 바로 그 '치트키' 문장입니다.
# 현재 폴더(.)에서 로컬 소스로 yolov5n.pt를 커스텀 로드!
model = torch.hub.load('.', 'custom', path='yolov5n.pt', source='local')
model.to('cuda')

cap = cv2.VideoCapture(video_file)
print("✅ 버전 A(Local SQLite) 실험 시작...")

# CSV 초기화
with open(CSV_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Timestamp', 'Label', 'Confidence', 'Latency_ms', 'FPS'])

try:
    while cap.isOpened():
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # 수파베이스 때와 동일한 추론 방식
        results = model(frame)
        detections = results.pandas().xyxy[0]

        for _, row in detections.iterrows():
            if row['confidence'] >= 0.5:
                label = row['name']
                conf = row['confidence']

                # 로컬 8081 포트로 전송 및 지연 측정
                req_start = time.time()
                try:
                    payload = {"label": label, "confidence": float(conf)}
                    response = requests.post(SERVER_URL, json=payload, timeout=0.5)
                    
                    latency = (time.time() - req_start) * 1000
                    fps = 1.0 / (time.time() - loop_start)

                    if response.status_code == 200:
                        print(f"✅ [Local] {label} 저장: {latency:.2f}ms")
                        with open(CSV_FILE, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), label, conf, latency, fps])
                except Exception as e:
                    print(f"⚠️ 전송 오류: {e}")

finally:
    cap.release()
    print(f"🏁 로컬 실험 종료! 결과: {CSV_FILE}")
