import torch
import cv2
import requests
import time
import threading
import os
import csv
from datetime import datetime

# 1. 실험 데이터 저장용 리스트
experiment_logs = []

# 2. 서버 전송 및 성능 측정 함수
def send_to_spring(label, confidence, current_fps):
    url = "http://localhost:8081/api/send"
    data = {
        "label": label,
        "confidence": float(confidence)
    }
    
    start_time = time.time() # 전송 시작 시간 기록
    try:
        res = requests.post(url, json=data, timeout=2.0)
        end_time = time.time() # 응답 도착 시간 기록
        
        if res.status_code == 200:
            latency = (end_time - start_time) * 1000 # ms 단위 변환
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            
            # 로그 기록: [시간, 버전, 지연시간(ms), 현재FPS]
            experiment_logs.append([timestamp, "Supabase", round(latency, 2), round(current_fps, 2)])
            print(f"   >> [DB] 저장 성공! Latency: {latency:.1f}ms")
    except Exception as e:
        print(f"   >> [Error] 전송 실패: {e}")

# 3. 환경 설정
video_file = '/home/jinwoo/test_video.mp4' # 경로 꼭 확인!
if not os.path.exists(video_file):
    print(f"❌ 파일을 찾을 수 없습니다: {video_file}")
    exit()

print("🚀 모델 로딩 중 (CUDA 사용)...")
model = torch.hub.load('.', 'custom', path='yolov5n.pt', source='local')
model.to('cuda')

cap = cv2.VideoCapture(video_file)
prev_time = 0
frame_count = 0

print("✅ 버전 B(Supabase) 실험 시작...")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame)
        df = results.pandas().xyxy[0]
        
        # FPS 계산
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        if not df.empty:
            label = df.iloc[0]['name']
            conf = df.iloc[0]['confidence']
            threading.Thread(target=send_to_spring, args=(label, conf, fps)).start()
            frame_count = 0

        print(f"🔥 FPS: {fps:.1f} | 인식: {df.iloc[0]['name'] if not df.empty else 'None'}", end='\r')

except KeyboardInterrupt:
    print("\n🛑 사용자에 의해 실험 중단")

finally:
    # 4. CSV 파일 저장 (그래프 생성용)
    with open('result_supabase.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Version', 'Latency_ms', 'FPS'])
        writer.writerows(experiment_logs)
    
    cap.release()
    print("\n🏁 실험 종료! 'result_supabase.csv' 파일이 저장되었습니다.")

