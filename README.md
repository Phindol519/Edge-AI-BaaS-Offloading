# Edge-AI-BaaS-Offloading
Performance analysis of BaaS(Supabase)-based data offloading architecture for AI inference in edge devices
Edge AI Offloading with BaaS (Supabase)
이 프로젝트는 제한된 엣지 컴퓨팅 환경(Jetson Nano, Raspberry Pi)에서 발생하는 데이터 I/O 병목 현상을 해결하기 위해, 기존 로컬 DB 방식 대신 BaaS(Supabase)를 도입한 오프로딩 아키텍처를 연구하고 성능을 분석한 프로젝트입니다.

1. Overview
엣지 AI 기기에서 AI 추론 중 발생하는 대량의 데이터 기록 작업은 기기의 SD 카드 I/O 부하를 가중시켜 전체적인 AI 처리량(FPS)을 저하시킵니다.
본 프로젝트는 이 병목을 네트워크 계층으로 분산(Offloading)하여, 기기 본연의 AI 추론 성능을 최적화하는 데 목적이 있습니다.

3. System Architecture
Local Storage: SQLite, MariaDB

Cloud Offloading: Supabase (PostgreSQL-based BaaS)

AI Framework: YOLOv5

Target Devices: NVIDIA Jetson Nano

3. Performance Analysis
각 DB 방식에 따른 자원 점유율(CPU, GPU, RAM) 및 성능(Latency, FPS)을 비교 분석하였습니다.

분석 데이터 (2x2 Metrics):

CPU/GPU Usage Correlation

Memory Consumption

End-to-End Latency

Average Throughput (FPS)

Key Result: Supabase 기반의 오프로딩 아키텍처는 기존 로컬 방식 대비 네트워크 오버헤드에도 불구하고, AI 추론 파이프라인의 일관된 안정성과 향상된 FPS를 보장함을 확인하였습니다.

4. Repository Structure
Plaintext
├── src/            # 메인 소스 코드
├── logs/           # 기기별 성능 측정 로그 (.txt, .csv)
├── docs/           # 분석 결과 PDF 그래프 및 논문 서류
└── README.md
