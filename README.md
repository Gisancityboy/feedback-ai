# Feedback AI — 하울링 예측 AI 시스템

공연 음향 시스템에서 하울링 발생을 **사전에 예측**하여 선제 차단하는 AI 연구 프로젝트

## 핵심 아이디어

기존 하울링 억제 시스템은 **발생 후 대응(reactive)** 방식.  
본 프로젝트는 루프 게인 빌드업 패턴을 AI로 감지하여 **발생 전 선제 차단**을 목표로 함.

```
[마이크 신호] → [실시간 스펙트럼 분석] → [AI 위험도 예측] → [사전 게인 감소]
```

## 위험도 분류

신호의 에너지 증가율을 기반으로 3단계 분류:

| 레이블 | 의미 | 에너지 추세 |
|--------|------|------------|
| 안전 | 하울링 가능성 없음 | 감소 중 |
| 주의 | 게인 빌드업 시작 | 유지 또는 약간 증가 |
| 위험 | 하울링 임박 | 빠르게 증가 |

## 파일 구조

```
feedback-ai/
├── simulator.py          # 하울링 현상 시뮬레이터
├── dataset_gen.py        # AI 학습 데이터 생성
├── model.py              # 신경망 모델 구조
├── train.py              # 모델 학습
├── predict_audio.py      # 오디오 파일 위험도 분석
├── generate_test_audio.py # 테스트 오디오 생성
└── demo.ipynb            # 결과 시각화 노트북
```

## 실행 방법

```bash
# 가상환경 활성화
source ../feedback_ai/bin/activate

# 1. 데이터 생성
python dataset_gen.py

# 2. 모델 학습
python train.py

# 3. 오디오 파일 분석
python predict_audio.py
```

## 기술 스택

- Python 3.11
- PyTorch (신경망 학습)
- NumPy / SciPy (신호처리)
- Librosa / Matplotlib (오디오 분석 및 시각화)

## 개발 단계

| 단계 | 내용 | 상태 |
|------|------|------|
| Phase 1 | 시뮬레이션 환경 구축 + 데이터 생성 | ✅ 완료 |
| Phase 2 | AI 모델 학습 + 오디오 파일 예측 검증 | ✅ 완료 |
| Phase 3 | 실시간 마이크 연동 + DSP 제어 | 🔄 진행 예정 |
