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

---

## 참고 자료 및 레퍼런스

### 논문
- **"Acoustic Howling Suppression with Neural Networks using Real-world Data"**  
  Jonah Stein, Ryan Corey, Andrew Singer — University of Illinois Urbana-Champaign  
  본 프로젝트의 신경망 기반 하울링 억제 접근법의 이론적 배경으로 참고.  
  실제 공연 환경 데이터를 활용한 딥러닝 모델 설계 방식을 참조함.

- **"Prediction Error Method based Adaptive Feedback Cancellation (PEM-AFC)"**  
  보청기 분야에서 개발된 피드백 예측 기반 적응 필터 제어 기법.  
  예측 오차를 최소화하는 적응 필터 구조를 본 프로젝트의 레이블링 전략 설계에 참조함.

### 오픈소스 코드
- **Acoustic Feedback Suppression (GLD 기반 AFC 구현)**  
  [https://github.com/ReidRR92/Acoustic-Feedback-Suppression](https://github.com/ReidRR92/Acoustic-Feedback-Suppression)  
  Generalized Levinson-Durbin (GLD) 알고리즘을 활용한 Android용 C 구현체.  
  `dataset_gen.py`와 `predict_audio.py`의 `levinson_durbin()` 함수 및 자기상관 기반 피처 추출 방식이 이 코드의 `GLD.c`에서 영감을 받아 Python으로 재구현됨.  
  구체적으로 참고한 부분:
  - `pmxcorr()` → 자기상관(rxx) 계산 방식
  - `GLD()` → Levinson-Durbin 재귀 알고리즘 구조
  - `canceller()` → FIR 필터 적용 원리
