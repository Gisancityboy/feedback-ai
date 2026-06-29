import numpy as np
import os

SAMPLE_RATE = 44100
DURATION = 1.0
SEGMENT_LENGTH = 1024

def pink_noise(n):
    f = np.fft.rfftfreq(n)
    f[0] = 1
    power = 1 / np.sqrt(f)
    power[0] = 0
    phase = np.random.uniform(0, 2 * np.pi, len(f))
    spectrum = power * np.exp(1j * phase)
    return np.fft.irfft(spectrum, n)

def multi_reflection_filter(signal, n_reflections=4):
    output = signal.copy()
    for _ in range(n_reflections):
        delay = np.random.randint(int(0.005 * SAMPLE_RATE), int(0.08 * SAMPLE_RATE))
        decay = np.random.uniform(0.1, 0.4)
        if delay < len(signal):
            output[delay:] += signal[:-delay] * decay
    return output

def add_impulse(signal):
    n_impulses = np.random.randint(0, 4)
    for _ in range(n_impulses):
        pos = np.random.randint(0, len(signal))
        amp = np.random.uniform(0.1, 0.5)
        width = np.random.randint(10, 100)
        impulse = amp * np.exp(-np.arange(width) / (width * 0.2))
        end = min(pos + width, len(signal))
        signal[pos:end] += impulse[:end - pos]
    return signal

def simulate_feedback(gain, distance):
    n_samples = int(SAMPLE_RATE * DURATION)
    delay_samples = int((distance / 343.0) * SAMPLE_RATE)

    output = np.zeros(n_samples)
    t = np.linspace(0, DURATION, n_samples)

    n_freqs = np.random.randint(3, 10)
    freqs = np.random.choice([80,120,180,250,350,500,700,1000,1500,2000,3000,4000,6000,8000], n_freqs, replace=False)
    amplitudes = np.random.uniform(0.1, 1.0, n_freqs)
    input_signal = sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(freqs, amplitudes))

    envelope = np.random.uniform(0.2, 1.0, 20)
    envelope = np.interp(np.linspace(0, 19, n_samples), np.arange(20), envelope)
    input_signal *= envelope
    input_signal += pink_noise(n_samples) * np.random.uniform(0.05, 0.3)
    input_signal = add_impulse(input_signal)
    input_signal *= 0.01
    input_signal = multi_reflection_filter(input_signal)

    for i in range(n_samples):
        feedback = output[i - delay_samples] * gain if i >= delay_samples else 0
        output[i] = np.clip(input_signal[i] + feedback, -10, 10)

    return output

def measure_energy_trend(signal):
    # 신호를 4구간으로 나눠서 에너지 추세 측정
    n = len(signal)
    q = n // 4
    energies = [np.mean(signal[i*q:(i+1)*q] ** 2) for i in range(4)]
    
    # 선형 회귀로 에너지 증가율 계산
    x = np.arange(4)
    slope = np.polyfit(x, energies, 1)[0]
    mean_energy = np.mean(energies)
    
    if mean_energy > 0:
        normalized_slope = slope / mean_energy  # 상대적 증가율
    else:
        normalized_slope = 0
    
    return normalized_slope, energies

def label_from_trend(slope):
    # 게인/거리 상관없이 신호 자체 에너지 추세로 판단
    if slope < -0.05:
        return 0   # 안전 (에너지 감소 중)
    elif slope < 0.15:
        return 1   # 주의 (에너지 유지 또는 약간 증가)
    else:
        return 2   # 위험 (에너지 빠르게 증가 = 하울링 임박)

def extract_features(segment):
    fft = np.abs(np.fft.rfft(segment))
    max_val = np.max(fft)
    if max_val > 0:
        fft = fft / max_val

    half = len(segment) // 2
    e1 = np.mean(segment[:half] ** 2)
    e2 = np.mean(segment[half:] ** 2)
    energy_ratio = e2 / e1 if e1 > 0 else 1.0
    peak = np.max(np.abs(segment))
    zcr = np.mean(np.abs(np.diff(np.sign(segment)))) / 2

    return np.concatenate([fft, [energy_ratio, peak, zcr]])

# 다양한 게인 + 거리 조합으로 데이터 생성
gains = np.arange(0.3, 1.2, 0.05)
distances = [5, 8, 10, 15, 20]
SAMPLES_PER_COMBO = 10

X = []
y = []
skipped = 0

for gain in gains:
    for distance in distances:
        for _ in range(SAMPLES_PER_COMBO):
            signal = simulate_feedback(gain, distance)
            slope, _ = measure_energy_trend(signal)
            label = label_from_trend(slope)

            for start in range(0, len(signal) - SEGMENT_LENGTH, SEGMENT_LENGTH):
                segment = signal[start:start + SEGMENT_LENGTH]
                features = extract_features(segment)
                X.append(features)
                y.append(label)

X = np.array(X)
y = np.array(y)

os.makedirs('data', exist_ok=True)
np.save('data/X.npy', X)
np.save('data/y.npy', y)

print(f'데이터 생성 완료!')
print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')
print(f'레이블 분포: 안전={np.sum(y==0)}, 주의={np.sum(y==1)}, 위험={np.sum(y==2)}')
