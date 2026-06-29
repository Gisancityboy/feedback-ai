import numpy as np
import os

SAMPLE_RATE = 44100
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

def simulate_feedback(gain, distance, duration=3.0):
    n_samples = int(SAMPLE_RATE * duration)
    delay_samples = int((distance / 343.0) * SAMPLE_RATE)
    output = np.zeros(n_samples)
    t = np.linspace(0, duration, n_samples)

    n_freqs = np.random.randint(3, 10)
    freqs = np.random.choice([80,120,180,250,350,500,700,1000,1500,2000,3000,4000,6000,8000], n_freqs, replace=False)
    amplitudes = np.random.uniform(0.1, 1.0, n_freqs)
    input_signal = sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(freqs, amplitudes))

    envelope = np.random.uniform(0.2, 1.0, 20)
    envelope = np.interp(np.linspace(0, 19, n_samples), np.arange(20), envelope)
    input_signal *= envelope
    input_signal += pink_noise(n_samples) * np.random.uniform(0.05, 0.3)
    input_signal *= 0.01
    input_signal = multi_reflection_filter(input_signal)

    for i in range(n_samples):
        feedback = output[i - delay_samples] * gain if i >= delay_samples else 0
        output[i] = np.clip(input_signal[i] + feedback, -10, 10)

    return output

def measure_energy_trend(signal):
    n = len(signal)
    n_bins = 6
    q = n // n_bins
    energies = [np.mean(signal[i*q:(i+1)*q] ** 2) for i in range(n_bins)]
    slope = np.polyfit(np.arange(n_bins), energies, 1)[0]
    mean_energy = np.mean(energies)
    return slope / mean_energy if mean_energy > 0 else 0

def label_from_trend(slope):
    if slope < -0.02:
        return 0   # 안전
    elif slope < 0.10:
        return 1   # 주의
    else:
        return 2   # 위험

LPC_ORDER = 20

def levinson_durbin(rxx, order):
    # GLD.c의 Levinson-Durbin 재귀를 Python으로 구현
    # rxx: 자기상관 배열 (lag 0부터 order까지)
    # return: LPC 계수 배열 (order개)
    a = np.zeros(order)
    E = rxx[0]
    for m in range(order):
        if E < 1e-10:
            break
        k = -rxx[m + 1]
        for i in range(m):
            k -= a[i] * rxx[m - i]
        k /= E
        a_new = a.copy()
        a_new[m] = k
        for i in range(m):
            a_new[i] = a[i] + k * a[m - 1 - i]
        a = a_new
        E *= (1 - k * k)
    return a

def extract_features(segment):
    # 1024 샘플 세그먼트 → 537개 피처
    # 1. FFT 스펙트럼 (513개)
    fft = np.abs(np.fft.rfft(segment))
    max_val = np.max(fft)
    if max_val > 0:
        fft = fft / max_val

    # 2. 에너지 관련 (3개)
    half = len(segment) // 2
    e1 = np.mean(segment[:half] ** 2)
    e2 = np.mean(segment[half:] ** 2)
    energy_ratio = e2 / e1 if e1 > 0 else 1.0
    peak = np.max(np.abs(segment))
    zcr = np.mean(np.abs(np.diff(np.sign(segment)))) / 2

    # 3. 자기상관 (GLD의 rxx 개념) — 첫 20 lag 정규화 (20개)
    autocorr = np.correlate(segment, segment, mode='full')
    mid = len(autocorr) // 2
    autocorr_lags = autocorr[mid:mid + LPC_ORDER + 1]
    if autocorr_lags[0] > 0:
        autocorr_lags = autocorr_lags / autocorr_lags[0]

    # 4. LPC 계수 (Levinson-Durbin, GLD 알고리즘 핵심) — 20개
    lpc = levinson_durbin(autocorr_lags, LPC_ORDER)

    # 5. 스펙트럼 평탄도 (1개) — 피드백시 특정 주파수 급격히 부각
    fft_raw = np.abs(np.fft.rfft(segment)) + 1e-10
    spectral_flatness = np.exp(np.mean(np.log(fft_raw))) / np.mean(fft_raw)

    return np.concatenate([fft, [energy_ratio, peak, zcr],
                           autocorr_lags[1:],  # lag 1~20 (20개)
                           lpc,                # LPC 계수 20개
                           [spectral_flatness]])  # 총 537개

gains = np.arange(0.3, 1.2, 0.05)
distances = [5, 8, 10, 15, 20]
SAMPLES_PER_COMBO = 15

X = []
y = []

for gain in gains:
    for distance in distances:
        for _ in range(SAMPLES_PER_COMBO):
            # 3초 신호로 레이블 결정
            long_signal = simulate_feedback(gain, distance, duration=3.0)
            slope = measure_energy_trend(long_signal)
            label = label_from_trend(slope)

            # 1024 샘플 세그먼트로 특징 추출
            for start in range(0, len(long_signal) - SEGMENT_LENGTH, SEGMENT_LENGTH):
                segment = long_signal[start:start + SEGMENT_LENGTH]
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
