import numpy as np
import scipy.io.wavfile as wav
import torch
import matplotlib.pyplot as plt
from model import FeedbackClassifier

plt.rcParams['font.family'] = 'Nanum Gothic'
plt.rcParams['axes.unicode_minus'] = False

SEGMENT_LENGTH = 1024
LABELS = ['안전', '주의', '위험']
COLORS = ['green', 'orange', 'red']

LPC_ORDER = 20

def levinson_durbin(rxx, order):
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
    autocorr = np.correlate(segment, segment, mode='full')
    mid = len(autocorr) // 2
    autocorr_lags = autocorr[mid:mid + LPC_ORDER + 1]
    if autocorr_lags[0] > 0:
        autocorr_lags = autocorr_lags / autocorr_lags[0]
    lpc = levinson_durbin(autocorr_lags, LPC_ORDER)
    fft_raw = np.abs(np.fft.rfft(segment)) + 1e-10
    spectral_flatness = np.exp(np.mean(np.log(fft_raw))) / np.mean(fft_raw)
    return np.concatenate([fft, [energy_ratio, peak, zcr],
                           autocorr_lags[1:], lpc, [spectral_flatness]])

model = FeedbackClassifier(input_size=557)
model.load_state_dict(torch.load('feedback_model.pth'))
model.eval()

def measure_energy_trend(signal):
    n = len(signal)
    q = n // 4
    energies = [np.mean(signal[i*q:(i+1)*q] ** 2) for i in range(4)]
    slope = np.polyfit(np.arange(4), energies, 1)[0]
    mean_energy = np.mean(energies)
    return slope / mean_energy if mean_energy > 0 else 0

def predict_file(filepath):
    sample_rate, data = wav.read(filepath)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32) / 32767.0

    results = []
    timestamps = []
    for start in range(0, len(data) - SEGMENT_LENGTH, SEGMENT_LENGTH):
        segment = data[start:start + SEGMENT_LENGTH]
        features = extract_features(segment)
        x = torch.FloatTensor(features).unsqueeze(0)
        with torch.no_grad():
            out = model(x)
            pred = torch.argmax(out).item()
        results.append(pred)
        timestamps.append(start / sample_rate)

    trend = measure_energy_trend(data)
    results = np.array(results)
    danger_ratio = np.mean(results == 2)
    caution_ratio = np.mean(results == 1)
    # 위험이 20% 이상이면 위험, 주의가 40% 이상이면 주의
    if danger_ratio >= 0.20:
        final_label = 2
    elif caution_ratio >= 0.40:
        final_label = 1
    else:
        final_label = 0
    return np.array(timestamps), results, trend, final_label

files = [
    ('test_safe.wav', '안전 구간'),
    ('test_caution.wav', '주의 구간'),
    ('test_danger.wav', '위험 구간 (하울링)'),
]

fig, axes = plt.subplots(3, 1, figsize=(12, 10))
for idx, (filename, title) in enumerate(files):
    times, preds, trend, final_label = predict_file(filename)
    for t, p in zip(times, preds):
        axes[idx].bar(t, p + 1, width=SEGMENT_LENGTH/44100, color=COLORS[p], alpha=0.7)
    final = LABELS[final_label]
    axes[idx].set_title(f'{title} → AI 판정: {final} (에너지 추세: {trend:.3f})')
    axes[idx].set_ylabel('위험도')
    axes[idx].set_ylim(0, 3.5)
    axes[idx].set_xlabel('시간 (초)')

plt.tight_layout()
plt.savefig('predict_result.png', dpi=150)
plt.show()

print('\n=== 분석 결과 ===')
for filename, title in files:
    times, preds, trend, final_label = predict_file(filename)
    counts = [np.sum(preds==i) for i in range(3)]
    final = LABELS[final_label]
    print(f'{title}: 안전 {counts[0]}개 / 주의 {counts[1]}개 / 위험 {counts[2]}개 → 판정: {final} | 에너지 추세: {trend:.3f}')
