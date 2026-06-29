import numpy as np
import matplotlib.pyplot as plt

SAMPLE_RATE = 44100
DURATION = 3.0
ROOM_DELAY = 0.023  # 23ms (약 8m 거리)

def simulate_feedback(gain, room_delay_sec, sample_rate, duration):
    n_samples = int(sample_rate * duration)
    delay_samples = int(room_delay_sec * sample_rate)
    output = np.zeros(n_samples)
    output[0] = 0.001

    for i in range(1, n_samples):
        if i >= delay_samples:
            feedback = output[i - delay_samples] * gain
        else:
            feedback = 0
        output[i] = feedback

    return output

gains = [0.5, 0.8, 0.95, 1.0, 1.05]
fig, axes = plt.subplots(len(gains), 1, figsize=(12, 10))

for idx, g in enumerate(gains):
    signal = simulate_feedback(g, ROOM_DELAY, SAMPLE_RATE, DURATION)
    time = np.linspace(0, DURATION, len(signal))
    axes[idx].plot(time, signal, linewidth=0.5)
    axes[idx].set_title(f'루프 게인: {g}')
    axes[idx].set_ylabel('Amplitude')
    axes[idx].set_ylim(-2, 2)

axes[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig('feedback_simulation.png', dpi=150)
plt.show()