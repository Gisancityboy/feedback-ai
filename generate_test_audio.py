import numpy as np
import scipy.io.wavfile as wav

SAMPLE_RATE = 44100
DURATION = 5.0

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

def generate_audio(gain, filename):
    n_samples = int(SAMPLE_RATE * DURATION)
    distance = np.random.uniform(5, 20)
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
        output[i] = np.clip(input_signal[i] + feedback, -1.0, 1.0)

    output_int = (output * 32767).astype(np.int16)
    wav.write(filename, SAMPLE_RATE, output_int)
    print(f'저장 완료: {filename} (게인: {gain})')

generate_audio(0.4,  'test_safe.wav')
generate_audio(0.75, 'test_caution.wav')
generate_audio(1.05, 'test_danger.wav')
