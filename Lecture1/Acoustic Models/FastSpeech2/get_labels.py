import pyworld
import librosa
import soundfile as sf
import numpy as np
from textgrid import TextGrid
from scipy.interpolate import interp1d

sampling_rate = 22050
hop_length = 256
frame_length = 1024
wav_path = "example.wav"
text_grid_path = "example.TextGrid"

def interpolate(signal):
    '''
    특정 구간에서 pitch가 무성음 또는 예측에 실패하여 0값인 경우에 대하여 주변 pitch 값을 이용해서 보간하는 함수.
    '''
    idx = np.arange(len(signal))
    valid = signal > 0
    if valid.sum() < 2:
        return np.zeros_like(signal)
    return interp1d(idx[valid], signal[valid], kind='linear', fill_value="extrapolate")(idx)

# 1. 음성 로딩
x, fs = sf.read(wav_path)
if fs != sampling_rate:
    x, orig_sr = librosa.load(wav_path, sr=None)
    x = librosa.resample(x, orig_sr=fs, target_sr=sampling_rate)
x = x.astype(np.float64)

# 2. TextGrid 로딩
tg = TextGrid()
tg.read(text_grid_path)

# 3. pyworld를 통해 pitch 추출
frame_period = 1000 * hop_length / sampling_rate  # 약 11.6ms (FastSpeech2 호환 hop size=256 기준)
f0, t = pyworld.dio(x.astype(np.float64), sampling_rate, frame_period=frame_period)
f0 = pyworld.stonemask(x.astype(np.float64), f0, t, sampling_rate)

# 4. energy 계산
energy = librosa.feature.rms(y=x, frame_length=frame_length, hop_length=hop_length).squeeze()

# 5. 선형 보간 
f0_interp = interpolate(f0)
energy_interp = interpolate(energy)

# 6. text grid에 phoneme-duration 정보가 잘 반영이 되어 있는지 확인
phones = None
for tier in tg.tiers:
    if "phone" in tier.name.lower():
        phones = tier
        break

# 7. duration 값에 따라 pitch와 energy 계산
results = []
for interval in phones:
    label = interval.mark.strip()
    start, end = interval.minTime, interval.maxTime

    if label == "" or end <= start:
        continue

    # 시간 → frame index 변환
    start_idx = int(np.floor(start * 1000 / frame_period))
    end_idx = int(np.ceil(end * 1000 / frame_period))

    f0_seg = f0_interp[start_idx:end_idx]
    energy_seg = energy_interp[start_idx:end_idx]

    mean_f0 = np.mean(f0_seg)
    mean_energy = np.mean(energy_seg)
    duration = end - start

    results.append((label, duration, mean_f0, mean_energy))

# 8. 결과 출력
print(f"{'Phone':6s} | {'Dur(s)':>6s} | {'F0(Hz)':>8s} | {'Energy':>8s}")
print("-" * 40)
for phone, dur, f0_val, e_val in results:
    print(f"{phone:6s} | {dur:6.3f} | {f0_val:8.2f} | {e_val:8.4f}")