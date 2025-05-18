import time
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt

# 1. 오디오 파일 로드 (torchaudio는 waveform, sample_rate 반환)
waveform, sr = torchaudio.load("./input.wav")  # (channel, time)

# 2. 파라미터 설정
n_fft = 512
hop_length = 256
n_mels = 80
f_min = 80
f_max = 8000

# 3. MelSpectrogram 변환기 정의
mel_transform = T.MelSpectrogram(
    sample_rate=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels,
    f_min=f_min,
    f_max=f_max,
    power=2.0  # 파워 스펙트럼 (전력 기반)
)
# 3-1. MelSpectrogram 값을 db로 전환
db_transform = T.AmplitudeToDB(stype="power", top_db=80)

# 4. 생성 처리 속도 측정
start_time = time.time()
mel_spec = mel_transform(waveform)  # (channel, n_mels, time)
mel_spec_db = db_transform(mel_spec)
end_time = time.time()
print(f"processing time: {end_time - start_time}")

# 5. 시각화 및 저장 (첫 번째 채널만 사용)
plt.figure(figsize=(10, 4))
plt.imshow(mel_spec_db[0].numpy(), origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel-Spectrogram (torchaudio)")
plt.xlabel("Time")
plt.ylabel("Mel Frequency Bin")
plt.tight_layout()
fname = f"mel_spectrogram_torchaudio_{n_fft}nfft_{f_max}fmax.png"
plt.savefig(fname, dpi=300)
plt.close()

print(f"Mel-spectrogram saved as {fname}")
