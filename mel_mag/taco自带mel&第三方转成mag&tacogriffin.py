import numpy as np

import mel_mag.audio2 as audio2
from mel_mag.hparams2 import hparams

print(hparams.num_mels)  # 80
print(hparams.sample_rate)  # 22050
wav_path = './1.wav'

wav = audio2.load_wav(wav_path, hparams.sample_rate)
melspectrogram = audio2.melspectrogram(wav, hparams).astype(np.float32)  # # (80, 448)
n_frames = melspectrogram.shape[1]

print(melspectrogram.shape)  # (80,449)

wav = audio2.inv_mel_spectrogram(melspectrogram, hparams)

audio2.save_wav(wav, './Rayhane-mamah版本mel转语音.wav', hparams.sample_rate)