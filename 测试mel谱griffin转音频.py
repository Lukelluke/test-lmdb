import numpy as np
import os
import audio
import lmdb
import io
import librosa
import scipy

sr = 24000 # Sample rate.
n_fft = 2048 # fft points (samples)
frame_shift = 0.0125 # seconds
frame_length = 0.05 # seconds
hop_length = int(sr*frame_shift) # samples.
win_length = int(sr*frame_length) # samples.
n_mels = 512 # Number of Mel banks to generate
power = 1.2 # Exponent for amplifying the predicted magnitude
n_iter = 100 # Number of inversion iterations
preemphasis = .97 # or None
max_db = 100
ref_db = 20
top_db = 15


def melspectrogram2wav(mel):
    '''# Generate wave file from spectrogram'''
    # transpose
    mel = mel.T

    # de-noramlize
    mel = (np.clip(mel, 0, 1) * max_db) - max_db + ref_db

    # to amplitude
    mel = np.power(10.0, mel * 0.05)
    m = _mel_to_linear_matrix(sr, n_fft, n_mels)
    mag = np.dot(m, mel)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = scipy.signal.lfilter([1], [1, -preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def _mel_to_linear_matrix(sr, n_fft, n_mels):
    m = librosa.filters.mel(sr, n_fft, n_mels)
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))

def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.
    '''
    X_best = copy.deepcopy(spectrogram)
    for i in range(n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, n_fft, hop_length, win_length=win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hop_length, win_length=win_length, window="hann")




#****************************

wav_path = './p225/1.wav'
wav = audio.load_wav(wav_path)
melspectrogram = audio.melspectrogram(wav).astype(np.float32)  # # (80, 448)
n_frames = melspectrogram.shape[1]

print("melspectrogram.shape = ", melspectrogram.shape)

print("n_frames = ", n_frames)

mag = audio._mel_to_linear(melspectrogram)

print("mag.shape = ", mag.shape)  # mag.shape =  (1025, 448)

orisp = audio.spectrogram(wav)
print("orisp.shape = ", orisp.shape)  # orisp.shape =  (1025, 448)

# wav = audio.griffin_lim(orisp)
wav = audio._griffin_lim(orisp)
audio.save_wav(wav, './ori-sp-to-wav.wav')

#
# wav = melspectrogram2wav(melspectrogram)
#
# audio.save_wav(wav, './hello-taco.wav')

'''
melspectrogram.shape =  (80, 448)
n_frames =  448
mag.shape =  (1025, 448)
orisp.shape =  (1025, 448)
'''

