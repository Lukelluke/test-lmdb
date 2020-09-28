import librosa
import numpy as np
import audio
import copy
import scipy
import scipy.signal as signal

sr = 20000 # Sample rate.
n_fft = 2048 # fft points (samples)
frame_shift = 0.0125 # seconds
frame_length = 0.05 # seconds
hop_length = int(sr*frame_shift) # samples.
win_length = int(sr*frame_length) # samples.
n_mels = 512 # Number of Mel banks to generate  512个 mel 滤波器 比80个 效果更好
power = 1.2 # Exponent for amplifying the predicted magnitude
n_iter = 100 # Number of inversion iterations
preemphasis = .97 # or None
max_db = 100
ref_db = 20
top_db = 15


#****************************

def get_spectrograms(fpath):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
 '''
    # Loading sound file
    global sr
    y, sr = librosa.load(fpath, sr=sr)

    # Trimming
    y, _ = librosa.effects.trim(y, top_db=top_db)

    # Preemphasis
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag

def melspectrogram2wav(mel):
    '''# Generate wave file from spectrogram'''
    # transpose
    print("mel.shape = ", mel.shape)
    mel = mel.T
    print("mel.T.shape = ",mel.shape)  # (512, 318):[特征维度，帧数]

    # de-noramlize
    mel = (np.clip(mel, 0, 1) * max_db) - max_db + ref_db  # np.clip：将所有数据处理到这个区间内

    # to amplitude
    mel = np.power(10.0, mel * 0.05)  # 将mel中所有数值*0.5，然后再作为指数，对10做指数乘法,还是原本尺寸
    print("line 82 : mel.shape = ", mel.shape)  # line 82 : mel.shape =  (512, 448)

    m = _mel_to_linear_matrix(sr, n_fft, n_mels)
    print("line 84: m.shape = ", m.shape, "mel.shape = ", mel.shape)  # line 84: m.shape =  (1025, 512) mel.shape =  (512, 448)
    mag = np.dot(m, mel)  # (1025, 448)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = scipy.signal.lfilter([1], [1, -preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

def spectrogram2wav(mag):
    '''# Generate wave file from spectrogram'''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * max_db) - max_db + ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

def _mel_to_linear_matrix(sr, n_fft, n_mels):
    m = librosa.filters.mel(sr, n_fft, n_mels)
    m_t = np.transpose(m)  # transpose()函数的作用就是调换数组的行列值的索引值，类似于求矩阵的转置
    p = np.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))

def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.
    '''
    # spectrogram = spectrogram.T  # 是转置的问题！！！这句自己加上
    X_best = copy.deepcopy(spectrogram)
    for i in range(n_iter):
        # print("这里134")
        X_t = invert_spectrogram(X_best)
        # print("这里13")
        est = librosa.stft(X_t, n_fft, hop_length, win_length=win_length)
        # print("这里138")
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    sr = 20000 # Sample rate.
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples.     = 20000 * 0.0125 = 250 ms
    win_length = int(sr*frame_length) # samples.    = 20000 * 0.05   = 1000 ms
    '''
    # return librosa.istft(spectrogram, hop_length, win_length=win_length, window="hann")
    return librosa.istft(spectrogram, hop_length, win_length=win_length)


fpath = './p225/1.wav'
# mel, mag = get_spectrograms(fpath)  # 第三方提取 mel 和 mag线性谱
'''mel 直接尺寸：
mel.shape =  (318, 512)

进入 melspectrogram2wav 之后：

line 83: m.shape =  (1025, 256) mel.shape =  (256, 318帧)
'''


# 借用tacotron的mel特征来合成试试
wav = audio.load_wav(fpath)
mel = audio.melspectrogram(wav).astype(np.float32)  # # (80, 448)

# taco 的 mel 用 _mel_to_linear 第三方函数，转成 mag，看行不行
mag = audio._mel_to_linear(mel)

print("line 176:mag = ", mag)

print("line 176:自身mel代入第三方mag转换函数：mag.shape = ", mag.shape)  # line 176:自身mel代入第三方mag转换函数：mag.shape =  (1025, 448)

# mag = audio.spectrogram(wav)
# print("taco自带的 mag.shape = ", mag.shape)  # taco自带的 mag.shape =  (1025, 448)
mel = mel.T
n_frames = mel.shape[0]
'''
进入 melspectrogram2wav 之后：

(448, 512)
mel.shape =  (448, 512)
mel.T.shape =  (512, 448)

ValueError: shapes (1025,256) and (512,448) not aligned: 256 (dim 1) != 512 (dim 0)
'''


# mel = mel[0:256,:]
print(mel.shape)

# wav = melspectrogram2wav(mel)
# wav = spectrogram2wav(mag)
# wav = audio.inv_spectrogram(mag)

# taco产生的 mel 经过第三方转换成 mag，再用第三方合成，看行不行:出错！ 这是sp没有正确转置的问题！解决了
'''
Traceback (most recent call last):
  File "/Users/huangshengjie/Desktop/测试lmdb/第三方mel-griffin.py", line 203, in <module>
    wav = spectrogram2wav(mag)
  File "/Users/huangshengjie/Desktop/测试lmdb/第三方mel-griffin.py", line 112, in spectrogram2wav
    wav = griffin_lim(mag)
  File "/Users/huangshengjie/Desktop/测试lmdb/第三方mel-griffin.py", line 134, in griffin_lim
    X_t = invert_spectrogram(X_best)
  File "/Users/huangshengjie/Desktop/测试lmdb/第三方mel-griffin.py", line 155, in invert_spectrogram
    return librosa.istft(spectrogram, hop_length, win_length=win_length)
  File "/Users/huangshengjie/opt/anaconda3/envs/py36/lib/python3.6/site-packages/librosa/core/spectrum.py", line 288, in istft
    ifft_window = util.pad_center(ifft_window, n_fft)
  File "/Users/huangshengjie/opt/anaconda3/envs/py36/lib/python3.6/site-packages/librosa/util/utils.py", line 304, in pad_center
    'at least input size ({:d})').format(size, n))
librosa.util.exceptions.ParameterError: Target size (894) must be at least input size (1000)
'''
# wav = spectrogram2wav(mag)  # 两个都是空白
wav = audio.inv_spectrogram(mag)

audio.save_wav(wav, './hello2.wav')
'''
第三方数据：

(318帧, 512维)
mel.shape =  (318, 512)
mel.T.shape =  (512, 318)

# 当 n_mels = 80：
(318, 80)
mel.shape =  (318, 80)
mel.T.shape =  (80, 318)
'''
'''
试验过程：
1. 用 taco 的 mel，输入第三方的 griffin 算法，结果成功，说明taco-mel提取没错，第三方合成没错
2. 用 taco 的 mag，输入第三方，出错

3. 用第三方的 mag，输入第三方，ok
4. 用第三方的 mag，输入taco，出错：

5. taco自带 mag & taco 自带合成：ok（n_mels = 512）

librosa.util.exceptions.ParameterError: Target size (634) must be at least input size (1000)
'''
