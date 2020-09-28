import librosa
import librosa.filters
import math
import numpy as np
import tensorflow as tf
import scipy
from hparams import hparams
import copy


def load_wav(path):
    return librosa.core.load(path, sr=hparams.sample_rate)[0]


def save_wav(wav, path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    scipy.io.wavfile.write(path, hparams.sample_rate, wav.astype(np.int16))


def preemphasis(x):
    return scipy.signal.lfilter([1, -hparams.preemphasis], [1], x)


def inv_preemphasis(x):
    return scipy.signal.lfilter([1], [1, -hparams.preemphasis], x)


def spectrogram(y):
    D = _stft(preemphasis(y))
    S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
    return _normalize(S)


def inv_spectrogram(spectrogram):
    '''Converts spectrogram to waveform using librosa'''
    S = _db_to_amp(_denormalize(spectrogram) + hparams.ref_level_db)  # Convert back to linear
    return inv_preemphasis(_griffin_lim(S ** hparams.power))  # Reconstruct phase


def inv_spectrogram_tensorflow(spectrogram):
    '''Builds computational graph to convert spectrogram to waveform using TensorFlow.

  Unlike inv_spectrogram, this does NOT invert the preemphasis. The caller should call
  inv_preemphasis on the output after running the graph.
  '''
    S = _db_to_amp_tensorflow(_denormalize_tensorflow(spectrogram) + hparams.ref_level_db)
    return _griffin_lim_tensorflow(tf.pow(S, hparams.power))


def melspectrogram(y):
    D = _stft(preemphasis(y))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
    return _normalize(S)


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
    window_length = int(hparams.sample_rate * min_silence_sec)
    hop_length = int(window_length / 4)
    threshold = _db_to_amp(threshold_db)
    for x in range(hop_length, len(wav) - window_length, hop_length):
        if np.max(wav[x:x + window_length]) < threshold:
            return x + hop_length
    return len(wav)


def griffin_lim(spectrogram):
  '''Applies Griffin-Lim's raw.
  https://blog.csdn.net/weixin_35576881/article/details/90300799?utm_medium=distribute.wap_relevant.none-task-blog-searchFromBaidu-6.wap_blog_relevant_no_pic&depth_1-utm_source=distribute.wap_relevant.none-task-blog-searchFromBaidu-6.wap_blog_relevant_no_pic
  '''
  X_best = copy.deepcopy(spectrogram)
  for i in range(hparams.griffin_lim_iters):
    # X_t = invert_spectrogram(X_best)  # 先用taco自带的
    X_t = _istft(X_best)
    # est = librosa.stft(X_t, n_fft, hop_length, win_length=win_length)
    est = _stft(X_t)
    phase = est / np.maximum(1e-8, np.abs(est))
    X_best = spectrogram * phase
  # X_t = invert_spectrogram(X_best)
  X_t = _istft(X_best)
  y = np.real(X_t)

  return y

# 自己写的第三方函数
def invert_spectrogram(spectrogram):
  '''
  spectrogram: [f, t]
  '''
  return librosa.istft(spectrogram, hop_length, win_length=win_length, window="hann")


def _griffin_lim(S):
    '''librosa implementation of Griffin-Lim
  Based on https://github.com/librosa/librosa/issues/434
  '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y


def _griffin_lim_tensorflow(S):
    '''TensorFlow implementation of Griffin-Lim
  Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
  '''
    with tf.variable_scope('griffinlim'):
        # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
        S = tf.expand_dims(S, 0)
        S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
        y = _istft_tensorflow(S_complex)
        for i in range(hparams.griffin_lim_iters):
            est = _stft_tensorflow(y)
            angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
            y = _istft_tensorflow(S_complex * angles)
        return tf.squeeze(y, 0)


def _stft(y):
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):
    _, hop_length, win_length = _stft_parameters()
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_tensorflow(signals):
    n_fft, hop_length, win_length = _stft_parameters()
    return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)


def _istft_tensorflow(stfts):
    n_fft, hop_length, win_length = _stft_parameters()
    return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)


def _stft_parameters():
    # num_freq=1025, n_fft = 2048;
    # frame_shift_ms=12.5; sample_rate=20000
    # hop_length = 0.0125 * 20000 = 250 ms
    n_fft = (hparams.num_freq - 1) * 2  # 2048
    hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)  # 250
    win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)  # 1000
    return n_fft, hop_length, win_length


# Conversions:

_mel_basis = None


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)

# by hsj-2020/08/18 ：mel 谱转线性谱
# https://blog.csdn.net/weixin_35576881/article/details/90300799?utm_medium=distribute.wap_relevant.none-task-blog-searchFromBaidu-6.wap_blog_relevant_no_pic&depth_1-utm_source=distribute.wap_relevant.none-task-blog-searchFromBaidu-6.wap_blog_relevant_no_pic
def _mel_to_linear(melspectrogram):
  '''将 mel_basis 转置，再和原来 mel_basis 做乘法，再额外做一下处理，从而得到线性谱 系数（还不是线性谱）'''
  # transpose
  # melspectrogram = melspectrogram.T

  # de-noramlize
  '''clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min'''
  melspectrogram = _denormalize(melspectrogram)


  _mel_basis = _build_mel_basis()

  m_t = np.transpose(_mel_basis)
  p = np.matmul(_mel_basis, m_t)
  d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
  m = np.matmul(m_t, np.diag(d))
  mag = np.dot(m, melspectrogram)
  return mag

'''
https://blog.csdn.net/qq_35277038/article/details/80766746

可以发现，当 np.diag(array) 中
array是一个1维数组时，结果形成一个以一维数组为对角线元素的矩阵
array是一个二维矩阵时，结果输出矩阵的对角线元素
'''



def _build_mel_basis():
    #  num_freq = 1025, 所以 n_fft = 2048
    n_fft = (hparams.num_freq - 1) * 2
    return librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=hparams.num_mels)


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _db_to_amp_tensorflow(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)


def _normalize(S):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db


def _denormalize_tensorflow(S):
    return (tf.clip_by_value(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db
