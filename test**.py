'''
教程： https://blog.csdn.net/qq_41185868/article/details/90294583


'''

import numpy as np
import os
import audio
import lmdb
import io


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
  '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.

    Args:
      in_dir: The directory where you have downloaded the LJ Speech dataset
      out_dir: The directory to write the output into
      num_workers: Optional number of worker processes to parallelize across
      tqdm: You can optionally pass tqdm to get a nice progress bar

    Returns:
      A list of tuples describing the training examples. This should be written to train.txt
  '''

  # We use ProcessPoolExecutor to parallelize across processes. This is just an optimization and you
  # can omit it and just call _process_utterance on each input if you want.
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []
  index = 1
  with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
    for line in f:
      parts = line.strip().split('|')
      wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
      text = parts[2]
      futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav_path, text)))
      index += 1
  return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text):
  '''Preprocesses a single utterance audio/text pair.

  This writes the mel and linear scale spectrograms to disk and returns a tuple to write
  to the train.txt file.

  Args:
    out_dir: The directory to write the spectrograms into
    index: The numeric index to use in the spectrogram filenames.
    wav_path: Path to the audio file containing the speech input
    text: The text spoken in the input audio file

  Returns:
    A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
  '''

  # Load the audio to a numpy array:
  wav = audio.load_wav(wav_path)

  # Compute the linear-scale spectrogram from the wav:
  spectrogram = audio.spectrogram(wav).astype(np.float32)
  n_frames = spectrogram.shape[1]

  # Compute a mel-scale spectrogram from the wav:
  mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

  # Write the spectrograms to disk:
  spectrogram_filename = 'ljspeech-spec-%05d.npy' % index
  mel_filename = 'ljspeech-mel-%05d.npy' % index
  np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

  # Return a tuple describing this training example:
  return (spectrogram_filename, mel_filename, n_frames, text)




wav_path = './p225/1.wav'
wav = audio.load_wav(wav_path)
spectrogram = audio.spectrogram(wav).astype(np.float32)
n_frames = spectrogram.shape[1]


mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

print("line 88: mel_spectrogram.shape = ", mel_spectrogram.shape)

mel = np.asarray(mel_spectrogram, dtype=np.float32)  # line 88: mel_spectrogram.shape =  (512, 448)

print("line 92: mel.shape = ", mel.shape)  # line 92: mel.shape =  (512, 448)







# import numpy as np
#
#
# arr1 = np.array([[1, 2], [3, 4]])
# print(arr1)
# print(type(arr1))
#
# print(arr1 ** 2)
# print("*****************")
# print(arr1.shape)
# shape=arr1.shape
# test = np.random.rand(*arr1.shape)
# '''
# https://ask.csdn.net/questions/718273
# 星号用于参数传递，*args会把可迭代对象解析出来用于参数传递
# 把tuple（2，2）里面的每一个值拿出来给np.random.rand当参数
# '''
# print(test)
# print("-----------------------")
# angles = np.exp(2j * np.pi * np.random.rand(*arr1.shape))
# '''
# [[ 0.59634454-0.80272859j -0.32027496+0.94732463j]
#  [-0.8415723 +0.54014449j  0.7772831 -0.629151j  ]]
# '''
# print(angles)