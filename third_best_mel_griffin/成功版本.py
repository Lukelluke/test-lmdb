from third_best_mel_griffin.hparams import *
import third_best_mel_griffin.audio3 as audio
import lmdb
import numpy as np
import matplotlib.pyplot as plt
import os

'''
这份可以完成 mel的提取 && mel转连续存储，并保存到lmdb && 从lmdb中读取数据并还原尺寸 
'''
fpath = './1.wav'
mel, mag = audio.get_spectrograms(fpath)  # 第三方提取 mel 和 mag线性谱

fig = plt.figure(figsize=(20, 20))
heatmap = plt.pcolor(mag)
fig.colorbar(mappable=heatmap)
plt.xlabel('Time(s)')
plt.ylabel("note")
plt.tight_layout()
plt.savefig("mag-图像测试")

print("mag.shape = ",mag.shape)

# plt.imshow(mag)
plt.show()




print("mel.shape = ", mel.shape)  # mel.shape =  (319, 80)
print("mag.shape = ", mag.shape)  # mag.shape =  (319, 1025)

wav = audio.melspectrogram2wav(mel)
print(wav.shape)  # (87450,)

audio.save_wav(wav, './test.wav')

'''
取消 trim 处理之后：
mel.shape =  (449, 512)
mag.shape =  (449, 1025)
(123200,)
'''

# ********************************
# 如果train文件夹下没有data.mbd或lock.mdb文件，则会生成一个空的，如果有，不会覆盖
# map_size定义最大储存容量，单位是kb，以下定义1TB容量
env = lmdb.open("./lmdb", map_size=1099511627776)
env.close()

env = lmdb.open("./lmdb")
txn = env.begin(write=True)

# 添加数据和键值
txn.put(str(1).encode(), 'aaa'.encode())
txn.put(str(2).encode(), 'bbb'.encode())
txn.put(str(3).encode(), 'ccc'.encode())

# mel = mel[0:300, :]
shape = mel.shape

mel_spectrogram = np.ascontiguousarray(mel)  # 转换成 连续存储的数据，才能放进lmdb里面
txn.put(('X_' + str(4)).encode(), mel_spectrogram)

for key, value in txn.cursor():
    if ('4').encode() in key:
        print("******** value = ", type(value))  # type(value) =  <class 'bytes'>
        print("*"*50)
        tmp = np.fromstring(value, dtype=np.float32)  # 将二进制串，重新读取成 numpy 形式
        print("-------tmp.shape = ", tmp.shape)  # -------tmp.shape =  (25520,)
        tmp = np.reshape(tmp, shape)
        print(tmp)
        print(tmp.shape)
        wav = audio.melspectrogram2wav(tmp)
        audio.save_wav(wav, './hello80.wav')
    print(key, value)

