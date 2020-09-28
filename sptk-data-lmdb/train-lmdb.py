'''
train-data lmdb
i =  800 count =  686

'''

from third_best_mel_griffin.hparams import *
import third_best_mel_griffin.audio3 as audio
import lmdb
import matplotlib.pyplot as plt
import argparse
import lmdb
import os
import torchvision.datasets as dset
import torch
import numpy as np

path2 = './train-data'
dirs = os.listdir(path2)
print("type(dirs) = ", type(dirs))  # type(dirs) =  <class 'list'>
print("len(dirs) = ", len(dirs))  # len(dirs) =  800
filelen = len(dirs)  # 800
print(dirs[0])

# path2wav = './train-data/p225_001.wav'
# mel, mag = audio.get_spectrograms(path2wav)  # 第三方提取 mel 和 mag线性谱
#
# mel = mel[0:80,:]
# print(mel.shape)
# mel = np.reshape(mel, [1, 80, 80])
# print(mel.shape)  # (1, 80, 80)
# print(type(mel))  # <class 'numpy.ndarray'>
# print(dirs[0])  # p226_298.wav
#
# print(type(dirs[0]))  # <class 'str'>
#
# print("*"*50)
# count = 1
# key = ((str(count)+'_'+dirs[0]).strip().replace('.wav', ''))
# print(key)
# print(((str(count)+'_'+dirs[0]).strip().replace('.wav', '')).encode())  # b'1_p226_298'
#
# tmp = key.encode()
# print("key.encode() = ", tmp)
# tmp = tmp.decode()  # string 类型直接 decode() 就可以，得到的还是 str 类型
# print("key.decode() = ", tmp)
# print("type(tmp) = ", type(tmp))  # type(tmp) =  <class 'str'>
# print(tmp.split('_'))  # ['1', 'p226', '298']
# print(type(tmp.split('_')))  # <class 'list'>


def main(split, wav_path, lmdb_path):
    assert split in {"train", "valid", "test"}
    # create target directory
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path, exist_ok=True)

    lmdb_split = {'train': 'train', 'valid': 'validation', 'test': 'test'}[split]
    lmdb_path = os.path.join(lmdb_path, '%s.lmdb' % lmdb_split)

    print("hello line28")

    # create lmdb
    # 如果train文件夹下没有data.mbd或lock.mdb文件，则会生成一个空的，如果有，不会覆盖
    # map_size定义最大储存容量，单位是kb，以下定义1TB容量
    env = lmdb.open(lmdb_path, map_size=1e12)  # 1*10^12

    with env.begin(write=True) as txn:
        count = 0
        for i in range(filelen):
            file_path = os.path.join(path2, dirs[i])  # 具体某个音频的路径
            mel, mag = audio.get_spectrograms(file_path)  # 第三方提取 mel 和 mag线性谱

            print("line 44: mel.shape = ", mel.shape)
            if mel.shape[0] < 80:
                continue
            else:
                mel = mel[0:80,:]  # 取前 80 帧（静音处理过的再取 80）
                np.reshape(mel, [1, 80, 80])
                count = count + 1  # 用来给文件开头名称 计数

            mel_spectrogram = np.ascontiguousarray(mel)  # 转换成连续值

            key = count-1
            # key_clean = ((str(count)+'_'+dirs[0]).strip().replace('.wav', ''))
            key_clean = (str(key))
            txn.put(key_clean.encode(), mel_spectrogram)  # key.encode() =  b'1_p226_298'


            # attr = data.attr[i, :]
            # with open(file_path, 'rb') as f:
            #     # rb	以二进制格式打开一个文件用于只读。文件指针将会放在文件的开头。这是默认模式。
            #     file_data = f.read()

            # txn.put(str(i).encode(), file_data)  # 得到的 b'数字'，b”“前缀代表的就是bytes

            # Python encode() 方法以 encoding 指定的编码格式编码字符串。errors参数可以指定不同的错误处理方案。
            # https://www.runoob.com/python/att-string-encode.html
            print("i = ", i+1, "count = ", count)  # i =  800 count =  686


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SPTK80 LMDB creator.')
    # experimental results
    parser.add_argument('--wav_path', type=str, default='./train-data',
                        help='location of wav for SPTK dataset')
    parser.add_argument('--lmdb_path', type=str, default='./LMDB',
                        help='target location for storing lmdb files')
    parser.add_argument('--split', type=str, default='train',
                        help='training or validation split', choices=["train", "valid", "test"])
    args = parser.parse_args()
    main(args.split, args.wav_path, args.lmdb_path)


#
#
# '''
# 这份可以完成 mel的提取 && mel转连续存储，并保存到lmdb && 从lmdb中读取数据并还原尺寸
# '''
# fpath = './1.wav'
# mel, mag = audio.get_spectrograms(fpath)  # 第三方提取 mel 和 mag线性谱
#
# fig = plt.figure(figsize=(20, 20))
# heatmap = plt.pcolor(mag)
# fig.colorbar(mappable=heatmap)
# plt.xlabel('Time(s)')
# plt.ylabel("note")
# plt.tight_layout()
# plt.savefig("mag-图像测试")
#
# print("mag.shape = ", mag.shape)
#
# # plt.imshow(mag)
# plt.show()
#
# print("mel.shape = ", mel.shape)  # mel.shape =  (319, 512)
# print("mag.shape = ", mag.shape)  # mag.shape =  (319, 1025)
#
# wav = audio.melspectrogram2wav(mel)
# print(wav.shape)  # (87450,)
#
# audio.save_wav(wav, './test.wav')
#
# '''
# 取消 trim 处理之后：
# mel.shape =  (449, 512)
# mag.shape =  (449, 1025)
# (123200,)
# '''
#
# # ********************************
# # 如果train文件夹下没有data.mbd或lock.mdb文件，则会生成一个空的，如果有，不会覆盖
# # map_size定义最大储存容量，单位是kb，以下定义1TB容量
# env = lmdb.open("./lmdb", map_size=1099511627776)
# env.close()
#
# env = lmdb.open("./lmdb")
# txn = env.begin(write=True)
#
# # 添加数据和键值
# txn.put(str(1).encode(), 'aaa'.encode())
# txn.put(str(2).encode(), 'bbb'.encode())
# txn.put(str(3).encode(), 'ccc'.encode())
#
# # mel = mel[0:300, :]
# shape = mel.shape
# mel_spectrogram = np.ascontiguousarray(mel)  # 转换成 连续存储的数据，才能放进lmdb里面
# txn.put(('X_' + str(4)).encode(), mel_spectrogram)
#
# for key, value in txn.cursor():
#     if ('4').encode() in key:
#         tmp = np.fromstring(value, dtype=np.float32)  # 将二进制串，重新读取成 numpy 形式
#         tmp = np.reshape(tmp, shape)
#         print(tmp)
#         print(tmp.shape)
#         wav = audio.melspectrogram2wav(tmp)
#         audio.save_wav(wav, './hello80.wav')
#     print(key, value)
