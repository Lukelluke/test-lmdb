import lmdb
import numpy as np

env = lmdb.open("./train.lmdb")
# txn = env.begin(write=True)

# with self.data_lmdb.begin(write=False, buffers=True) as txn:
with env.begin(write=False, buffers=True) as txn:
    data = txn.get(str(0).encode())
    mel = np.asarray(data, dtype=np.float32)
    # mel = np.fromfile(data, dtype=np.float32)  # AttributeError: 'memoryview' object has no attribute 'flush'

    print("mel = ", mel)  # mel =  [218.   8. 141. ... 204.  43.  50.]
    print("type(mel[0]) = ", type(mel[0]))  # type(mel[0]) =  <class 'numpy.float32'>
    print("line 86 ****** mel.shape = ", mel.shape)  # line 86 ****** mel.shape =  (25600,)
    # mel = np.reshape(mel, self.shape)  # numpy 一维数据转为三维数据【channel=1, frame, dim】


#
# with env.begin(write=False) as txn:
#     data = txn.get(str(0).encode())
#     mel = np.fromstring(data, dtype=np.float32)  # byte 转为 numpy
#     print("mel = ", mel)  # mel =  [2.7545816e-01 2.6855630e-01 3.3646148e-01 ... 1.1347583e-01 7.4190088e-029.9999999e-09]
#     print("type(mel[0]) = ", type(mel[0]))  # type(mel[0]) =  <class 'numpy.float32'>
#     print("line 86 ****** mel.shape = ", mel.shape)  # line 86 ****** mel.shape =  (25600,)
#     # mel = np.reshape(mel, self.shape)  # numpy 一维数据转为三维数据【channel=1, frame, dim】