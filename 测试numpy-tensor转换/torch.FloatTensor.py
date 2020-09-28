import numpy as np
import torch

a = np.ones(5)
print("a = ", a)
print("type(a) = ", type(a))  # type(a) =  <class 'numpy.ndarray'>

k = torch.from_numpy(a)
print("k = ", k)  # k =  tensor([1., 1., 1., 1., 1.], dtype=torch.float64)

b = torch.from_numpy(a)
print("b = ", b)  # tensor([1., 1., 1., 1., 1.], dtype=torch.float64)
print(b[0])

c = torch.FloatTensor(a)
print("c = ", c)  # c =  tensor([1., 1., 1., 1., 1.])
print(c[0])
print(c[0].item())
print("*"*50)

d = torch.LongTensor(a)
print("d = ", d)  # d =  tensor([1, 1, 1, 1, 1])
print(type(d[0]))

e = torch.IntTensor(a)
print("e = ", e)  # e =  tensor([1, 1, 1, 1, 1], dtype=torch.int32)
