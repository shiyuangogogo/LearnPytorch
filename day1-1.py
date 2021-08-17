# _*_ coding: utf-8 _*_
#########
# https://pytorch.apachecn.org/docs/1.7/
# Pytorch深度学习： 60分钟快速入门
#########

# 什么是pytorch
# pytorch是基于以下两个目的打造的计算框架
# .无缝替换numpy,并且通过gpu的算力来实现神经网络的加速。
# .通过自动微积分机制，使得神经网络更容易实现。

# 张量 ： 张量如同数组和矩阵一样，是一种特殊的数据结构。张量能够在gpu上运行。


import torch
import numpy as np

# 张量初始化
# 1. 由原始数据直接生成张量
data = [[1,2], [3,4]]
x_data = torch.tensor(data)
# print(type(data), type(data[0][0]))
# print(type(x_data), x_data.dtype)

# 2. 通过numpy数组来生成张量
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
# print(type(x_np), x_np.dtype)

# 3. 通过已有的张量来生成新的张量
x_ones = torch.ones_like(x_data)
# print(f"One Tensor: \n {x_ones} {x_ones.dtype} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float) # 重写x_data 的数据类型 int->float
# print(f"Random Tensor: \n {x_rand} {x_rand.dtype} \n")

# 4. 通过指定数据维度来生成张量
# shape是元组类型， 用来描述张量的维数
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor} \n")


# 张量属性
# 从张量属性我们可以得到张量的维数、数据类型以及它们所存储的设备（CPU或GPU）
tensor = torch.rand(3,4)
# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")



# 判断当前环境gpu是否可用，然后将tensor导入gpu内运行
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    # print(tensor.device)
    # print('put into gpu.')

# 张量运算
# 1. 张量的索引和切片
tensor = torch.ones(4,4)
tensor[:,1] = 0
# print(tensor)

# 2. 张量的拼接
t1 = torch.cat([tensor, tensor, tensor], dim=1)
# print(t1)

# 3. 张量的乘积和矩阵乘法
# 逐个元素相乘结果
# print(f"tensor.mul(tensor): \n {tensor.mul(tensor)} \n")
# 等价写法：
# print(f"tensor * tensor: \n {tensor * tensor}")

# 张量与张量的矩阵乘法
# print(f"tensor.matmul(tensor.T): \n {tensor.matmul(tensor.T)} \n")
# 等价写法
# print(f"tensor @ tensor.T: \n {tensor @ tensor.T}")


# 4. 自动赋值运算
# 自动赋值运算通常在方法后有 _ 作为后缀， 例如： x.copy_(y), x.t_()操作会改变x的取值。
# print(tensor, "\n")
tensor.add_(5)
# print(tensor)


# tensor与numpy的转化
# 张量和numpy array数组在CPU上可以共用一块内存区域，改变其中一个另一个也会随之改变。
# 1. 由张量变换为numpy array数组
t = torch.ones(5)
# print(f"t: {t}")
n = t.numpy()
# print(f"n: {n}")

# 2. 由numpy array数组转为张量
n = np.ones(5)
t = torch.from_numpy(n)
# 修改numpy array数组的值，则张量也会随之改变。
np.add(n, 1, out=n)
# print(f"t: {t}")
# print(f"n: {n}")














