---
tags:
    - condes
    - meachine learning


---



```python
import numpy as np
import torch
from torch.utils import data
import d2l.torch as d2l
```


```python
####生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```


```python
###我们将features和labels作为API的参数传递，并通过数据迭代器指定batch_size。 此外，布尔值is_train表示是否希望数据迭代器对象在每个迭代周期内打乱数据
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))
```




    [tensor([[ 6.3585e-02,  1.6857e+00],
             [-7.0598e-04, -9.5375e-01],
             [-1.2699e+00,  1.1126e+00],
             [ 5.6441e-01, -5.7444e-02],
             [ 6.9521e-01,  1.1611e+00],
             [-1.2566e+00,  2.1381e-01],
             [ 5.7550e-02,  5.3928e-01],
             [-1.0612e-01,  1.0140e-01],
             [-9.2195e-01,  9.9895e-02],
             [-3.7519e-01, -1.1524e+00]]),
     tensor([[-1.3928],
             [ 7.4439],
             [-2.1106],
             [ 5.5108],
             [ 1.6452],
             [ 0.9392],
             [ 2.4839],
             [ 3.6438],
             [ 2.0152],
             [ 7.3660]])]




```python
##使用框架的预定义好的层
##Sequential类将多个层串联在一起。 当给定输入数据时，Sequential实例将数据传入到第一层， 然后将第一层的输出作为第二层的输入

# nn是神经网络的缩写
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))
```


```python
###初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```




    tensor([0.])




```python
###定义损失函数
###均方误差使用MSELoss类，也称为L2平方范数
loss = nn.MSELoss()
```


```python
###小批量随机梯度下降算法是一种优化神经网络的标准工具
###当我们实例化一个SGD实例时，我们要指定优化的参数 （可通过net.parameters()从我们的模型中获得）以及优化算法所需的超参数字典。 小批量随机梯度下降只需要设置lr值，这里设置为0.03
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```


```python
####训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

    epoch 1, loss 0.000226
    epoch 2, loss 0.000098
    epoch 3, loss 0.000098



```python
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
```

    w的估计误差： tensor([ 1.7643e-05, -3.8862e-05])
    b的估计误差： tensor([0.0004])



```python

```
