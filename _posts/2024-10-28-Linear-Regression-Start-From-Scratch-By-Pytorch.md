```python
%matplotlib inline
import random
import torch
import d2l.torch as d2l
```


```python
####用带噪声的线性模型构造一个人造数据集

def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:', features[0],'\nlabel:', labels[0])
```

    features: tensor([3.3016, 0.9787]) 
    label: tensor([7.4830])



```python
d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1);
```


![png1](/img/2024-10-28-1.png)
    



```python
###训练模型时要对数据集进行遍历，每次抽取一小批量样本，并使用它们来更新我们的模型
###定义一个data_iter函数， 该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量。 每个小批量包含一组特征和标签。
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```


```python
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```

    tensor([[ 1.0048,  1.4365],
            [-0.5837,  0.2239],
            [-0.0485, -1.3440],
            [ 0.2103,  0.9754],
            [ 2.0756,  1.2518],
            [-0.6629,  0.0966],
            [-1.0292,  0.2521],
            [-0.2634, -0.2322],
            [-0.3908,  0.5081],
            [-1.1442, -1.4172]]) 
     tensor([[1.3245],
            [2.2626],
            [8.6837],
            [1.2934],
            [4.0973],
            [2.5568],
            [1.2907],
            [4.4695],
            [1.6854],
            [6.7266]])



```python
###在我们开始用小批量随机梯度下降优化我们的模型参数之前，从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重， 并将偏置初始化为0
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```


```python
###定义模型，其实用了广播机制用一个向量加一个标量，标量会被加到向量的每一个分量上
def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b
```


```python
###3定义损失函数
def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```


```python
###定义优化算法
###从抽取的小批量中根据参数计算损失的梯度，然后随着减少损失的方向更新参数
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```


```python
####训练
####逻辑 在每次迭代中，抽取一小部分训练样本通过模型获得一组预测
###计算完损失后，反向传播
###存储每个参数的梯度，通过优化算法sgd来更新模型参数
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

    epoch 1, loss 0.038479
    epoch 2, loss 0.000137
    epoch 3, loss 0.000048



```python
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
```

    w的估计误差: tensor([ 2.2650e-05, -1.1389e-03], grad_fn=<SubBackward0>)
    b的估计误差: tensor([1.9073e-06], grad_fn=<RsubBackward1>)



```python

```
