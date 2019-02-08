

# torch.optim

[`torch.optim`](#module-torch.optim "torch.optim") 是一个实现了目前大多数常用优化器的大礼包。并且由于提供的接口有很强的适用性，套用已有的接口，可以方便实现更加复杂的优化器。

## 如何使用优化器？

想要使用优化器 [`torch.optim`](#module-torch.optim "torch.optim")，首先要实例化一个优化器对象。它能保存当前张量的值并且根据梯度更新该张量。

### 构建优化器

构建优化器 [`Optimizer`](#torch.optim.Optimizer "torch.optim.Optimizer") 之前，需要指定被优化的张量 `Variable`。同时你可以指定一些参数，例如学习速率、权重衰减等。

注意

如果你想通过方法 `.cuda()` 将一个模型的张量存储在GPU，那么该操作必须在优化器被实例化之前。因为在调用方法`.cuda()`之后，该模型的所有张量都将被更新且旧的张量将与优化器无关。

一般来说，在优化器被构建和使用的过程中，需要保证被优化张量在内存中或显存中的地址是不变的。

例子：

```py
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr = 0.0001)

```

### 为多组张量指定不同的参数

当传入参数为一组字典[`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.7)")时，表示优化器[`Optimizer`](#torch.optim.Optimizer "torch.optim.Optimizer") 为多组张量指定不同的优化参数。每个字典中必须包含键`params`来指定被优化的张量，其他参数表示`params`指定的张量在优化时遵循的参数。

注意

在给优化器传入参数时，如果想给多组张量指定相同的参数，可以直接将该参数写在张量参数dic数组的外面。如果dic又指定了该参数，则该参数会被复写。例如，如果想给多组被优化的张量指定相同的学习速率时：
```py
optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)

```

其中`model.base` 的学习速率为外侧的`1e-2`, 而`model.classifier`中的学习速率为`1e-3`，且动量为0.9。

### 执行一次优化操作

所有的优化器都实现了一个张量更新方法[`step()`](#torch.optim.Optimizer.step "torch.optim.Optimizer.step")。该方法有两种使用方式：

#### `optimizer.step()`

这是一种被大多数优化器支持的简便使用方法。在梯度被计算`backward()`完成之后，该方法就可以被调用并且完成一次张量更新。

例子：

```py
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

```

#### `optimizer.step(closure)`

有些优化器例如共轭梯度优化器和LBFGS优化器等在计算的过程中需要多次更新优化器的状态，也就是说我们无法在某一时刻指定优化器何时进行张量更新。所以你需要传入一个函数，让优化器在合适的时候自动去调用该函数来完成张量更新。传入的函数中至少有包含两个操作：梯度清空、计算误差并返回。

例子：

```py
for input, target in dataset:
    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        return loss
    optimizer.step(closure)

```

## 算法

```py
class torch.optim.Optimizer(params, defaults)
```

该类是所有优化器的基类。

警告

优化器中传入的变量集合必须是稳定的（每次遍历该集合时，输出的元素顺序必须一致）。而一些常见的集合类型例如set和dic类型就不符合要求。

| 参数： | 

*   **params** (_iterable_) – 一个张量 [`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor") 的集合或者一个字典 [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.7)") 的集合，这个集合指定了需要被优化器优化的变量。
*   **defaults** – (dict): 一个指定优化参数的字典。


 |
| --- | --- |

```py
add_param_group(param_group)
```

将一组张量手动的添加到优化器[`Optimizer`](#torch.optim.Optimizer "torch.optim.Optimizer")的 `param_groups`中。

由于在优化器实例化时，需要指定哪些张量被优化。而在优化器被实例化后，可以通过调用方法`add_param_group`将一组张量添加到优化器中。该方法在对已有模型微调时很有用。

| 参数: | 

*   **param_group** ([_dict_](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.7)")) – 指定需要被添加到优化器中的张量
*   **optimization options.** (_specific_) – 被添加张量在优化时的参数（学习速率等）

 |
| --- | --- |

```py
load_state_dict(state_dict)
```

获取一个优化器当前的状态。

| Parameters: | **state_dict** ([_dict_](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.7)")) – 当前优化器的状态，其返回值与调用 [`state_dict()`](#torch.optim.Optimizer.state_dict "torch.optim.Optimizer.state_dict")一样。|
| --- | --- |

```py
state_dict()
```
返回当前优化器的状态，该返回值为字典类型[`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.7)")。
它包含两个部分：

*   ```py
    state - 包含当前优化器状态的字典
    ```

    differs between optimizer classes.
*   param_groups - 一个包含所有参数集合的字典

```py
step(closure)
```
执行一次优化过程（将待优化的张量更新一次）

| 参数: | **closure** (_callable_) – 一个可以更新模型中张量大小并且返回误差的回调函数。 |
| --- | --- |

```py
zero_grad()
```

清空所有张量[`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor")的梯度。

```py
class torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
```
实现 [ADADELTA: An Adaptive Learning Rate Method](https://arxiv.org/abs/1212.5701)优化算法。

| 参数： | 

*   **params** (_iterable_) – 一个可遍历的张量集合或者一个包含张量和优化参数的字典集合
*   **rho** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – 用于计算平方梯度运行平均值的系数（默认为0.9）
*   **eps** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – 在除法计算时为了防止数字过小添加到分母上的平滑项（默认为1e-6）
*   **lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – 学习速率（默认为1.0）
*   **weight_decay** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – 权重惩罚参数（L2正则参数）（默认为0）

 |
| --- | --- |

```py
step(closure=None)
```

执行一次参数更新操作。

| 参数: | **closure** (_callable__,_ _optional_) – 一个可以更新模型中张量大小并且返回误差的回调函数。 |
| --- | --- |

```py
class torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
```
实例化一个[Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://jmlr.org/papers/v12/duchi11a.html)优化器。

| Parameters: | 

*   **params** (_iterable_) – 一个可遍历的张量集合或者一个包含张量和优化参数的字典集合
*   **lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – 学习速率（默认为1e-2）
*   **lr_decay** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – 学习速率惩罚项（默认为0）
*   **weight_decay** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – L2正则项（默认为0）

 |
| --- | --- |

```py
step(closure=None)
```

执行一次参数更新操作。

| 参数: | **closure** (_callable__,_ _optional_) – 一个可以更新模型中张量大小并且返回误差的回调函数。 |
| --- | --- |

```py
class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```

实例化一个 [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)优化器。


| 参数: | 

*   **params** (_iterable_) – 一个可遍历的张量集合或者一个包含张量和优化参数的字典集合
*   **lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – 学习速率（默认为1e-3）
*   **betas** (_Tuple__[_[_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_]__,_ _optional_) – coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
*   **eps** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – term added to the denominator to improve numerical stability (default: 1e-8)
*   **weight_decay** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – L2正则项（默认为0）
*   **amsgrad** (_boolean__,_ _optional_) – 是否使用该算法的变体AMSGrad[On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)完成优化（默认不使用）

 |
| --- | --- |

```py
step(closure=None)
```

执行一次参数更新操作。

| 参数: | **closure** (_callable__,_ _optional_) – 一个可以更新模型中张量大小并且返回误差的回调函数。 |
| --- | --- |

```py
class torch.optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
```
实例化一个适用于稀疏张量的Adam优化器。在该算法中，只有权重需要被更新时，只有权重产生变化的那部分张量会被更新到模型参数中。

| 参数: | 

*   **params** (_iterable_) – 一个可遍历的张量集合或者一个包含张量和优化参数的字典集合
*   **lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – 学习速率（默认为1e-3）
*   **betas** (_Tuple__[_[_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_]__,_ _optional_) – coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
*   **eps** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – term added to the denominator to improve numerical stability (default: 1e-8)

 |
| --- | --- |

```py
step(closure=None)
```

执行一次参数更新操作。

| Parameters: | **closure** (_callable__,_ _optional_) – 一个可以更新模型中张量大小并且返回误差的回调函数。 |
| --- | --- |

```py
class torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
```
实例化一个 [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)优化器。

| Parameters: | 

*   **params** (_iterable_) – 一个可遍历的张量集合或者一个包含张量和优化参数的字典集合
*   **lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – 学习速率（默认为2e-3）
*   **betas** (_Tuple__[_[_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_]__,_ _optional_) – coefficients used for computing running averages of gradient and its square
*   **eps** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – term added to the denominator to improve numerical stability (default: 1e-8)
*   **weight_decay** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – L2正则项（默认为0）

 |
| --- | --- |

```py
step(closure=None)
```

执行一次参数更新操作。

| 参数: | **closure** (_callable__,_ _optional_) 一个可以更新模型中张量大小并且返回误差的回调函数。 |
| --- | --- |

```py
class torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
```
实例化一个[Acceleration of stochastic approximation by averaging](http://dl.acm.org/citation.cfm?id=131098)优化器
 
| Parameters: | 

*   **params** (_iterable_) – 一个可遍历的张量集合或者一个包含张量和优化参数的字典集合
*   **lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – 学习速率（默认为1e-2）
*   **lambd** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – decay term (default: 1e-4)
*   **alpha** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – power for eta update (default: 0.75)
*   **t0** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – point at which to start averaging (default: 1e6)
*   **weight_decay** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – L2正则项（默认为0）

 |
| --- | --- |

```py
step(closure=None)
```

执行一次张量更新操作。

| 参数: | **closure** (_callable__,_ _optional_) – 一个可以更新模型中张量大小并且返回误差的回调函数。 |
| --- | --- |

```py
class torch.optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
```
实例化一个L-BFGS优化器
注意
该优化器现在不支持dic类型的多组参数优化（也就是说只能传入一组被优化的参数）
注意
在现在的版本中，所有的参数必须存储在一个设备中（要不都在GPU，要不都在CPU）。
注意
这种算法需要的内存空间较大（需要额外占用`param_bytes * (history_size + 1)` bytes内存）。如果您的内存大小无法满足要求，请调节参数`history_size`的大小或考虑使用其他算法。

| Parameters: | 

*   **lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – 学习速率（默认为1）
*   **max_iter** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 每次权重更新时最大的迭代次数（默认为20）
*   **max_eval** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – maximal number of function evaluations per optimization step (default: max_iter * 1.25).
*   **tolerance_grad** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – termination tolerance on first order optimality (default: 1e-5).
*   **tolerance_change** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – termination tolerance on function value/parameter changes (default: 1e-9).
*   **history_size** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 历史记录大小（默认为100）

 |
| --- | --- |

```py
step(closure)
```

执行一次张量更新操作。

| Parameters: | **closure** (_callable_) – 一个可以更新模型中张量大小并且返回误差的回调函数。|
| --- | --- |

```py
class torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
```
实例化一个[Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850v5.pdf)优化器。

| 参数: | 

*   **params** (_iterable_) – 一个可遍历的张量集合或者一个包含张量和优化参数的字典集合
*   **lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – 学习速率（默认为1e-2）
*   **momentum** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – 动量（默认为0）
*   **alpha** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – 平滑因子（默认为0.99）
*   **eps** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – 添加到分母上提高稳定性的常数（默认为1e-8）
*   **centered** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – if `True`, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
*   **weight_decay** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – L2正则项（默认为0）

 |
| --- | --- |

```py
step(closure=None)
```

执行一次张量更新操作。

| 参数: | **closure** (_callable__,_ _optional_) – 一个可以更新模型中张量大小并且返回误差的回调函数。 |
| --- | --- |

```py
class torch.optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
```
实现一个弹性传播优化器

| 参数: | 

*   **params** (_iterable_) – 一个可遍历的张量集合或者一个包含张量和优化参数的字典集合
*   **lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – 学习速率（默认为1e-2）
*   **etas** (_Tuple__[_[_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_]__,_ _optional_) – pair of (etaminus, etaplis), that are multiplicative increase and decrease factors (default: (0.5, 1.2))
*   **step_sizes** (_Tuple__[_[_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_]__,_ _optional_) – a pair of minimal and maximal allowed step sizes (default: (1e-6, 50))

 |
| --- | --- |

```py
step(closure=None)
```

执行一次张量更新操作。

| 参数: | **closure** (_callable__,_ _optional_) – 一个可以更新模型中张量大小并且返回误差的回调函数。 |
| --- | --- |

```py
class torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
```

实例化一个随机梯度下降[On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf)优化器


| 参数: | 

*   **params** (_iterable_) – 一个可遍历的张量集合或者一个包含张量和优化参数的字典集合
*   **lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – 学习速率
*   **momentum** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – 动量（默认为0）
*   **weight_decay** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – L2正则项（默认为0）
*   **dampening** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – 动量控制项（默认为0）
*   **nesterov** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – 是否启用Nesterov动量（默认不启用）

 |
| --- | --- |

Example

```py
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> optimizer.zero_grad()
>>> loss_fn(model(input), target).backward()
>>> optimizer.step()

```


注意

Nesterov 动量更新张量的计算公式与其他的动量计算公式少有不同，区别在于：
![](img/2f90cce3dc946e821ab9d2ae2dfe32c8.jpg)

图中就是一般的参数更新公式，其中p、g、v和ρ分别代表张量、梯度、速度和动量。而在Nesterov动量公式中，张量更新的方式是这样的：

![](img/63bd0746ed6acdf5617d079c80bcfbce.jpg)


```py
step(closure=None)
```


命令优化器执行一次张量更新操作。

| 参数: | **closure** (_callable__,_ _optional_) – 一个可以更新模型中张量大小并且返回误差的回调函数。 |
| --- | --- |

## 如何调整学习速率

`torch.optim.lr_scheduler` 提供了一系列方法可以根据学习的轮数调整学习速率。 [`torch.optim.lr_scheduler.ReduceLROnPlateau`](#torch.optim.lr_scheduler.ReduceLROnPlateau "torch.optim.lr_scheduler.ReduceLROnPlateau") 允许我们通过验证模型参数动态的修改学习速率的大小。

```py
class torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
```
将优化器的学习速率设置为初始学习速率与给定函数的乘积，当参数`last_epoch`=-1时，将初始学习速率设置为当前学习速率。
| 参数: | 

*   **optimizer** ([_Optimizer_](#torch.optim.Optimizer "torch.optim.Optimizer")) – 被调整的优化器。
*   **lr_lambda** (_function_ _or_ [_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)")) – A function which computes a multiplicative factor given an integer parameter epoch, or a list of such functions, one for each group in optimizer.param_groups.
*   **last_epoch** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 上一次The index of last epoch. Default: -1.

 |
| --- | --- |

Example

```py
>>> # Assuming optimizer has two groups.
>>> lambda1 = lambda epoch: epoch // 30
>>> lambda2 = lambda epoch: 0.95 ** epoch
>>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
>>> for epoch in range(100):
>>>     scheduler.step()
>>>     train(...)
>>>     validate(...)

```

```py
load_state_dict(state_dict)
```

读取监视器的状态。

| 参数: | **state_dict** ([_dict_](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.7)")) – 监视器的状态。其返回值与直接调用方法 [`state_dict()`](#torch.optim.lr_scheduler.LambdaLR.state_dict "torch.optim.lr_scheduler.LambdaLR.state_dict")的返回值一样。 |
| --- | --- |

```py
state_dict()
```

返回当前监视器的状态，返回值为字典[`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.7)")类型。这个返回值中包含了成员  self.__dict__ 中每一个张量，但不包含优化器本身。至于修改学习速率的lambda表达式是否保存在返回值中，取决于lambda表达式的类型。如果表达式是一个可回掉的对象，则其会包含在返回值中，否则不会被保存。

```py
class torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
```

在学习周期中每经过step_size次训练，将学习速率设置为初始学习速率与gamma的乘积。当last_epoch=-1时，令初始学习速率等于当前学习速率。

| 参数: | 

*   **optimizer** ([_Optimizer_](#torch.optim.Optimizer "torch.optim.Optimizer")) – 被监视的优化器。
*   **step_size** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 每经过多少次学习修改学习速率。
*   **gamma** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – 每次修改学习速率时，学习速率乘以的因子。默认为0.1。
*   **last_epoch** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 表示最后一次循环的标识，默认为-1。

 |
| --- | --- |

例子

```py
>>> # Assuming optimizer uses lr = 0.05 for all groups
>>> # lr = 0.05     if epoch < 30
>>> # lr = 0.005    if 30 <= epoch < 60
>>> # lr = 0.0005   if 60 <= epoch < 90
>>> # ...
>>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
>>> for epoch in range(100):
>>>     scheduler.step()
>>>     train(...)
>>>     validate(...)

```

```py
class torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
```

预先定义一个数组s，当训练周期数i等于s中任意一个元素时，令当前学习速率等于初始学习速率乘以gamma。当last_epoch=-1时，令初始学习速率等于当前学习速率。

| 参数: | 

*   **optimizer** ([_Optimizer_](#torch.optim.Optimizer "torch.optim.Optimizer")) – 被监视的优化器。
*   **milestones** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)")) – 预定义的数组s，需要注意的是，s中的元素必须是有序且递增的。
*   **gamma** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – 修改学习速率时乘以的因子。默认为0.1。
*   **last_epoch** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 标识最后一个训练周期的标识。默认为-1。

 |
| --- | --- |

例子

```py
>>> # Assuming optimizer uses lr = 0.05 for all groups
>>> # lr = 0.05     if epoch < 30
>>> # lr = 0.005    if 30 <= epoch < 80
>>> # lr = 0.0005   if epoch >= 80
>>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
>>> for epoch in range(100):
>>>     scheduler.step()
>>>     train(...)
>>>     validate(...)

```

```py
class torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
```

Set the learning rate of each parameter group to the initial lr decayed by gamma every epoch. When last_epoch=-1, sets initial lr as lr.

| 参数: | 

*   **optimizer** ([_Optimizer_](#torch.optim.Optimizer "torch.optim.Optimizer")) – Wrapped optimizer.
*   **gamma** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – Multiplicative factor of learning rate decay.
*   **last_epoch** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – The index of last epoch. Default: -1.

 |
| --- | --- |

```py
class torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
```

Set the learning rate of each parameter group using a cosine annealing schedule, where ![](img/2f9e362a8e230566b17e8fc7b4eb533b.jpg) is set to the initial lr and ![](img/e2d59d3a9a4c76df4ed231b491dda3d5.jpg) is the number of epochs since the last restart in SGDR:

![](img/886672c91b10a5c2c26bb14fc638ba50.jpg)

When last_epoch=-1, sets initial lr as lr.

It has been proposed in [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983). Note that this only implements the cosine annealing part of SGDR, and not the restarts.

| 参数: | 

*   **optimizer** ([_Optimizer_](#torch.optim.Optimizer "torch.optim.Optimizer")) – Wrapped optimizer.
*   **T_max** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Maximum number of iterations.
*   **eta_min** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – Minimum learning rate. Default: 0.
*   **last_epoch** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – The index of last epoch. Default: -1.

 |
| --- | --- |

```py
class torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
```

当优化器的损失函数无法进一步缩小时，降低学习速率。对于许多模型来说，一旦损失函数在学习了2-10个循环后仍无法降低，降低学习速率通常能改善该情况。该方法会观察优化器在学习过程中损失函数的大小，如果在经过了`patience`个循环后，损失依旧无法江都，该方法会降低学习速率。


| 参数: | 

*   **optimizer** ([_Optimizer_](#torch.optim.Optimizer "torch.optim.Optimizer")) – 被观察的优化器。
*   **mode** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")) – `min`或者 `max`，如果是`min`标识当损失无法继续减小时，降低学习速率。而`max`标识当损失无法继续增大时，改变学习速率。默认为 `min`。
*   **factor** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – 当学习速率需要降低时，降低的比例。默认为0.1。
*   **patience** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 降低学习速率前最少学习多少个循环。例如当`patience = 2`时，前两次循环不会降低学习速率。默认为10。
*   **verbose** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果是 `True` 则当张量更新时，将更新信息输入。默认为`False`（不输出）。
*   **threshold** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – 区分损函数是否继续改变的阈值，如果损失函数的变化低于这个阈值，则认为损失函数处于“平原”区，考虑降低学习速率（默认1e-4）。
*   **threshold_mode** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")) – One of `rel`, `abs`. In `rel` mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in `min` mode. In `abs` mode, dynamic_threshold = best + threshold in `max` mode or best - threshold in `min` mode. Default: ‘rel’.
*   **cooldown** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
*   **min_lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)")) – 一个数字或者一个数组。如果是一个数组，标识传入的每一组张量的最小学习速率。默认为0。
*   **eps** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – 学习速率最小衰减值。如果新的学习速率和旧的学习速率之间的差小于该值，则本次张量更新操作将被忽略。默认为1e-8。

 |
| --- | --- |

Example

```py
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> scheduler = ReduceLROnPlateau(optimizer, 'min')
>>> for epoch in range(10):
>>>     train(...)
>>>     val_loss = validate(...)
>>>     # Note that step should be called after validate()
>>>     scheduler.step(val_loss)

```

