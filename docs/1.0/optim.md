

# torch.optim

[`torch.optim`](#module-torch.optim "torch.optim") 是一个优化器的集合包。目前绝大多数常用的优化器在这个包都已经被实现，并且由于接口有很强的适用性很强，更加复杂的优化器可以很方便的被套用出来。

## 如何使用优化器？

想要使用优化器 [`torch.optim`](#module-torch.optim "torch.optim")，首先要构建一个优化器对象。它能保存当前参数的状态并且根据计算出的梯度更新参数。

### 构建优化器

想要构建一个优化器 [`Optimizer`](#torch.optim.Optimizer "torch.optim.Optimizer") ，首先要给它一个包含参数的迭代器去优化（所有参数都必须是`Variable`类型）。然后你可以指定一些优化参数，例如学习速率、权重衰减等。

注意

如果你想通过方法 `.cuda()` 使一个模型的参数存储在GPU，那么该操作必须在对于该模型构建优化器之前。因为在调用方法`.cuda()`之后，该模型的所有参数都将被更新且新的参数与旧参数将看作是不同的两组参数。

一般来说，在优化器被构建和使用的过程中，需要保证被优化的参数在内存中或显存中的地址是不变的。

例子：

```py
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr = 0.0001)

```

### 为多个优化变量指定不同参数

当传入参数为一组字典[`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.7)")时，表示优化器[`Optimizer`](#torch.optim.Optimizer "torch.optim.Optimizer") 为多组待优化的变量指定不同的优化参数。每个字典中必须包含键`params`，其他值是优化器对`params`指定的变量优化时使用的参数。


注意

You can still pass options as keyword arguments. They will be used as defaults, in the groups that didn’t override them. This is useful when you only want to vary a single option, while keeping all others consistent between parameter groups.

For example, this is very useful when one wants to specify per-layer learning rates:

```py
optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)

```

其中`model.base` 中的变量在被优化时，其学习速率为默认的`1e-2`, 而在优化`model.classifier`中的参数时，其学习速率为`1e-3`，且动量为0.9。

### Taking an optimization step

所有的优化器都实现了一个参数更新方法[`step()`](#torch.optim.Optimizer.step "torch.optim.Optimizer.step")。该方法有两种使用方式：

#### `optimizer.step()`

这是一种被大多数优化器支持的简便使用方法。在梯度被计算`backward()`完成之后，该方法才可以被调用。

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

有些优化器例如共轭梯度算法和LBFGS等算法在计算的过程中需要多次更新算法的状态。所以你需要传入一个函数，让优化器在合适的时候自动去调用传入的方法。传入的方法一般需要将梯度清空，计算损失函数且能返回误差。

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

## Algorithms

```py
class torch.optim.Optimizer(params, defaults)
```

该类是所有优化器的基类。

Warning

Parameters need to be specified as collections that have a deterministic ordering that is consistent between runs. Examples of objects that don’t satisfy those properties are sets and iterators over values of dictionaries.

| Parameters: | 

*   **params** (_iterable_) – an iterable of [`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor") s or [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.7)") s. Specifies what Tensors should be optimized.
*   **defaults** – (dict): a dict containing default values of optimization options (used when a parameter group doesn’t specify them).

 |
| --- | --- |

```py
add_param_group(param_group)
```

Add a param group to the [`Optimizer`](#torch.optim.Optimizer "torch.optim.Optimizer") s `param_groups`.

This can be useful when fine tuning a pre-trained network as frozen layers can be made trainable and added to the [`Optimizer`](#torch.optim.Optimizer "torch.optim.Optimizer") as training progresses.

| Parameters: | 

*   **param_group** ([_dict_](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.7)")) – Specifies what Tensors should be optimized along with group
*   **optimization options.** (_specific_) –

 |
| --- | --- |

```py
load_state_dict(state_dict)
```

Loads the optimizer state.

| Parameters: | **state_dict** ([_dict_](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.7)")) – optimizer state. Should be an object returned from a call to [`state_dict()`](#torch.optim.Optimizer.state_dict "torch.optim.Optimizer.state_dict"). |
| --- | --- |

```py
state_dict()
```

Returns the state of the optimizer as a [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.7)").

It contains two entries:

*   ```py
    state - a dict holding current optimization state. Its content
    ```

    differs between optimizer classes.
*   param_groups - a dict containing all parameter groups

```py
step(closure)
```

Performs a single optimization step (parameter update).

| Parameters: | **closure** (_callable_) – A closure that reevaluates the model and returns the loss. Optional for most optimizers. |
| --- | --- |

```py
zero_grad()
```

Clears the gradients of all optimized [`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor") s.

```py
class torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
```

Implements Adadelta algorithm.

It has been proposed in [ADADELTA: An Adaptive Learning Rate Method](https://arxiv.org/abs/1212.5701).

| Parameters: | 

*   **params** (_iterable_) – iterable of parameters to optimize or dicts defining parameter groups
*   **rho** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – coefficient used for computing a running average of squared gradients (default: 0.9)
*   **eps** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – term added to the denominator to improve numerical stability (default: 1e-6)
*   **lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – coefficient that scale delta before it is applied to the parameters (default: 1.0)
*   **weight_decay** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – weight decay (L2 penalty) (default: 0)

 |
| --- | --- |

```py
step(closure=None)
```

Performs a single optimization step.

| Parameters: | **closure** (_callable__,_ _optional_) – A closure that reevaluates the model and returns the loss. |
| --- | --- |

```py
class torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
```

Implements Adagrad algorithm.

It has been proposed in [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://jmlr.org/papers/v12/duchi11a.html).

| Parameters: | 

*   **params** (_iterable_) – iterable of parameters to optimize or dicts defining parameter groups
*   **lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – learning rate (default: 1e-2)
*   **lr_decay** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – learning rate decay (default: 0)
*   **weight_decay** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – weight decay (L2 penalty) (default: 0)

 |
| --- | --- |

```py
step(closure=None)
```

Performs a single optimization step.

| Parameters: | **closure** (_callable__,_ _optional_) – A closure that reevaluates the model and returns the loss. |
| --- | --- |

```py
class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```

Implements Adam algorithm.

It has been proposed in [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).

| Parameters: | 

*   **params** (_iterable_) – iterable of parameters to optimize or dicts defining parameter groups
*   **lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – learning rate (default: 1e-3)
*   **betas** (_Tuple__[_[_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_]__,_ _optional_) – coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
*   **eps** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – term added to the denominator to improve numerical stability (default: 1e-8)
*   **weight_decay** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – weight decay (L2 penalty) (default: 0)
*   **amsgrad** (_boolean__,_ _optional_) – whether to use the AMSGrad variant of this algorithm from the paper [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ) (default: False)

 |
| --- | --- |

```py
step(closure=None)
```

Performs a single optimization step.

| Parameters: | **closure** (_callable__,_ _optional_) – A closure that reevaluates the model and returns the loss. |
| --- | --- |

```py
class torch.optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
```

Implements lazy version of Adam algorithm suitable for sparse tensors.

In this variant, only moments that show up in the gradient get updated, and only those portions of the gradient get applied to the parameters.

| Parameters: | 

*   **params** (_iterable_) – iterable of parameters to optimize or dicts defining parameter groups
*   **lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – learning rate (default: 1e-3)
*   **betas** (_Tuple__[_[_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_]__,_ _optional_) – coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
*   **eps** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – term added to the denominator to improve numerical stability (default: 1e-8)

 |
| --- | --- |

```py
step(closure=None)
```

Performs a single optimization step.

| Parameters: | **closure** (_callable__,_ _optional_) – A closure that reevaluates the model and returns the loss. |
| --- | --- |

```py
class torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
```

Implements Adamax algorithm (a variant of Adam based on infinity norm).

It has been proposed in [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).

| Parameters: | 

*   **params** (_iterable_) – iterable of parameters to optimize or dicts defining parameter groups
*   **lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – learning rate (default: 2e-3)
*   **betas** (_Tuple__[_[_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_]__,_ _optional_) – coefficients used for computing running averages of gradient and its square
*   **eps** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – term added to the denominator to improve numerical stability (default: 1e-8)
*   **weight_decay** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – weight decay (L2 penalty) (default: 0)

 |
| --- | --- |

```py
step(closure=None)
```

Performs a single optimization step.

| Parameters: | **closure** (_callable__,_ _optional_) – A closure that reevaluates the model and returns the loss. |
| --- | --- |

```py
class torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
```

Implements Averaged Stochastic Gradient Descent.

It has been proposed in [Acceleration of stochastic approximation by averaging](http://dl.acm.org/citation.cfm?id=131098).

| Parameters: | 

*   **params** (_iterable_) – iterable of parameters to optimize or dicts defining parameter groups
*   **lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – learning rate (default: 1e-2)
*   **lambd** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – decay term (default: 1e-4)
*   **alpha** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – power for eta update (default: 0.75)
*   **t0** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – point at which to start averaging (default: 1e6)
*   **weight_decay** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – weight decay (L2 penalty) (default: 0)

 |
| --- | --- |

```py
step(closure=None)
```

Performs a single optimization step.

| Parameters: | **closure** (_callable__,_ _optional_) – A closure that reevaluates the model and returns the loss. |
| --- | --- |

```py
class torch.optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
```

Implements L-BFGS algorithm.

Warning

This optimizer doesn’t support per-parameter options and parameter groups (there can be only one).

Warning

Right now all parameters have to be on a single device. This will be improved in the future.

Note

This is a very memory intensive optimizer (it requires additional `param_bytes * (history_size + 1)` bytes). If it doesn’t fit in memory try reducing the history size, or use a different algorithm.

| Parameters: | 

*   **lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – learning rate (default: 1)
*   **max_iter** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – maximal number of iterations per optimization step (default: 20)
*   **max_eval** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – maximal number of function evaluations per optimization step (default: max_iter * 1.25).
*   **tolerance_grad** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – termination tolerance on first order optimality (default: 1e-5).
*   **tolerance_change** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – termination tolerance on function value/parameter changes (default: 1e-9).
*   **history_size** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – update history size (default: 100).

 |
| --- | --- |

```py
step(closure)
```

Performs a single optimization step.

| Parameters: | **closure** (_callable_) – A closure that reevaluates the model and returns the loss. |
| --- | --- |

```py
class torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
```

Implements RMSprop algorithm.

Proposed by G. Hinton in his [course](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).

The centered version first appears in [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850v5.pdf).

| Parameters: | 

*   **params** (_iterable_) – iterable of parameters to optimize or dicts defining parameter groups
*   **lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – learning rate (default: 1e-2)
*   **momentum** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – momentum factor (default: 0)
*   **alpha** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – smoothing constant (default: 0.99)
*   **eps** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – term added to the denominator to improve numerical stability (default: 1e-8)
*   **centered** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – if `True`, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
*   **weight_decay** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – weight decay (L2 penalty) (default: 0)

 |
| --- | --- |

```py
step(closure=None)
```

Performs a single optimization step.

| Parameters: | **closure** (_callable__,_ _optional_) – A closure that reevaluates the model and returns the loss. |
| --- | --- |

```py
class torch.optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
```

Implements the resilient backpropagation algorithm.

| Parameters: | 

*   **params** (_iterable_) – iterable of parameters to optimize or dicts defining parameter groups
*   **lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – learning rate (default: 1e-2)
*   **etas** (_Tuple__[_[_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_]__,_ _optional_) – pair of (etaminus, etaplis), that are multiplicative increase and decrease factors (default: (0.5, 1.2))
*   **step_sizes** (_Tuple__[_[_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_]__,_ _optional_) – a pair of minimal and maximal allowed step sizes (default: (1e-6, 50))

 |
| --- | --- |

```py
step(closure=None)
```

Performs a single optimization step.

| Parameters: | **closure** (_callable__,_ _optional_) – A closure that reevaluates the model and returns the loss. |
| --- | --- |

```py
class torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
```

Implements stochastic gradient descent (optionally with momentum).

Nesterov momentum is based on the formula from [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf).

| Parameters: | 

*   **params** (_iterable_) – iterable of parameters to optimize or dicts defining parameter groups
*   **lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – learning rate
*   **momentum** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – momentum factor (default: 0)
*   **weight_decay** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – weight decay (L2 penalty) (default: 0)
*   **dampening** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – dampening for momentum (default: 0)
*   **nesterov** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – enables Nesterov momentum (default: False)

 |
| --- | --- |

Example

```py
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> optimizer.zero_grad()
>>> loss_fn(model(input), target).backward()
>>> optimizer.step()

```

Note

The implementation of SGD with Momentum/Nesterov subtly differs from Sutskever et. al. and implementations in some other frameworks.

Considering the specific case of Momentum, the update can be written as

![](img/2f90cce3dc946e821ab9d2ae2dfe32c8.jpg)

where p, g, v and ![](img/787a6ae8db26f884126803d73bf4d66c.jpg) denote the parameters, gradient, velocity, and momentum respectively.

This is in contrast to Sutskever et. al. and other frameworks which employ an update of the form

![](img/63bd0746ed6acdf5617d079c80bcfbce.jpg)

The Nesterov version is analogously modified.

```py
step(closure=None)
```

Performs a single optimization step.

| Parameters: | **closure** (_callable__,_ _optional_) – A closure that reevaluates the model and returns the loss. |
| --- | --- |

## How to adjust Learning Rate

`torch.optim.lr_scheduler` provides several methods to adjust the learning rate based on the number of epochs. [`torch.optim.lr_scheduler.ReduceLROnPlateau`](#torch.optim.lr_scheduler.ReduceLROnPlateau "torch.optim.lr_scheduler.ReduceLROnPlateau") allows dynamic learning rate reducing based on some validation measurements.

```py
class torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
```

Sets the learning rate of each parameter group to the initial lr times a given function. When last_epoch=-1, sets initial lr as lr.

| Parameters: | 

*   **optimizer** ([_Optimizer_](#torch.optim.Optimizer "torch.optim.Optimizer")) – Wrapped optimizer.
*   **lr_lambda** (_function_ _or_ [_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)")) – A function which computes a multiplicative factor given an integer parameter epoch, or a list of such functions, one for each group in optimizer.param_groups.
*   **last_epoch** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – The index of last epoch. Default: -1.

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

Loads the schedulers state.

| Parameters: | **state_dict** ([_dict_](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.7)")) – scheduler state. Should be an object returned from a call to [`state_dict()`](#torch.optim.lr_scheduler.LambdaLR.state_dict "torch.optim.lr_scheduler.LambdaLR.state_dict"). |
| --- | --- |

```py
state_dict()
```

Returns the state of the scheduler as a [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.7)").

It contains an entry for every variable in self.__dict__ which is not the optimizer. The learning rate lambda functions will only be saved if they are callable objects and not if they are functions or lambdas.

```py
class torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
```

Sets the learning rate of each parameter group to the initial lr decayed by gamma every step_size epochs. When last_epoch=-1, sets initial lr as lr.

| Parameters: | 

*   **optimizer** ([_Optimizer_](#torch.optim.Optimizer "torch.optim.Optimizer")) – Wrapped optimizer.
*   **step_size** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Period of learning rate decay.
*   **gamma** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – Multiplicative factor of learning rate decay. Default: 0.1.
*   **last_epoch** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – The index of last epoch. Default: -1.

 |
| --- | --- |

Example

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

Set the learning rate of each parameter group to the initial lr decayed by gamma once the number of epoch reaches one of the milestones. When last_epoch=-1, sets initial lr as lr.

| Parameters: | 

*   **optimizer** ([_Optimizer_](#torch.optim.Optimizer "torch.optim.Optimizer")) – Wrapped optimizer.
*   **milestones** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)")) – List of epoch indices. Must be increasing.
*   **gamma** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – Multiplicative factor of learning rate decay. Default: 0.1.
*   **last_epoch** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – The index of last epoch. Default: -1.

 |
| --- | --- |

Example

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

| Parameters: | 

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

| Parameters: | 

*   **optimizer** ([_Optimizer_](#torch.optim.Optimizer "torch.optim.Optimizer")) – Wrapped optimizer.
*   **T_max** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Maximum number of iterations.
*   **eta_min** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – Minimum learning rate. Default: 0.
*   **last_epoch** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – The index of last epoch. Default: -1.

 |
| --- | --- |

```py
class torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
```

Reduce learning rate when a metric has stopped improving. Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. This scheduler reads a metrics quantity and if no improvement is seen for a ‘patience’ number of epochs, the learning rate is reduced.

| Parameters: | 

*   **optimizer** ([_Optimizer_](#torch.optim.Optimizer "torch.optim.Optimizer")) – Wrapped optimizer.
*   **mode** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")) – One of `min`, `max`. In `min` mode, lr will be reduced when the quantity monitored has stopped decreasing; in `max` mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
*   **factor** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
*   **patience** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Number of epochs with no improvement after which learning rate will be reduced. For example, if `patience = 2`, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
*   **verbose** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If `True`, prints a message to stdout for each update. Default: `False`.
*   **threshold** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
*   **threshold_mode** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")) – One of `rel`, `abs`. In `rel` mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in `min` mode. In `abs` mode, dynamic_threshold = best + threshold in `max` mode or best - threshold in `min` mode. Default: ‘rel’.
*   **cooldown** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
*   **min_lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)")) – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
*   **eps** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.

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

