# 新一代 Kaldi 中的 RandomCombiner

> 本文介绍新一代 Kaldi 中的 RandomCombiner：
>
> 相关代码：https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless5/conformer.py

## 1. 残差连接

对于 [ResNet](https://arxiv.org/pdf/1512.03385.pdf) 中的残差连接，大家应该十分熟悉。残差连接已被广泛应用于各种网络模型中，如 Transformer 和 Conformer 等，用于解决训练深层模型时遇到的梯度消失问题。

举个简单的例子，在以下的公式中，$f(x)$ 表示某个神经网络模块，$x$ 表示该模块的输入，$y$ 表示残差连接后的输出：
$$
y = f(x) + x
$$
我们在对 $y$ 反向求导的过程中，除了通过模块 $f(x)$ 外，还会通过残差连接的一项，将梯度直接传递给输入 $x$。

## 2. RandomCombiner 方法介绍

为了训练深层模型，如 18 或者 24 层的 Conformer，利用残差连接的思想设计了 RandomCombiner 模块，其核心操作为：

* 在训练的过程中，RandomCombiner 会**随机结合不同的层以及最后一层的输出**，作为模型的最终输出；因此，损失函数的梯度可以直接传递到浅层网络，稳定训练过程。

* 在解码的过程中，RandomCombiner **只返回最后一层的输出**。

> 使用 RandomCombiner 时， 以某个周期选择用于结合的层数，如每 3 层选择一层。

### 随机结合机制

RandomCombiner 同时使用了两种随机结合的策略，来结合最后一层和其它层的输出，分别为 **one-hot** 策略和**加权求和**策略。

* one-hot 策略，参考函数 `_get_random_pure_weights`

对于每一帧而言，以概率 $p$ 选择最后一层；以概率 $1-p$ 随机选择其它层中的某一层，可参考代码块：

```
# self.num_inputs is the number of layers to be combined
# final_prob is the probability for the last layer: p

# select the last layer
final = torch.full(
  (num_frames,), self.num_inputs - 1, device=device
)

# select one of other layers
nonfinal = torch.randint(
  self.num_inputs - 1, (num_frames,), device=device
)

indexes = torch.where(
  torch.rand(num_frames) < final_prob, final, nonfinal
)
ans = torch.nn.functional.one_hot(
  indexes, num_classes=self.num_inputs
)
```

* 加权求和策略，参考函数 `_get_random_mixed_weights`

对每一帧而言，生成对应于 $N$ 个层的随机数 $x = \{x_1, x_2, \dots, x_N\}$，其中 $x_N$ 对应于最后一层。通过在 $x_N$ 基础上增加 $\log \frac{p \times (N-1)}{1 - p}$，调整最后一层与其它层之间的比例，并应用 $\text{softmax}$ 函数进行归一化，得到对应于 $N$ 个层 的权重 $w = \{w_1, w_2, \dots, w_N\}$。

对应于最后一层的权重为：
$$
w_N = \frac{p \times \exp(x_N)}{S}
$$
对应于其它各个层的权重为：
$$
w_{j\in\{1,2,\dots, N-1\}} = \frac{\frac{1-p}{N-1} \times \exp(x_j) }{S}
$$
其中，归一化分母为：
$$
S = p \times \exp(x_N)+ (1-p)\sum_{j \neq N}\exp(x_j)
$$

代码片段为：
```

""" defined in __init__ function
self.final_log_weight = (
  torch.tensor(
    (final_weight / (1 - final_weight)) * (self.num_inputs - 1)
  ).log().item()
)
"""

logprobs = (
  torch.randn(
    num_frames, self.num_inputs, dtype=dtype, device=device
  )
  * self.stddev
)
logprobs[:, -1] += self.final_log_weight

weights = logprobs.softmax(dim=1)
```

* 最后，对于每一帧，随机选择上述两种结合策略中的其中一种，即 **one-hot** 或者 **加权求和**，可参考函数 `_get_random_weights`:
```
# one-hot weights
p = self._get_random_pure_weights(
  dtype, device, num_frames
)

# float weights
m = self._get_random_mixed_weights(
  dtype, device, num_frames
)

return torch.where(
  torch.rand(num_frames, 1, device=device) < self.pure_prob, p, m
)
```

> 值得注意得是，上述策略独立地应用于不同的 batch 中，即不同的 batch 会生成不同的随机数。

## 3. 实验结果

RandomCombiner 实现于 [Reworked Conformer](https://mp.weixin.qq.com/s/2WrEh3wHzYE6TCKuw_laLw) 之前，详情可参考 Dan 的 PR https://github.com/k2-fsa/icefall/pull/229 。

> Reworked Conformer 中的 model-level warmup，同样采用了残差连接方式来稳定训练过程。

（1） 根据 Dan 在该 PR 中介绍，使用 train-clean-100 训练 Conformer 模型，在 test-clean 和 test-other 测试集上利用 greedy search 解码，使用了 RandomCombiner 可以将 WER 从 8.xx / 22.xx 降低到 7.58 / 20.36。表明对于 **普通的 Conformer 模型**，RandomCombiner 明显有助于模型收敛。

（2） 在 Reworked Conformer 上应用 RandomCombiner 仍然可以得到轻微的性能提升。下表比较了使用 full librispeech 训练时，在 test-clean 和 test-other 测试集上 的 WER。

* 没有使用 RandomCombiner

| parameters | encoder layers | feedforward dim |  heads | encoder dim | greedy search | modified beam search | fast beam search | comment |
|----|----|----|-----|-----|----|----|----|---|
|87.8M| 24 | 1536 | 8 |  384 | 2.48/5.80 | 2.45/5.72 | 2.45/5.71| --epoch 34 --avg 19 |
|30.5M| 18 | 1024 |4| 256| 2.82/6.99|2.78/6.82|2.77/6.91|--epoch 39 --avg 6|
|116.55M|18|2048|8|512|2.42/5.77|2.39/5.73|2.39/5.73| --epoch 39 --avg 13 |

* 使用 RandomCombiner

| parameters | encoder layers | feedforward dim | num heads | encoder dim | greedy search | modified beam search | fast beam search | comment |
|----|----|----|-----|-----|----|----|----|---|
|88.98M|24 |1536 |8|384|**2.41**/**5.70**| **2.41**/**5.69**| **2.41/5.69**| --epoch 31 --avg 17|
|30.9M|18|1024|4|256|2.88/**6.69**|2.83/**6.59**|2.83/**6.61**|--epoch 39 --avg 17 |
|118.13M|18|2048|8|512|**2.39**/**5.57**|**2.35**/**5.50**|**2.38**/**5.50**|--epoch 39 --avg 7|
