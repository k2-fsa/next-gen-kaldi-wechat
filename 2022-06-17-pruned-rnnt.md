# 多快好省的 RNN-T 训练

> 本文将介绍新一代 Kaldi 团队在 RNN-T 训练上近期所做的工作。其中内容已通过论文
>
> “Pruned RNN-T for fast, memory-efficient ASR training” 
>
> 投稿至 Interspeech 2022。该论文已经被**接收**，将于近期发表。



## 简介

在 CTC 模型、RNN-T 模型和 attention-based 模型中，RNN-T 是最适合用于产线部署的流式解码模型。但 RNN-T 具有以下几个痛点:

  - 训练时，与其他模型相比，占用的内存至少高一个数量级
  - 解码时，很难通过以 batch 的方式实现并行解码

新一代 Kaldi 团队近期所做的工作致力于解决上面两个痛点。本文介绍训练部分。解码部分，将于近期在 “新一代 Kaldi” 公众号**连载**。

基于 RNN-T 模型，我们已经在若干个大数据集的 Leaderboard 上，排名第一，比如 GigaSpeech（1 万小时英文）和 WenetSpeech (1万小时中文）。

> 小编注: 新一代 Kaldi 团队所做的工作，全部都是开源的。代码链接如下：
> - [k2](https://github.com/k2-fsa/k2) （底层实现）
> - [icefall](https://github.com/k2-fsa/icefall)  （训练）
> - [lhotse](https://github.com/lhotse-speech/lhotse)（数据处理）
> - [sherpa](https://github.com/k2-fsa/sherpa) （部署）

## 要解决的问题

RNN-T 模型最后一层的输出是一个 4-D 的 tensor，维度是 `(N, T, U, C)`, 其中
- `N`: batch size。数值大小: 一般是几十
- `T`:  encoder 的输出帧数。数值大小：一般是好几百
- `U`: decoder 的输出帧数。数值大小：几十至上百
- `C`: vocabulary size。数值大小：几百至上千

所以，RNN-T 训练时，所需的内存正比于 `N`, `T` , `U`, `C` 这 4 个数的乘积 `NTUC`。训练 CTC 或者 attention-based 模型时，所需的内存一般与 `NTC` 或者 `NUC` 成正比。

相比较之下，RNN-T 模型的训练，对内存的要求高了一个数量级。为了避免训练时出现 out-of-memory (OOM) 错误，通常的做法是：
- 减少 `N`，使用一个小的 batch size
- 减少 `C`，使用一个较小的 vocabulary size
- 降低模型参数量

但是，使用小的 batch size 会增加模型训练所需的时间；而使用小的 vocabulary size， 可能会影响模型的性能。例如，若以单个汉字为建模单元，vocabulary size 一般是 4000 到 7000 之间。如果使用一个很小的 vocabulary size，那么对于 out-of-vocabulary (OOV) 这种问题，就会更加常见。而降低模型的参数量，也会影响模型的性能。

那么如何在不降低模型性能的前提下，做到以下几点呢?

- 降低训练时所需的内存
- 降低训练所需的时间

微软的研究者们在论文

[Improving RNN Transducer Modeling for End-to-End Speech Recognition](https://arxiv.org/abs/1909.12415)

中提出，可以通过移除一个 batch 中所需的 padding 来减少内存的占用量。

我们对上述方法提供了一个开源的实现，链接如下:
[optimized_transducer](https://github.com/csukuangfj/optimized_transducer)

相关 Benchmark 数据表明，这种方法在所有**标准的 RNN-T** 开源实现中，所需内存最低。

## 我们提出的方法

新一代 Kaldi 团队提出的 **pruned RNN-T**， 采用了一种截然不同的处理方式来降低内存使用量。相关 bechmark 结果表明，pruned RNN-T 具有如下特点:

- 多
- 快
- 好
- 省

其中具体含义，解释如下：

### “多”
- 可以用更大的 batch size
- 可以用更大的 vocabulary size

### “快”
- 目前为止，所有常用开源实现中，训练速度最快
- 训练出来的模型，在不降低性能的前提下，解码速度更快

### “好”

- 训练出来的模型，在 GigaSpeech 和 WenetSpeech 的 Leaderboard 上，排名第一

### “省”

- 省内存
- 目前为止，所有常用开源实现中，所需内存最少

那么，pruned RNN-T 究竟是怎么做到 “多快好省”的呢？

**简单直白**一点的说，pruned RNN-T 改变了 RNN-T 模型中最后一层的输出维度：
- 从 `(N, T, U, C)` 变成了 `(N, T, S, C)`

其中，`S` 是用户指定的一个参数。我们所做的实验中，一般选用 5；而 `U` 则一般是几十甚至上百。

结果就是，所需的内存与 `NTSC` 成正比，而不再是 `NTUC`。

我们下面截取论文中的几个图，来**感性地**认识一下  pruned RNN-T。

![](https://github.com/k2-fsa/next-gen-kaldi-wechat/blob/master/pic/2022-06-17-figure.png)

图1（a）是针对标准的 RNN-T，图中所有的结点都参与了RNN-T loss 的计算。

我们**也许是第一个**提出下面这个问题的人：
- 是不是所有的结点都应该参与计算？

图2 显示了图 1（a）中每个结点在训练时某一时刻的梯度。我们可以看到，随着训练的进行，靠近对角线上的结点对计算起到的作用最大。也就是说，不同位置的结点，在训练中起到的作用不同。

图1（b）则是针对 pruned RNN-T，图中只有部分节点参与了 RNN-T loss 的计算。

参与计算的节点数量越少，所需的计算量则越少、计算速度就越快，并且所需的内存也越少。

那么，问题就来了：
- 哪些结点应该被选出来参与计算呢？
- 又如何选择这些结点呢？

本文不探讨上述问题。感兴趣的读者可以阅读随后放出来的论文，同时也可以阅读具体的实现代码。代码链接如下：[rnnt_loss.py](https://github.com/k2-fsa/k2/blob/master/k2/python/k2/rnnt_loss.py)

## 效果

在这部分，我们向大家汇报 pruned RNN-T 在以下几方面的结果:

- 训练速度
- 训练所需内存
- 在LibriSpeech test-clean 测试集上的 WER 及 RTF

在进行 benchmark 时，我们采用 LibriSpeech test-clean 测试集来生成 RNN-T 模型训练时所需的维度信息， 而不是针对特定的维度进行 benchmark。这样可以考虑每个 batch 中 padding 所造成的影响，尽量还原真实的应用场景。

我们设置了两种 bechmark 模式:
- (1) 以随机的方式组成 batch。batch size 为 30。
- (2) 按照样本时长进行排序的方式组成 batch。每个 batch 中最多包含 1万 帧特征。

在论文中，我们对比了 pruned RNN-T 和 常用的开源 RNN-T loss 实现的性能。结果如表 1 和 表 2 所示。我们可以看出，不管是在训练时间还是在内存使用量上， pruned RNN-T 与其他实现相比，都有很大的优势。

![](https://github.com/k2-fsa/next-gen-kaldi-wechat/blob/master/pic/2022-06-17-table.png)

Bechmark 的代码已开源在如下 repo 中: [ransducer-loss-benchmarking](https://github.com/csukuangfj/transducer-loss-benchmarking)

> 小编注: 论文中，我们只列出了若干个常用的开源实现。上述 repo 中，我们对比了更多的开源实现。

表 3 对比了 pruned RNN-T 和  optimized_transducer 在 LibriSpeech 数据集上的训练时间和性能。我们可以看出，在不损失性能的前提下，pruned RNN-T 在训练速度上有绝对的优势。

![](https://github.com/k2-fsa/next-gen-kaldi-wechat/blob/master/pic/2022-06-17-table-2.png)

在提交论文后，我们对使用的模型做了进一步的优化。截止目前，在**不使用任何外部语言模型**的情况下，pruned RNN-T 在 LibriSpeech test-clean 测试集 上的 WER 是 **2.00**， 在 test-other 上 的 WER 是 4.63。

关于 RTF 方面，我们使用 Colab notebook 里面提供的 GPU 进行测试，对 LibriSpeech test-clean进行解码。得到的 **RTF** 是 **0.0090** (使用greedy search)， WER 是 2.05。Colab notebook 的访问链接如下: [Colab notebook](https://colab.research.google.com/drive/1JX5Ph2onYm1ZjNP_94eGqZ-DIRMLlIca?usp=sharing)

> 小编注: Colab notebook 提供的 GPU 型号是 Tesla T4, RAM 大小是 14.75 GB。如果你有更大 RAM 的 GPU, 例如如 32 GB 的 V100，那么你得到的 RTF 会更低。

## 总结

本文介绍了新一代 Kaldi 中 “多快好省”的 RNN-T 训练方法 --- pruned RNN-T。在训练速度和内存使用量上，pruned RNN-T 优于所有常用的开源实现。并且，使用 pruned RNN-T 训练的模型，已经在若干个大数据集上做到了 state-of-the-art （SOTA）的性能。

我们希望，pruned RNN-T 的开源，能够助力 RNN-T 在生产环境中的使用，缩短模型训练所需的时间，简化端到端模型的部署，为企业节约成本。
