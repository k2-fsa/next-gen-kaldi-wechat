# 低时延 RNN-T 训练

> Dan神又出新东西了！ ~~(前面的都学完了吗？)~~   
> 本文主要介绍 `k2` 中的低时延 `RNN-T` 训练，这是一篇短小的写给~~懒人~~的科普文，不会有详细的理论推导，感兴趣的大佬可以直接阅读论文：**https://arxiv.org/pdf/2211.00490.pdf**

## 流式模型与时延
这里说的时延主要是指由模型本身带来的输出延迟，比如一个字是在第 100 帧说的，但是直到送了 150 帧数据进去才输出来。时延问题可以说是端到端模型基因里带来的缺点，一个大家都比较认可的解释是，`RNN-T/CTC` 这样基于序列的损失函数对于 `Alignments` 的优化是无差别的，即只管优化能输出 transcript 对应的路径，不管这个路径是先输出 `symbol` 还是先输出 `blank`。所以，对于流式模型的训练，由于当前看到的 `context` 有限，模型总是倾向于看到更多的数据后再决定是否输出 `symbol`。

![](https://files.mdnice.com/user/32622/3020d35f-681d-4c87-ab26-331508f01341.png)
<center>图一</center>

从图一中可以获得一个感性的认识，图中从上到下三条线分别代表：没有使用时延正则的流式模型，使用了时延正则的流式模型和非流式模型在训练过程中的时延曲线。  
可以看到，非流式模型在训练过程中的时延几乎是不变的，而且由于能看到全部 `right context`，时延是很低的。而对于流式模型，可以看到随着模型优化得越来越好，时延反而越来越大，这也从侧面验证了模型倾向于看更多的数据来提高输出置信度。中间那条线是使用了我们提出的时延正则的流式训练，可以看到时延是随着模型的优化持续降低。

## 路径与时延
对于 `RNN-T` 的训练 `lattice` 是一个 `U * T` 的矩阵（如图二所示），理论上从左下角到右上角的所有路径都是合法的路径，由于向上是输出 `symbol`，向右是输出 `blank`，所以偏向左上角的路径的时延要小于偏向右下角的路径，即图中红色路径的时延比蓝色路径的时延低。

![](https://files.mdnice.com/user/32622/de0b847a-8a73-46d7-ac50-eef3b0467342.png)

<center>图二</center>

## 时延正则
时延正则的目标是给低时延的路径一些鼓励（加分），给高时延的路径一些抑制（减分）。**此处有非常详细的理论推理，十几个公式，这里就不展开了，感兴趣的大佬可以读原论文（链接见文章开头）**。最终的实现就是给 `lattice` 中每条输出 `symbol` 的边加一个分数，这个分数根据边所在的帧而不同，以中轴线为基准，左侧加`正值`（鼓励），右侧加`负值`（惩罚），示意图如图三所示。这样位于左上角的路径的分数得到增强，位于右下角的路径分数会被抑制，从而达到降低时延的目的。

![](https://files.mdnice.com/user/32622/c258b4d1-fff8-4830-af03-090a5d7ab987.png)

<center>图三</center>

## 实验及结果
目前 [k2](https://github.com/k2-fsa/k2 "k2") 和 [fast_rnnt](https://github.com/danpovey/fast_rnnt "fast_rnnt") 两个仓库都已经合并了 `delay-penalty` 的实现(见 [delay-penalty](https://github.com/k2-fsa/k2/pull/976 "delay-penalty"))，只需要在使用 `pruned rnnt` 损失函数时多传入一个 `delay_penalty` 参数就可以实现低延时的 `RNN-T` 训练（注意：`rnnt_loss_smoothed` 和 `rnnt_loss_pruned` 两个地方都要加）。
 我们在 Streaming Conformer 和 LSTM 上都做了一些实验，结果证明我们提出的时延正则方法很有效果，并且能简单的通过调整超参数来平衡准确率和时延。结果中的 `MAD` 表示 token 的平均时延，`MED` 表示最后一个 token 的平均时延，时延都是根据 [Montreal-Forced-Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner "Montreal-Forced-Aligner") 对齐结果来计算的。

![](https://files.mdnice.com/user/32622/42e511d3-62bc-43c9-847e-44fd2dd9dc18.png)

我们还对比了使用不同 `chunk size` 解码的结果，`chunk` 解码本身就会带来时延，`chunk size` 越大，带来的时延越大。下图是不同 `chunk size` 解码的准确率和时延的关系图（这里的时延为总时延，即 `chunk_size / 2 * MAD`). 可以看出，使用大些的 `chunk size`，在相同时延情况下，可以取得更好的准确率。

![](https://files.mdnice.com/user/32622/66073f7b-e018-40e0-a985-1b5dabe017b6.png)

另外，说起时延不得不提 Google 提出的 [FastEmit](https://arxiv.org/pdf/2010.11148.pdf "FastEmit"), 我们也与 `fast emit` 做了对比，发现结果不相上下，有时略好。不过我们相信我们的方法有一个更清晰的理论解释（比如考虑了 `symbol` 输出的时间信息）。

![](https://files.mdnice.com/user/32622/8a2b7cd5-6bea-4f6e-a9dd-100a3f9028ce.png)

当然，如果执念要使用 `FastEmit`，我们在 `k2` 中也提供了实现，见 [k2 FastEmit](https://github.com/k2-fsa/k2/pull/1069 "k2 FastEmit")，~~合并是不会合并的~~, 欢迎尝试。

## 总结
本文介绍了`新一代 Kaldi` 中提出的`低时延 RNN-T` 训练的方法，粗略介绍了时延产生的原因，阐明了我们做时延正则的方案，欢迎大家尝鲜！欲知更多细节和推导，请阅读原论文：**https://arxiv.org/pdf/2211.00490.pdf**。

