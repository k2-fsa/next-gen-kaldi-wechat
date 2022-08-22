# 玩转新一代 Kaldi 流式模型

从大家在 github 和微信群的反馈中我们发现有好些人对新一代 Kaldi 中的流式模型还不甚了解，本文试图作一个引导性的介绍。

> 本文只是引导性介绍，不包含如何安装软件，也不会有 step by step 的运行流程，里面会有一些示例性的运行脚本，具体细节请参看 github 仓库中的 README 和官方文档。

## Introduction

在 `Icefall` 项目中，我们使用了两种类型的 `Encoder` 来实现流式的 `Xformer-transducer` 模型，分别是 [Conformer](https://arxiv.org/pdf/2005.08100.pdf) 和 [Emformer](https://arxiv.org/pdf/2010.10759.pdf). `Conformer` 模型本身不支持流式，但我们可以使用 Mask 让模型只看到有限的上下文，实现 Chunk by Chunk 的计算。`Emformer` 模型是 Facebook 团队提出的，本质上还是一个使用 Mask 实现 Chunk by Chunk 计算的 transformer 类模型，不过增加了一些机制以提升流式识别的准确率。
### Conformer
`Icefall` 中实现流式 `Conformer` 模型的方法基本和 [Wenet](https://arxiv.org/pdf/2012.05481.pdf)中的一致，使用 Unified non-streaming and streaming 的方法训练，即一部分的数据使用 Full Attention，一部分的数据使用  Dynamic Chunk Attention。 不过，为了使模型更适宜 batch 推理，我们总是使用有限的 `left_context` 进行训练。

![Wenet 中的 Attention 示意图](https://cdn.jsdelivr.net/gh/filess/img10@main/2022/08/19/1660896744717-82b35dd2-765a-4012-aa5a-72dd3f23cdb1.png)

### Emformer
`Emformer` 主要在 Transformer 基础上做了两点比较大的改动，一个是引入了 **Memory Bank** 的机制让模型能看到更多的历史信息，有点像 LSTM 中的 Cell State。另一个是在训练中引入了 **Right Context**，为实现 Right Context 它使用了一种 Hard Copy 的方式在特征层面为每个 Chunk 配上对应的 Right Context 输入。整体的模型结构如下图所示，Chunk 中增加了 Memory Bank 和 Right Context ~~【虽然知道单凭这个图你们肯定看不明白，但小编还是得放呀】~~。

![Emformer 模型结构](https://cdn.jsdelivr.net/gh/filess/img9@main/2022/08/19/1660896744713-6236b4b8-dc50-4981-93c6-ad337af1b776.png)

在 Icefall 中我们对 Emformer 做了些小改动，以提升其训练速度和准确率，代码见 `egs/librispeech/ASR/conv_emformer_transducer_stateless2`。
1. 将模型里面的组件都换成了 [Reworwed Model](https://mp.weixin.qq.com/s/2WrEh3wHzYE6TCKuw_laLw) 里面的 Scaled 类型的模块。
2. 参照 Conformer 在各层添加了 Convolution 模块。
3. 简化了 Momery Bank 的计算。

总的来说，Emformer 模型是比较复杂的，训练速度也较慢一些， 当然，也有小伙伴反馈 Emformer 模型效果比 Streaming Conformer 要好。【想了解更详细的 Emformer 模型实现，点赞关注转发，也许小编下一期就安排上了】

> 我们致力于整合 Conformer 和 Emformer 模型的优点，目前来看可能会造成性能差异的两者的不同点是:  
1）Conformer 模型中包含 **Relative Position Encoding** 模块而 Emformer 中没有 【攻城狮内心OS：实在是加不上呀】。  
2）Emformer 在训练中可以看到  **Right Context** ， 而 Conformer 不行。
 
## Training
> 小编注：数据准备相关的工作对流式和非流式模型没有区别，本文不会涉及，我们假设读者已经成功运行了数据集下面的 `prepare.sh` 脚本。

目前 `Icefall` 中 LibriSpeech 数据集的 `pruned_transducer_statelessX ` 模型大多已经支持了流式模型的训练，跟训练非流式模型相比，只需要增加几个训练参数就可以做到。

1. `dynamic-chunk-training=True`, 训练流式或非流式模型的开关。
2. `short-chunk-size=20`, Unified 方法训练中有一部分使用 Full Attention 训练，一部分使用 Chunk Attention 训练，这个参数是 Chunk Attention 训练中最大的 chunk size。为控制时延，我们一般不会用太大的 chunk size 解码，所以这个值取 20 左右就行。
3. `num-left-chunks=4`, 在用 Chunk Attention 训练时当前 Chunk 能看到的左上下文的 Chunk 数 （注意：chunk size 是动态变化的），设置为 -1 表示看所有历史信息，但为了更好支持 batch 推理，一般设置一个固定的较小值。
4. `causal-convolution=True`, 使用流式模型时，每一层的卷积必须是 causal 类型的。

训练的命令行大概长这样子：
```
./pruned_transducer_stateless4/train.py \
  --exp-dir pruned_transducer_stateless4/exp \
  --full-libri 1 \
  --dynamic-chunk-training 1 \
  --causal-convolution 1 \
  --short-chunk-size 20 \
  --num-left-chunks 4 \
  --max-duration 300 \
  --world-size 4 \
  --start-epoch 1 \
  --num-epochs 25
```

**注意：Conformer 里面的 `--short-chunk-size` 是 Subsampling 之后的帧数。**

各个 recipe 的更多训练细节详见 [icefall](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md#librispeech-bpe-training-results-pruned-stateless-streaming-conformer-rnn-t).

Emformer 模型本身就是为流式而设计的，所以训练时没有流式和非流式不同的困扰，训练的命令行大概长这样子：

```
./conv_emformer_transducer_stateless2/train.py \
  --exp-dir conv_emformer_transducer_stateless2/exp \
  --full-libri 1 \
  --num-encoder-layers 12 \
  --chunk-length 32 \
  --cnn-module-kernel 31 \
  --left-context-length 32 \
  --right-context-length 8 \
  --memory-size 32 \
  --max-duration 280 \
  --world-size 6 \
  --num-epochs 30 \
  --start-epoch 1
```

**注意： Emformer 里面的 `--chunk-length`  `--left-context-length`  `--right-context-length` 都是 subsampling 之前的帧数.**

更多训练细节请参考 [Conv-Emformer](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md#librispeech-bpe-training-results-pruned-stateless-conv-emformer-rnn-t) 和 [Conv-Emformer2](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md#librispeech-bpe-training-results-pruned-stateless-conv-emformer-rnn-t-2)

## Decoding

对于流式的模型，Icefall 中支持两种形式的解码方法，即，simulate-streaming 和 real-streaming。Simulate streaming 解码采用的是和训练一样的 Mask 策略，让模型解码时只看到一部分的上下文信息。Simulate streaming 的解码实现位于各 Recipe 下面的 `decode.py` 里面。对于 Conformer 模型，只需将 `--simulate-streaming `设置为 `True`，并配上相应的 chunk size 就行。

```
./pruned_transducer_stateless4/decode.py \
    --simulate-streaming 1 \
    --decode-chunk-size 8 \
    --left-context 32 \
    --causal-convolution 1 \
    --epoch 25 \
    --avg 3 \
    --exp-dir ./pruned_transducer_stateless4/exp \
    --max-duration 1000 \
    --decoding-method greedy_search  # modified_beam_search; fast_beam_search
```

**注意：`--decode-chunk-size` 和 `--left-context` 都是 subsampling 之后的帧数。**

Real-streaming 解码方式则是完全流式的，即，Attention 是 Chunk by Chunk 计算的。Icefall 中我们实现了异步并行的解码，所谓异步并行即在一个 batch 里面各条音频的帧的位置都可以是不同的，比如，有些音频刚刚开始解码，进入该 batch 的可能是 第0 ~ 8 帧，有些音频已经快解完了，进入该 batch 的可能就是最后几帧。real-streaming 解码的实现在各 Recipe 下面的 `streaming_decode.py` 里面， 解码的命令跟 simulate streaming 类似，只是少了 `--simulate-streaming` 参数，并且 `--max-duration` 换成了 `--num-decode-streams`，即几条音频同时解码。
```
./pruned_transducer_stateless4/streaming_decode.py \
    --decode-chunk-size 8 \
    --left-context 32 \
    --causal-convolution 1 \
    --epoch 25 \
    --avg 3 \
    --exp-dir ./pruned_transducer_stateless4/exp \
    --num-decode-streams 1000 \
    --decoding-method greedy_search  # modified_beam_search; fast_beam_search
```
**注意：`--decode-chunk-size` 和 `--left-context` 都是 subsampling 之后的帧数。**

Conformer 模型的 `streaming_decode.py` 里还有一个 `--right-context` 参数，我们尝试在解码时引入更多的上下文信息，实现方法非常简单，即让相邻的两个 chunk 有些重叠。但这个策略并不是对所有的模型和解码参数有效，可能的原因是 right context 的引入导致了训练解码不一致问题。

Emformer 的解码也包含 simulate streaming 和 real streaming 方式，实现在相应的 `decode.py` 和 `streaming_decode.py` 文件里，只是参数名称略有不同。

## Deploy
如果想把训练好的模型部署成服务，你需要做两件事：
1. 在 Icefall 中用 torch.jit.script 导出模型。
2. 在 sherpa 中启动服务。

模型的导出脚本在各 Recipe 下面的 `export.py` 文件里，对于 Conformer 模型，导出命令长这样：
```
./pruned_transducer_stateless5/export.py \
    --streaming-model 1 \
    --causal-convolution 1 \
    --epoch 25 \
    --avg 5 \
    --exp-dir ./pruned_transducer_stateless5/exp \
    --jit 1
```
**注意：sherpa 只支持用 torch.jit.script 导出的模型，所以 `--jit` 要设置为 `True`。另外，如果你要导出 流式的 Conformer 模型，必须设置 `--streaming-model` 为 `True`， 并且 `--causal-convolution` 也为 `True`。**

Emformer 模型的导出也类似，同样需要设置 `--jit` 为 `True`。不过没有了 streaming 和非 streaming 的区别。

模型导出后在 sherpa 上的部署也非常简单，sherpa 中支持 Icefall 中的 transducer 类模型，你只需要指定模型和 BPE 的路径，就可以启动相应的服务。
```
./sherpa/bin/streaming_pruned_transducer_statelessX/streaming_server.py \
  --port 6006 \
  --nn-model-filename path/to/your/model.pt \
  --bpe-model-filename path/to/your/bpe.model
```

Emformer 模型的部署也同样简单，找到 sherpa 中对应的文件夹就行：
```
./sherpa/bin/conv_emformer_transducer_stateless2/streaming_server.py \
  --port 6006 \
  --nn-model-filename path/to/your/model.pt \
  --bpe-model-filename path/to/your/bpe.model
```

你可以使用自己训练的模型，也可以使用我们提供的模型，各 Recipe 的模型可以在[README.md](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md)找到链接。

## Adapt to streaming
目前 Icefall 中 Conformer 类 Transducer 模型，只有 LibriSpeech 数据集中的 `pruned_transducer_stateless{,2, 3, 4, 5}` 和 WenetSpeech 中得 `pruned_transducer_stateless5` 支持流式训练。欢迎大家向其他数据集增加流式模型的支持，并跑出 SOTA 的结果。在非流式 Conformer Transducer 模型中支持流式只需要做几件事：
1. 增加 Chunk Attention 所需要的 Mask。
2. 将各层卷积改成 causal 类型。
3. 支持流式解码。

你所要做的所有修改都在这个 [PR](https://github.com/k2-fsa/icefall/pull/454) 里。

Emformer 类模型，也只有 LibriSpeech 数据集中的 `pruned_stateless_emformer_rnnt2`，`conv_emformer_transducer_stateless`，`conv_emformer_transducer_stateless2`。如果你想将 Emformer 模型用到其他数据集，我们推荐你使用 `conv_emformer_transducer_stateless2`。

欢迎大家踊跃参与，贡献完整 recipe 者我们将赠送官方的文化衫。
