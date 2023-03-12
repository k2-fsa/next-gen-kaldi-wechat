# 新一代 Kaldi 中玩转 NeMo 预训练 CTC 模型

> 本文介绍如何使用新一代 `Kaldi` 部署来自 `NeMo` 中的预训练 `CTC` 模型。

## 简介

[`NeMo`](https://github.com/NVIDIA/NeMo) 是 `NVIDIA` 开源的一款基于 `PyTorch` 的框架，
为开发者提供构建先进的对话式 `AI` 模型，如自然语言处理、文本转语音和自动语音识别。

使用 `NeMo` 训练好一个自动语音识别的模型后，一般可以选用以下两种方式进行部署:

  (1) [`TensorRT`](https://github.com/NVIDIA/TensorRT)

  (2) [`Riva`](https://developer.nvidia.com/riva)

本文向大家介绍第三种方式：基于新一代 `Kaldi` 的自动语音识别部署
框架[`sherpa`](https://github.com/k2-fsa/sherpa)。

最近，`sherpa` 通过以下两个 `pull-request`：

  (1) https://github.com/k2-fsa/sherpa/pull/332

  (2) https://github.com/k2-fsa/sherpa/pull/335

支持了来自 `NeMo` 的预训练 `CTC` 模型 [`EncDecCTCModelBPE`](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/ctc_bpe_models.py#L34)
和 [`EncDecCTCModel`](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/ctc_models.py#L41)。

新一代 `Kaldi` 目前拥有以下 3 个针对自动语音识别的部署框架：

  (1) [sherpa-ncnn](https://github.com/k2-fsa/sherpa-ncnn)。 使用 [ncnn](https://github.com/tencent/ncnn)
  进行神经网络计算。虽然它也支持 Windows/Linux/macOS，但它特别适合移动端和嵌入式端。

  (2) [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)。使用[onnxruntime](https://github.com/microsoft/onnxruntime)
  进行神经网络计算。支持 Windows/Linux/macOS，也支持移动端和嵌入式端。

  (3) `sherpa`。使用 `PyTorch` 进行神经网络计算。
  支持 Linux/macOS/Windows，也支持 `NIVIDA GPU`。

这三个框架中，`sherpa-ncnn` 和 `sherpa-onnx` 目前只支持 `transducer` 模型；
而`sherpa` 同时支持 `transducer` 和 `CTC` 模型。

下面我们向大家介绍原理及使用方法。

# 原理

要支持一个基于 `CTC loss` 训练的语音识别模型，我们需要解决以下 3 个问题：

(1) 特征提取

(2) 神经网络计算

(3) `CTC` 解码

## 特征提取

不同的框架，一般采用不同的特征提取方式。`NeMo` 中，这一步骤被称为 `Preprocessor`。
我们需要重点关注以下四个方面：

- (a) 采样率
- (b) 输入 `audio samples` 的范围。比如，是 `[-1, 1]` 还是 `[-32768, 32767]`
- (c) 使用 `Fbank`、`MFCC`或是其他特征，以及计算特征的参数
- (d) 是否需要对特征进行归一化。比如，
`NeMo` 中可以使用 [`per_feature`](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/preprocessing/features.py#L62) 等方法对特征做归一化

特征提取这一块，应该是所有步骤中，最为繁琐也是最容易出错的一步。
幸运的是，`NeMo` 采用了和 `Kaldi` 相兼容的 `Fbank`
作为特征，我们只需要在  `sherpa` 中支持对特征进行归一化这一额外的操作即可。

## 神经网络计算

`sherpa` 使用 PyTorch 进行神经网络计算，我们需要把 `NeMo` 中的模型导出
成 `torchscript` 格式。 在`NeMo` 中, 只需要使用下面一行语句即可:

```
model.export("model.pt")
```

得到 `torchscript` 模型后，我们需要重点关注以下一点：

- `model.forward` 方法的调用形式: 输入参数和返回值，分别是什么。比如，返回值
是 `nn.Linear`, `nn.Softmax`, 还是 `nn.LogSoftmax` 的输出。


## CTC 解码

`sherpa` 使用 [`k2`](https://github.com/k2-fsa/k2) 进行 `CTC` 解码，支持
CPU/GPU, 同时也支持以 batch 的方式进行并行解码。我们只需要得到神经网络模型的
`LogSoftmax`  的输出，即可使用 `sherpa` 中现有的解码代码。

这一步是所有步骤中，最简单的一步。


# 使用方法

下述文档

https://k2-fsa.github.io/sherpa/cpp/pretrained_models/offline_ctc/nemo.html

列出了目前我们已经转换过的模型。模型种类包括 `Citrinet` 和 `Conformer`；
支持的语言有英语、中文、德语等。

文档中也记载了如何把一个 `NeMo` 预训练模型导出成 `sherpa`  所支持的格式。

https://catalog.ngc.nvidia.com/orgs/nvidia/collections/nemo_asr

列出了 `NeMo` 所有的针对各种语言的语音识别预训练模型，比如法语和西班牙语等。

值的值出的是，文档中只演示了 [sherpa-offline](https://github.com/k2-fsa/sherpa/blob/master/sherpa/cpp_api/bin/offline-recognizer.cc) 的用法，你还可以使用 `sherpa` 提供的 WebSocket 功能
[offline-websocket-server](https://github.com/k2-fsa/sherpa/blob/master/sherpa/cpp_api/websocket/offline-websocket-server.cc)
和 [offline-websocket-client](https://github.com/k2-fsa/sherpa/blob/master/sherpa/cpp_api/websocket/offline-websocket-client.cc)
以及 Python API [offline_ctc_asr.py](https://github.com/k2-fsa/sherpa/blob/master/sherpa/bin/offline_ctc_asr.py)


# 总结

本文介绍了如何在新一代 `Kaldi` 的 `sherpa` 部署框架中使用来自 `NeMo` 的
预训练 `CTC` 模型。

`sherpa` 是一个极易扩展的框架，目前我们已经支持了来自
[icefall](https://github.com/k2-fsa/icefall)、[WeNet](https://github.com/wenet-e2e/wenet)、
[torchaudio](https://github.com/pytorch/audio)、`NeMo`
的 `CTC` 预训练模型。

如果你想支持其他的框架，比如 [ESPnet](https://github.com/espnet/espnet)
和 [SpeechBrain](https://github.com/speechbrain/speechbrain)，欢迎给我们提 `pull-request`
或者 `feature-request`.

如果你对新一代 `Kaldi` 有任何的问题（不局限于 `sherpa`），请通过以下任
一方式联系我们：

- 微信公众号：新一代 `Kaldi`
- 微信交流群: 新一代 `Kaldi` 交流群 （请关注公众号，添加工作人员微信，我们邀请您进群）
- QQ 交流群：744602236

