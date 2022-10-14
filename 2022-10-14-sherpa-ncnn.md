# sherpa + ncnn 进行语音识别

## 简介

[sherpa][sherpa] 目前支持使用 ``PyTorch`` 做推理框架，进行语音识别。
当模型使用 ``PyTorch`` 训练好之后，我们可以把模型导出成 ``torchscript``
支持的格式，脱离 ``Python``, 使用 ``C++`` 在 ``sherpa`` 中进行部署。

``PyTorch`` 对 ``CPU`` 和 ``GPU`` 都有良好的支持，适合在基于 ``x86`` 架
构的服务器上使用。另一方面，``PyTorch`` 是一个重量级的框架，对资源的使用
要求较高，对嵌入式的支持也不是那么的友好。某些情况下，我们也希望构建一个
没有依赖或者引入很轻量级依赖的语音识别应用。这时，我们就需要寻找 ``PyTorch``
以外的推理框架。

由于模型是使用 ``PyTorch`` 训练的，如果我们使用其他的框架，首先要解决的
问题，就是模型格式的转换问题。``PyTorch`` 对 [onnx][onnx] 提供了内置的
支持，因此支持 ``onnx`` 格式的框架是我们的首选。

但是，并不是所有的 ``PyTorch`` 算子，都支持使用 ``onnx`` 进行导出。因此在选用
推理框架的时候，我们还要考虑该框架是否容易扩展。经过综合考虑，我们决定先支持
 [ncnn][ncnn] 推理框架。一方面，它提供了 [PNNX][PNNX] 模型转换工具，
可以很方便的把 ``PyTorch`` 模型转为 ``ncnn`` 支持的格式；
``ncnn`` 和 ``PNNX`` 的代码可读性和可扩展性都很不错，当碰到不支持的算子
时，我们可以很方便的扩展 ``ncnn`` 和 ``PNNX``。另一方面，尽管 ``ncnn`` 开源
已经有长达 5 年的时间，它的开发者社区仍然很活跃，并且 up 主还在不断的更新和维护
``ncnn``; 当我们碰到问题的时候，可以很容易的获得帮助。


本文介绍如何使用 [sherpa-ncnn][sherpa-ncnn] 进行语音识别。目前支持和测试过的
平台有 ``Linux``, ``macOS``, ``Windows``，和 ``Raspberry Pi`` 等。

下面的视频演示了如何使用 ``sherpa-ncnn`` 进行实时的语音识别。

> 你能找出那些（个）识别错误的字么？ 你能想出通过什么方法来解决这个识别
> 错误的问题么？

https://user-images.githubusercontent.com/5284924/195835352-d0bfb6e6-fb71-4bf8-a6ac-c6102bdd0aa3.mp4


[sherpa]: http://github.com/k2-fsa/sherpa
[sherpa-ncnn]: https://github.com/k2-fsa/sherpa-ncnn
[ncnn]: https://github.com/Tencent/ncnn
[onnx]: https://github.com/onnx/onnx
[PNNX]: https://github.com/Tencent/ncnn/tree/master/tools/pnnx
