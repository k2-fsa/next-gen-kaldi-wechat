新一代 Kaldi 语音识别部署框架之 sherpa-onnx
===========================================

简介
----

今天我们向大家介绍继 [sherpa][sherpa] 和 [sherpa-ncnn][sherpa-ncnn] 之后
新一代 `Kaldi` 中第三个语音识别部署框架 [sherpa-onnx][sherpa-onnx].

这三个框架的最主要区别在于使用的神经网络推理库不同。具体用到的推理库，如下表所示。

| 框架名字 | 神经网络推理库|
|----|----|
|`sherpa`| [PyTorch](https://github.com/pytorch/pytorch)|
|`sherpa-ncnn`|[ncnn](https://github.com/tencent/ncnn)|
|`sherpa-onnx`|[onnxruntime](https://github.com/microsoft/onnxruntime)|

下表列出了 `sherpa-onnx` 所提供的具体功能：

|分类|具体内容|
|---|---|
|操作系统| Windows, Linux, macOS, Android, iOS |
|CPU| x86_64, aarch64|
|协议| WebSocket|
|API| C++, C, Python, Swift, Kotlin|
|移动端| iPhone, iPad, Android|
|嵌入式端|树莓派4 等|

本文的目的，是希望通过几个使用 `sherpa-onnx` 进行`实时语音识别`的视频, 向大家
演示 `sherpa-onnx` 的功能。`sherpa-onnx` 的详细使用文档，请见如下地址：

https://k2-fsa.github.io/sherpa/onnx/index.html

视频 1：使用 `sherpa-onnx` 在 iPhone 上进行实时的语音识别
---------------------------------------------------------

这里我们只提供 `sherpa-onnx` 在 `iPhone` 上的实时语音识别例子。如果你想在
`Android` 或 `iPad` 上进行实时的语音识别，请参考 `sherpa-onnx` 的文档。

https://www.bilibili.com/video/BV1gL411Z7dD/


视频 2：使用 `sherpa-onnx` 在 树莓派4 上进行实时的语音识别
--------------------------------------------------------

这里我们只提供 `sherpa-onnx` 在 树莓派上的实时语音识别例子。如果你想在
其他嵌入式板子上进行实时的语音识别，请参考 `sherpa-onnx` 的文档。

https://www.bilibili.com/video/BV15Y4y1m7NT/


视频 3：使用 `sherpa-onnx` 在 macOS 上进行实时的语音识别
--------------------------------------------------------

这里我们只提供 `sherpa-onnx` 在 `macOS` 上的实时语音识别例子。如果你想在
Linux 或者 Windows 上进行实时的语音识别，请参考 `sherpa-onnx` 的文档。

https://www.bilibili.com/video/BV1L24y1V7TT/

视频 4：使用 `sherpa-onnx` WebSocket服务进行实时的语音识别
----------------------------------------------------------

这个视频向大家演示 `sherpa-onnx` 的 WebSocket server 支持多客户端的例子。

https://www.bilibili.com/video/BV1BX4y1Q7jD/

总结
---

本文通过4个视频，向大家演示了 `sherpa-onnx` 的具体功能。
https://github.com/k2-fsa/sherpa-onnx/issues 列出了若干个我们还未完成的
事项。如果你碰巧也感兴趣，欢迎帮忙提交  `pull-request`。

如果你对新一代 `Kaldi` 有任何的问题（不局限于 `sherpa-onnx`），请通过以下任
一方式联系我们：

- 微信公众号：新一代 `Kaldi`
- 微信交流群: 新一代 `Kaldi` 交流群 （请关注公众号，添加工作人员微信，我们邀请您进群）
- QQ 交流群：744602236


[sherpa]: https://github.com/k2-fsa/sherpa
[sherpa-ncnn]: https://github.com/k2-fsa/sherpa-ncnn
[sherpa-onnx]: https://github.com/k2-fsa/sherpa-onnx
[ncnn]: https://github.com/tencent/ncnn
[onnxruntime]: https://github.com/microsoft/onnxruntime

