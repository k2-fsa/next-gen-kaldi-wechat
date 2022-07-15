# 极速上手新一代 Kaldi 服务端框架 sherpa

> 本文介绍如何**极速**（而非快速）上手新一代 Kaldi 服务端框架 sherpa。
>
> **无需安装或者下载**，你只需一个浏览器，便可以体验新一代 Kaldi 服务端框架
> sherpa 为你带来的 **自动语音识别**。
>
> 你可以在电脑上（Windows, Linux, 或者 macOS），手机上，iPad 上，或者在
> 其他可以运行浏览器的设备上，体验 sherpa。
>
> 阅读本文后，你将知道，如何快速体验使用新一代 Kaldi 训练框架 icefall 得到
> 的训练模型。
>
> 你只需一个浏览器，仅此而已。

## sherpa 简介

[sherpa](https://github.com/k2-fsa/sherpa) 是新一代 Kaldi 数个子项目中的一员。
它是一个服务端框架。其中，特征提取、神经网络计算以及解码这三部分，使用 C++
实现并提供 Python 接口；框架中的**所有其他**部分，如网络通信和线程池管理及调度等，
使用 Python 实现。

当一个线程从 Python 中调用 C++ 代码时，我们会释放
[GIL](https://docs.python.org/3/c-api/init.html#releasing-the-gil-from-extension-code)。
通过这种方式，我们可以在 Python 中使用多线程，充分利用多处理器带来的并行处理优势。

与 Kaldi 不同，sherpa 在设计之初，就充分考虑了它在 Python 中的易用性。即使用户
没有 C++ 相关的经验，也能迅速上手。

sherpa 既支持 server/client 的通信模式，也支持类似 Kaldi 中的 binary 调用，
从命令行给定模型及音频，输出文字。

本文介绍如何在浏览器中使用 sherpa 进行语音识别。各位读者可以使用以下两种方式:

- 上传一个音频进行语音识别
- 在浏览器中进行录音，然后进行识别

本文不讨论如何通过 server/client 的方式，使用 sherpa。

# 在浏览器中体验 sherpa

