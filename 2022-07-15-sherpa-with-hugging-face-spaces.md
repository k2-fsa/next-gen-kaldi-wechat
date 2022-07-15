# 极速上手新一代 Kaldi 服务端框架 sherpa

> 本文介绍如何**极速**（而非快速）上手新一代 Kaldi 服务端框架 sherpa。
>
> **无需安装或者下载**，你只需一个浏览器，便可以体验新一代 Kaldi 服务端框架
> sherpa 为你带来的 **自动语音识别**。
>
> 你可以在电脑上（Windows, Linux, 或者 macOS），手机上，iPad 上，或者在
> 其他可以运行浏览器的设备上，体验 sherpa。
>
> 阅读本文后，你将知道，如何快速体验 在 sherpa 中，使用来自 icefall 训练框架中的
> 预训练模型进行自动语音识别。
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

本文介绍如何在浏览器中使用 sherpa 进行语音识别; 不讨论如何通过 server/client 的方式，
使用 sherpa。


## 在浏览器中体验 sherpa

我们已经把 sherpa 集成到了 hugging face spaces。总入口在

https://hf.co/k2-fsa

或者

https://huggingface.co/k2-fsa


目前，我们支持 [icefall](https://github.com/k2-fsa/icefall) 中使用如下数据集训练出来的模型:

- [WenetSpeech](https://github.com/wenet-e2e/WenetSpeech)， 中文
- [GigaSpeech](https://github.com/SpeechColab/GigaSpeech)， 英文
- [LibriSpeech](https://www.openslr.org/12)， 英文
- [TAL_CSASR](https://ai.100tal.com/dataset)，中英混合


各位读者可以使用以下两种方式在浏览器中进行体验:

- 上传一个音频，然后进行语音识别
- 在浏览器中进行录音，然后进行识别

### 上传一个音频进行识别

- 使用 WenetSpeech 数据集的预训练模型

点击如下链接：

https://huggingface.co/spaces/k2-fsa/icefall-asr-wenetspeech-pruned-transducer-stateless2

出现如下页面：

![](https://github.com/k2-fsa/next-gen-kaldi-wechat/raw/master/pic/2022-07-15-wenetspeech-1.png)

读者可以上传一段中文语音进行识别。识别效果如下图所示：

![](https://github.com/k2-fsa/next-gen-kaldi-wechat/raw/master/pic/2022-07-15-wenetspeech-2.png)

> 支持常见的音频格式，如 .wav, .mp3, .opus, .aac 等。如果所给音频的采样率不是 16k，我们会自动
> 对采样率进行转换。

> 上传的文件，请不要在文件名中使用特殊字符，如空格或者小括号 `()`等。

读者可以用上面类似的方法，对其他数据集预训练模型进行测试。例如：

- 使用 TAL_CSASR  数据集的预训练模型

点击如下链接：

https://huggingface.co/spaces/k2-fsa/icefall-asr-tal_csasr-pruned-transducer-stateless5

出现如下页面：

![](https://github.com/k2-fsa/next-gen-kaldi-wechat/blob/master/pic/2022-07-15-tal_csasr-1.png)

读者可以上传一段中文语音进行识别。识别效果如下图所示：

![](https://github.com/k2-fsa/next-gen-kaldi-wechat/blob/master/pic/2022-07-15-tal_csasr-2.png)


类似的，读者可以在 https://hf.co/k2-fsa 网页找到其他数据集的预训练模型进行体验。此处
不再赘述。

### 从浏览器中进行录音然后识别

- 使用 WenetSpeech 数据集的预训练模型

点击如下链接：

https://huggingface.co/spaces/k2-fsa/icefall-asr-wenetspeech-pruned-transducer-stateless2-from-recordings

出现如下页面：

![](https://github.com/k2-fsa/next-gen-kaldi-wechat/raw/master/pic/2022-07-15-wenetspeech-3.png)

使用 microphone 进行录音后，点击 `submit` 按钮。出现如下结果：

![](https://github.com/k2-fsa/next-gen-kaldi-wechat/raw/master/pic/2022-07-15-wenetspeech-4.png)

类似的，读者可以在 https://hf.co/k2-fsa 网页找到其他数据集的预训练模型进行体验。此处不再赘述。

> 读者可以从手机端打开上述链接进行测试。

## 总结

本文介绍了如何从浏览器中体验 sherpa，进行自动语音识别，而不需要读者下载或者安装额外的文件。

各位读者后续可以关注 https://hf.co/k2-fsa 。我们会陆续支持更多数据集预训练模型。
