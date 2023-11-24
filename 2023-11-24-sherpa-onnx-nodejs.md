# 新一代 Kaldi: 支持 JavaScript 啦!

## 简介

新一代 ``Kaldi`` 部署框架 [sherpa-onnx][sherpa-onnx] 支持的编程语言 ``API`` 大家庭中，
最近增加了一个新成员: [JavaScript](https://github.com/k2-fsa/sherpa-onnx/pull/438)。

为了方便大家查看，我们把目前所有支持的编程语言汇总成如下一张图。

![](https://github.com/k2-fsa/next-gen-kaldi-wechat/blob/master/pic/2023-11-24-api.jpg)

> 注： 这个家庭还在不断的扩充，总有一款适合你！

> 后续我们会增加 ``Dart``、``Rust``、``WebAssembly`` 等支持。
>
> 如果你想贡献其他的语言，欢迎参与。

增加了对 ``JavaScript`` 的支持，意味着我们可以在 ``JavaScript`` 中使用
``sherpa-onnx`` 提供的各种功能，比如

  - 语音识别 （支持流式和非流式)
    - 支持 [Zipformer][zipformer]
    - 支持 [Paraformer][paraformer]
    - 支持 [Whisper][whisper]
  - 语音合成
    - 支持 [VITS][vits]
  - VAD
    - 支持 [Silero VAD][silero-vad]

下图总结了所有的功能。

![](https://github.com/k2-fsa/next-gen-kaldi-wechat/blob/master/pic/2023-11-24-asr-tts-vad.jpg)

本文介绍如何在 ``JavaScript`` 中使用这些功能。

> 注：我们目前支持的是 ``NodeJS``。 还不支持在浏览器中运行 ``sherpa-onnx``。
> 请关注后续我们对 ``WebAssembly`` 的支持。

## 安装

我们已经把 ``sherpa-onnx`` 封装成 ``npm`` 包。如下图所示:

![](https://github.com/k2-fsa/next-gen-kaldi-wechat/blob/master/pic/2023-11-24-sherpa-onnx-npm.png)

你只需要下面一条语句，就可以安装 ``sherpa-onnx``:

```
npm install sherpa-onnx
```

本文使用 [naudiodon2](https://www.npmjs.com/package/naudiodon2) 读取麦克风。为了完成本文的测试，
你需要额外安装如下依赖：

```
npm install naudiodon2
```

下面的视频记录了完整的安装过程:

> 注: 请使用 node v13.14.0 版本进行测试。
>
> 当下面这个 ``issue`` 解决后，你可以使用新版本的 node.
> https://github.com/node-ffi-napi/node-ffi-napi/issues/97

(请看 https://mp.weixin.qq.com/s/aBENCGdnS7wBEKnbipcVjA)

## 使用

安装完 ``sherpa-onnx`` 后，我们就可以开始使用它啦！

下面我们介绍如何使用 ``sherpa-onnx`` 的 ``JavaScript`` ``API`` 进行
 - 语音端点检测 (``VAD``)
 - 语音识别
 - 语音合成

本文用到的所有示例代码，都可以从下面的链接中找到。

https://github.com/k2-fsa/sherpa-onnx/tree/master/nodejs-examples

![](https://github.com/k2-fsa/next-gen-kaldi-wechat/blob/master/pic/2023-11-24-examples.jpg)

本文的所有演示，都在 ``nodejs-examples`` 目录进行。

### 语音端点检测 (VAD)

本例子演示如何使用 ``Silero VAD`` 在 ``JavaScript`` 中进行语音端点检测。

我们使用如下文件进行测试:

https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-examples/test-vad-microphone.js

测试命令如下：

```
cd nodejs-examples
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
node ./test-vad-microphone.js
```

演示视频如下。

(请看 https://mp.weixin.qq.com/s/aBENCGdnS7wBEKnbipcVjA)


### 语音合成 (TTS)

我们使用下述文件进行测试

https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-examples/test-offline-tts-zh.js

为了给大家演示语音合成的 JavaScript API 有多么简单，我们把所有的代码截图如下：

![](https://github.com/k2-fsa/next-gen-kaldi-wechat/blob/master/pic/2023-11-24-tts-zh.png)

测试命令为：
```
cd nodejs-examples
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-zh-aishell3.tar.bz2
tar xvf vits-zh-aishell3.tar.bz2
node ./test-offline-tts-zh.js
```

生成的音频文件如下：

演示视频如下：

(请看 https://mp.weixin.qq.com/s/aBENCGdnS7wBEKnbipcVjA)


### 语音识别

我们给大家展示两个例子：

 - 流式 ``VAD`` + 非流式 ``Whisper`` 英文语音识别
 - 流式 ``Zipformer`` 中英文语音识别

更多的例子，请参考下面的文档

https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-examples/README.md


#### 流式 ``VAD`` + 非流式 ``Whisper`` 英文语音识别

我们使用 ``Whisper`` ``tiny.en`` 这个模型进行测试。测试文件为

https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-examples/test-vad-microphone-offline-whisper.js

测试命令为：

```
cd nodejs-examples
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
node ./test-vad-microphone-offline-whisper.js
```

演示视频如下：

(请看 https://mp.weixin.qq.com/s/aBENCGdnS7wBEKnbipcVjA)


#### 流式 ``Zipformer`` 中英文语音识别

我们使用一个中英文的流式 ``Zipformer`` 模型进行测试。

测试文件为

https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-examples/test-online-transducer-microphone.js

测试命令为:

```
cd nodejs-examples
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
node ./test-online-transducer-microphone.js
```

演示视频如下：

(请看 https://mp.weixin.qq.com/s/aBENCGdnS7wBEKnbipcVjA)


## 总结

本文向大家介绍了如何使用 ``sherpa-onnx`` 的 ``JavaScript`` ``API``
进行语音识别、语音合成和语音端点检测 (``VAD``)。所有的计算
都在本地进行，不需要访问网络。

如果你对新一代 ``Kaldi`` 感兴趣或者有任何的问题，请通过下面的二维码联系我们。

![](https://github.com/k2-fsa/next-gen-kaldi-wechat/blob/master/pic/our-qr-code.png)

[sherpa-onnx]: http://github.com/k2-fsa/sherpa-onnx
[silero-vad]: https://github.com/snakers4/silero-vad
[zipformer]: https://arxiv.org/abs/2310.11230
[vits]: https://github.com/jaywalnut310/vits
[paraformer]: https://github.com/alibaba-damo-academy/FunASR
[whisper]: https://github.com/openai/whisper
