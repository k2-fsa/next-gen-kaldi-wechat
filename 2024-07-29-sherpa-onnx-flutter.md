# 语音实战之 Dart 和 Flutter

## 1. 简介

新一代 ``Kaldi`` 部署框架 [sherpa-onnx][sherpa-onnx] 支持的编程语言 ``API`` 大家庭中，
最近增加了一个新成员 [Dart][Dart]，使得支持的语言达到了 10 种。

支持了 ``Dart``, 意味着大家可以在 ``Dart`` 中使用 ``sherpa-onnx`` 提供的所有功能，即

  - 语音合成 (Text to speech)
  - 语音识别 (Speech recognition)
  - 语音活动检测 (Voice activity detection)
  - 声音事件检测 (Audio tagging)
  - 说话人识别和验证 (Speaker identification and verification)
  - 关键词检测 (keyword spotting)

下图总结了 ``sherpa-onnx`` 目前支持的编程语言及 ``Dart`` API 支持的功能。

![](https://github.com/k2-fsa/next-gen-kaldi-wechat/blob/master/pic/2024-07-29-dart-functions.png)

注: 上述图片的高清 pdf 文档，可以从下面[地址](https://github.com/k2-fsa/next-gen-kaldi-wechat/blob/master/pdf/sherpa-onnx-dart-functions.pdf)获得

```
https://github.com/k2-fsa/next-gen-kaldi-wechat/blob/master/pdf/sherpa-onnx-dart-functions.pdf
```

支持了 ``Dart``, 也意味着大家可以利用 [Flutter][Flutter] 开发跨平台的应用。对此，我们提供了
``sherpa_onnx`` 包，并发布在 https://pub.dev/ 。如下图所示。


![](https://github.com/k2-fsa/next-gen-kaldi-wechat/blob/flutter/pic/2024-07-29-dart-pub-dev.png)

发布的 Dart 包，支持如下平台：

  - Android (arm64-v8a, armv7-eabi, x86, x86_64)
  - iOS (arm64)
  - Linux (x64)
  - macOS (x64, arm64)
  - Windows (x64)

本文剩余部分通过手把手教大家的方式，介绍如何通过 ``Dart`` 和 ``Flutter``
使用本文第一张图片中的语音功能。

## 2. 安装

由于我们提供了 ``sherpa_onnx`` [Dart 包](https://pub.dev/packages/sherpa_onnx)，
大家可以像使用其他 ``Dart`` 包一样，在 ``pubspec.yaml`` 文件中加入如下两行即可：

```
dependencies:
  sherpa_onnx: ^1.10.20
```

> 注： 推荐永远使用最新版本。

我们提供的 ``Dart`` 包里面包含了每个平台的预编译动态库，因此安装过程只涉及文件下载，
并不需要编译 ``sherpa-onnx`` 的 ``C++`` 源代码。

## 3. 语音合成

## 4. 语音识别及语音活动检测

### 4.1 实时语音识别
### 4.2 非流式语音识别
### 4.3 语音活动检测
### 4.4 语音活动检测和非流式语音识别相结合

## 5. 说话人识别

## 6. 关键词检测

## 7. 声音事件检测

## 总结

本文向大家介绍了如何使用 ``sherpa-onnx`` 的 ``Dart`` ``API``
进行语音识别、语音合成、语音活动检测、说话人识别、声音时间检测和关键词检测 。所有的计算
都在本地进行，不需要访问网络。

值得指出的是，虽然本文使用了 ``Dart`` 作为例子，但是大家可以选用我们支持的其他``9``种语言
中任意一种语言去开发本文提到的各种语音功能。

如果你对新一代 ``Kaldi`` 感兴趣或者有任何的问题，请通过下面的二维码联系我们。

![](https://github.com/k2-fsa/next-gen-kaldi-wechat/blob/master/pic/our-qr-code.png)


[sherpa-onnx]: http://github.com/k2-fsa/sherpa-onnx
[Dart]: https://dart.dev/
[Flutter]: https://flutter.dev/
