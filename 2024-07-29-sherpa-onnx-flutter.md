# 语音实战之 Dart 和 Flutter

## 1. 简介

新一代 ``Kaldi`` 部署框架 [sherpa-onnx][sherpa-onnx] 支持的编程语言 ``API`` 大家庭中，
最近增加了一个新成员 [Dart][Dart]，使得支持的编程语言达到了 10 种。

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


![](https://github.com/k2-fsa/next-gen-kaldi-wechat/blob/master/pic/2024-07-29-dart-pub-dev.png)

发布的 Dart 包，支持如下平台：

  - Android (arm64-v8a, armv7-eabi, x86, x86_64)
  - iOS (arm64)
  - Linux (x64)
  - macOS (x64, arm64)
  - Windows (x64)

本文剩余部分通过手把手教的方式，介绍如何通过 ``Dart`` 和 ``Flutter``，
使用语音合成、语音识别、语音活动检测和声音事件检测。

## 2. 安装

我们提供了 ``sherpa_onnx`` [Dart 包](https://pub.dev/packages/sherpa_onnx)，
大家可以像使用其他 ``Dart`` 包一样，在 ``pubspec.yaml`` 文件中加入如下两行即可：

```
dependencies:
  sherpa_onnx: ^1.10.20
```

> 注： 推荐永远使用最新版本。

我们提供的 ``Dart`` 包里面包含了每个平台的预编译动态库，因此安装过程只涉及文件下载，
并不需要编译 ``sherpa-onnx`` 的 ``C++`` 源代码。

## 3. 语音合成

目前我们支持基于 [VITS][VITS] 的语音合成模型。 提供超过40种语言的语音合成，
比如中文、英文、``中英文``、德语等。

我们提供的语音合成模型，可以从下述[地址](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models)下载

```
https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
```

![](https://github.com/k2-fsa/next-gen-kaldi-wechat/blob/master/pic/2024-07-29-tts-models.png)

下面我们分别介绍使用纯 ``Dart`` API 的语音合成和基于 Flutter 的语音合成。

### 3.1 基于 Dart 的语音合成

基于纯 ``Dart`` 的语音合成 例子代码，可以从下述[网址](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/tts)找到。

```
https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/tts
```

我们使用一个英文的模型作为例子，演示如何运行:

```bash
git clone https://github.com/k2-fsa/sherpa-onnx
cd sherpa-onnx/dart-api-examples/tts

dart pub get

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2
tar xf vits-piper-en_US-libritts_r-medium.tar.bz2
rm vits-piper-en_US-libritts_r-medium.tar.bz2

dart run \
  ./bin/piper.dart \
  --model ./vits-piper-en_US-libritts_r-medium/en_US-libritts_r-medium.onnx \
  --tokens ./vits-piper-en_US-libritts_r-medium/tokens.txt \
  --data-dir ./vits-piper-en_US-libritts_r-medium/espeak-ng-data \
  --sid 109 \
  --speed 1.0 \
  --text 'liliana, the most beautiful and lovely assistant of our team!' \
  --output-wav en-109.wav

dart run \
  ./bin/piper.dart \
  --model ./vits-piper-en_US-libritts_r-medium/en_US-libritts_r-medium.onnx \
  --tokens ./vits-piper-en_US-libritts_r-medium/tokens.txt \
  --data-dir ./vits-piper-en_US-libritts_r-medium/espeak-ng-data \
  --sid 200 \
  --speed 1.0 \
  --text "That's one small step for a man, a giant leap for mankind." \
  --output-wav en-200.wav

dart run \
  ./bin/piper.dart \
  --model ./vits-piper-en_US-libritts_r-medium/en_US-libritts_r-medium.onnx \
  --tokens ./vits-piper-en_US-libritts_r-medium/tokens.txt \
  --data-dir ./vits-piper-en_US-libritts_r-medium/espeak-ng-data \
  --sid 500 \
  --speed 1.0 \
  --text "Ask not what your country can do for you; ask what you can do for your country." \
  --output-wav en-500.wav
```

生成的音频如下

 音频|文本|
|------|-----|
|https://github.com/user-attachments/assets/6db35e33-cb3c-423a-bc4f-82ec0d654ac3|liliana, the most beautiful and lovely assistant of our team! |
|https://github.com/user-attachments/assets/b574a8bb-e35d-44a7-bd81-6ac29ee10e25|That's one small step for a man, a giant leap for mankind. |
|https://github.com/user-attachments/assets/92cadd12-f0fe-4e91-86f3-9844b0704f04|Ask not what your country can do for you; ask what you can do for your country. |


### 3.2 基于 Flutter 的语音合成

基于 Flutter 的语音合成例子代码可以在下述[链接](https://github.com/k2-fsa/sherpa-onnx/tree/master/flutter-examples/tts)找到

```
https://github.com/k2-fsa/sherpa-onnx/tree/master/flutter-examples/tts
```

为了方便大家操作，我们特意制作了一个视频，从零开始，一步一步向大家演示如何
进行中英文的语音合成。视频如下:


> 注：你也可以前往 B 站观看上述视频。[地址](https://www.bilibili.com/video/BV1fgvyeqEMp/)为
> https://www.bilibili.com/video/BV1fgvyeqEMp/

> 注：视频里用的是 macOS，但基于 Flutter 的语音合成也能够在
> Android, iOS, Linux 和 Windows 等平台运行。

## 4. 语音识别及语音活动检测

我们提供的 ``Dart`` API, 既支持流式实时语音识别（即边说边识别），也支持
非流式语音识别（即说完后再识别）。

### 4.1 流式实时语音识别

实时语音识别的 Flutter 例子[代码链接](https://github.com/k2-fsa/sherpa-onnx/tree/master/flutter-examples/streaming_asr)如下

```
https://github.com/k2-fsa/sherpa-onnx/tree/master/flutter-examples/streaming_asr
```

我们特意分别在 Android, iOS 和 Windows 三个平台上录制了实时语音识别
的视频，向大家展示识别的效果。B站视频地址如下

|Android Flutter实时语音识别| ``https://www.bilibili.com/video/BV1Cb421E7NC/``|
|---|---|
|iOS Flutter实时语音识别|``https://www.bilibili.com/video/BV1p4421U7Gj/``|
|Windows Flutter 实时语音识别|``https://www.bilibili.com/video/BV1Nm421G7kN/``|


### 4.2 非流式语音识别

非流式语音识别的 Dart API 例子代码[地址](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/non-streaming-asr)如下
```
https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/non-streaming-asr
```

目前我们的 Dart API 支持基于下述模型的非流式语音识别:
  - [Zipformer][Zipformer]
  - [Whisper][Whisper]
  - [SenseVoice][SenseVoice]
  - [Paraformer][Paraformer]
  - [TeleSpeech-ASR][TeleSpeech-ASR]

下面我们以``Whisper`` 为例，介绍如何使用 Dart API 识别一个文件。

```bash
git clone https://github.com/k2-fsa/sherpa-onnx
cd sherpa-onnx/dart-api-examples/non-streaming-asr

dart pub get

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
rm sherpa-onnx-whisper-tiny.en.tar.bz2

dart run \
  ./bin/whisper.dart \
  --encoder ./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.int8.onnx \
  --decoder ./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.int8.onnx \
  --tokens ./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt \
  --input-wav ./sherpa-onnx-whisper-tiny.en/test_wavs/0.wav
```

上述命令的识别结果为
```
After early nightfall, the yellow lamps would light up here and there the squalid quarter of the brothels.
```

### 4.3 语音活动检测

我们支持在 ``Dart`` 里面使用 [silero-vad][silero-vad], 同时支持 silero-vad
v4 和最新版 v5.

例子代码[地址](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/vad)如下
```
https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/vad
```

下面的命令，我们使用 Dart API, 去除音频中非人声部分， 只保留人声。
```bash
git clone https://github.com/k2-fsa/sherpa-onnx
cd sherpa-onnx/dart-api-examples/vad

dart pub get

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav

dart run \
  ./bin/vad.dart \
  --silero-vad ./silero_vad.onnx \
  --input-wav ./lei-jun-test.wav \
  --output-wav ./lei-jun-test-no-silence.wav
```

上述命令把 ``lei-jun-test.wav`` 作为输入，去除非人声部分后，输出文件
``lei-jun-test-no-silence.wav``。

我们把两个音频文件放入如下表格，大家可以对比下结果。

|原始音频|处理后的音频|
|---|---|
|https://github.com/user-attachments/assets/bd85495d-b264-4074-8fc2-e2044d3aaaee|https://github.com/user-attachments/assets/15d7a5d7-2716-4632-b3ad-0e2d70cf3ae9|

### 4.4 语音活动检测和非流式语音识别相结合

一般的，非流式语音识别模型对输入的音频长度会有限制。比如， ``Whisper``
运行一次，只接收 ``30`` 秒的音频。

对于非常长的音频文件，我们可以用 VAD 根据音频中的停顿，对音频进行切割，
然后对切割后的音频，使用非流式模型进行语音识别。如果我们记住切割后的每一段
音频的起止时间，就可以生成字幕。

对此，我们特意提供了一个 VAD + 非流式语音识别的例子，代码[链接](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/vad-with-non-streaming-asr)如下:
```
https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/vad-with-non-streaming-asr
```

我们使用上面的测试音频 ``lei-jun-test.wav`` 和 ``SenseVoice`` 模型进行测试。
测试命令如下:


```bash
git clone https://github.com/k2-fsa/sherpa-onnx
cd sherpa-onnx/dart-api-examples/vad-with-non-streaming-asr

dart pub get
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

dart run \
  ./bin/sense-voice.dart \
  --silero-vad ./silero_vad.onnx \
  --model ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx \
  --tokens ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
  --use-itn true \
  --input-wav ./lei-jun-test.wav > ./lei-jun-test.txt
```

生成的 ``./lei-jun-test.txt`` 文件如下:

> 注: 运行时，我们使用了 ``SenseVoice`` 的 ``use_itn=true``。
> 因此识别结果中有标点，也对数字做了 ``inverse text normalzation``。

<details open>
<summary> ./lei-jun-test.txt </summary>

```bash
28.940 -- 36.006 : 朋友们晚上好，欢迎大家来参加今天晚上的活动，谢谢大家。
42.124 -- 46.374 : 这是我第四次颁年度演讲。
46.924 -- 50.118 : 前三次呢，因为疫情的原因。
50.412 -- 55.750 : 都在小米科技园内举办，现场的人很少。
56.140 -- 57.574 : 这是第四次。
58.188 -- 66.854 : 我们仔细想了想，我们还是想办一个比较大的聚会，然后呢让我们的新朋友老朋友一起聚一聚。
67.724 -- 70.886 : 今天的话呢我们就在北京的。
71.660 -- 75.142 : 国家会议中心呢举办了这么一个活动。
75.436 -- 79.526 : 现场呢来了很多人，大概有3500人。
79.948 -- 82.278 : 还有很多很多的朋友呢。
82.700 -- 85.798 : 通过观看直播的方式来参与。
86.348 -- 90.886 : 再一次呢对大家的参加表示感谢，谢谢大家。
98.476 -- 99.910 : 两个月前。
100.36 -- 104.49 : 我参加了今年武汉大学的毕业典礼。
105.93 -- 107.33 : 今年呢是。
107.92 -- 110.69 : 武汉大学建校130周年。
111.76 -- 112.84 : 作为校友。
113.36 -- 114.89 : 被母校邀请。
115.21 -- 117.22 : 在毕业典礼上致辞。
118.06 -- 119.56 : 这对我来说。
119.82 -- 122.60 : 是至高无上的荣誉。
123.66 -- 125.67 : 站在讲台的那一刻。
126.25 -- 128.61 : 面对全校师生。
129.20 -- 131.46 : 关于武大的所有的记忆。
131.69 -- 134.18 : 一下子涌现在脑海里。
134.99 -- 137.67 : 今天呢我就先和大家聊聊。
138.28 -- 139.49 : 大往事。
141.84 -- 143.81 : 那还是36年前。
145.93 -- 147.65 : 1987年。
148.68 -- 151.62 : 我呢考上了武汉大学的计算机系。
152.68 -- 155.17 : 在武汉大学的图书馆里。
155.40 -- 156.71 : 看了一本书。
157.58 -- 158.63 : 微捕之火。
159.34 -- 161.64 : 建立了我一生的梦想。
163.31 -- 164.45 : 看完书以后。
165.29 -- 166.44 : 热血沸腾。
167.60 -- 169.32 : 激动的睡不着觉。
170.41 -- 171.24 : 我还记得。
172.01 -- 172.97 : 那天晚上。
173.32 -- 174.66 : 星光很亮。
175.40 -- 177.67 : 我就在五大的操场上。
178.35 -- 179.78 : 就是屏幕上这个超场。
180.78 -- 182.47 : 走了一圈又一圈。
182.96 -- 185.22 : 走了整整一个晚上。
186.48 -- 187.75 : 我心里有团火。
188.94 -- 190.31 : 我也想搬一个。
190.60 -- 191.81 : 伟大的公司。
193.96 -- 194.82 : 就是这样。
197.61 -- 198.82 : 梦想之火。
199.28 -- 202.50 : 在我心里彻底点燃了。
209.77 -- 210.53 : 但是。
210.76 -- 212.55 : 一个大一的新生。
220.97 -- 222.73 : 一个大一的新生。
223.82 -- 227.05 : 一个从县城里出来的年轻人。
228.14 -- 230.63 : 什么也不会，什么也没有。
231.53 -- 236.33 : 就想创办一家伟大的公司，这不就是天方夜谭吗？
237.58 -- 240.10 : 这么离谱的一个梦想。
240.36 -- 242.28 : 该如何实现呢？
243.85 -- 244.93 : 那天晚上。
245.20 -- 246.92 : 我想了一整晚上。
247.98 -- 248.97 : 说实话。
250.35 -- 253.80 : 越想越糊涂，完全理不清头绪。
254.99 -- 256.10 : 后来我在想。
256.78 -- 258.02 : 干脆别想了。
258.35 -- 259.88 : 把书练好。
260.43 -- 261.38 : 是正慑。
262.16 -- 262.98 : 所以呢。
263.37 -- 265.67 : 我就下定决心认认真真读书。
268.49 -- 271.40 : 我怎么能够把书读的不同反响呢？
```
</details>

## 5. 声音事件检测

目前我们支持使用 ``Zipformer`` 和 [CED][CED] 进行声音事件检测 (audio tagging)。
比如，识别给定音频中是否有婴儿哭声，是否有猫叫声、是否有警报声等等。

例子代码[地址](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/audio-tagging)为
```
https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/audio-tagging
```

下面我们用 ``Zipformer`` 的 audio tagging 模型，演示如何使用。

```bash
git clone https://github.com/k2-fsa/sherpa-onnx
cd sherpa-onnx/dart-api-examples/audio-tagging

dart pub get

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
tar xvf sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
rm sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2

dart run \
  ./bin/zipformer.dart \
  --model ./sherpa-onnx-zipformer-audio-tagging-2024-04-09/model.int8.onnx \
  --labels ./sherpa-onnx-zipformer-audio-tagging-2024-04-09/class_labels_indices.csv \
  --wav ./sherpa-onnx-zipformer-audio-tagging-2024-04-09/test_wavs/5.wav
```

输出结果为
```
[AudioEvent(name: Slap, smack, index: 467, prob: 0.9186466336250305),
AudioEvent(name: Finger snapping, index: 62, prob: 0.9182724952697754),
AudioEvent(name: Whip, index: 472, prob: 0.18996259570121765),
AudioEvent(name: Clapping, index: 63, prob: 0.11238119006156921),
AudioEvent(name: Inside, small room, index: 506, prob: 0.02442842721939087)]
```

模型输出了概率最高的前5个时间。我们可以看到，判断为``打响指``的概率为 ``0.91``.

5.wav 测试音频如下:

https://github.com/user-attachments/assets/6cb77732-5bb7-4cf5-bda3-be82abbd4d1c


我们分别选取 6.wav 和 7.wav 进行测试，输出结果如下

6.wav

https://github.com/user-attachments/assets/ac2634df-195f-443d-bc35-3f0955d4aa52

```
[AudioEvent(name: Baby cry, infant cry, index: 23, prob: 0.8852226734161377),
AudioEvent(name: Crying, sobbing, index: 22, prob: 0.5102355480194092),
AudioEvent(name: Whimper, index: 24, prob: 0.0780656635761261),
AudioEvent(name: Inside, small room, index: 506, prob: 0.02283826470375061),
AudioEvent(name: Speech, index: 0, prob: 0.010384321212768555)]
```

判断有``婴儿哭声`` 的概率为  ``0.88``


7.wav

https://github.com/user-attachments/assets/f7ea88f6-fe5a-4100-b721-796741f49ec0

```
[AudioEvent(name: Smoke detector, smoke alarm, index: 399, prob: 0.3430711627006531),
AudioEvent(name: Sine wave, index: 501, prob: 0.30125075578689575),
AudioEvent(name: Beep, bleep, index: 481, prob: 0.24322959780693054),
AudioEvent(name: Buzzer, index: 398, prob: 0.1124192476272583),
AudioEvent(name: Fire alarm, index: 400, prob: 0.08048766851425171)]
```
判断为``烟雾报警``的概率为 ``0.34``.


## 总结

本文向大家介绍了如何使用 ``sherpa-onnx`` 的 ``Dart`` ``API``
进行语音识别、语音合成、语音活动检测和声音事件检测 。所有的计算
都在本地进行，不需要访问网络。

值得指出的是，虽然本文使用了 ``Dart`` 作为例子，但是大家可以选用我们支持的其他``9``种语言
中任意一种语言去开发本文提到的各种语音功能。

如果你对新一代 ``Kaldi`` 感兴趣或者有任何的问题，请通过下面的二维码联系我们。

![](https://github.com/k2-fsa/next-gen-kaldi-wechat/blob/master/pic/our-qr-code.png)


[sherpa-onnx]: http://github.com/k2-fsa/sherpa-onnx
[Dart]: https://dart.dev/
[Flutter]: https://flutter.dev/
[VITS]: https://arxiv.org/pdf/2106.06103
[Zipformer]: https://arxiv.org/abs/2310.11230
[Whisper]: https://github.com/openai/whisper
[SenseVoice]: https://github.com/FunAudioLLM/SenseVoice
[Paraformer]: https://github.com/modelscope/FunASR
[TeleSpeech-ASR]: https://github.com/Tele-AI/TeleSpeech-ASR
[silero-vad]: https://github.com/snakers4/silero-vad
[CED]: https://github.com/RicherMans/CED
