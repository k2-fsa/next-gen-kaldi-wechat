# 新一代 Kaldi 之实时语音识别 Python 包

## 简介

本文介绍一款基于`新一代 Kaldi` 的、超级容易安装的、`实时`语音识别
`Python` 包：[sherpa-ncnn][sherpa-ncnn]。

> 小编注： 它有可能是目前为止，`最容易` 安装的**实时**语音识
别 `Python` 包（`谁试谁知道`）。
> 它的使用方法也是极简单的。

## 安装

```
pip install sherpa-ncnn
```

对的，就是这一句，所有的依赖都从源码安装。

其实目前 `sherpa-ncnn` 只有下面 `3` 个依赖:

- [ncnn][ncnn] , 用于神经网络计算
- [kaldi-native-fbank][kaldi-native-fbank] , 用于计算 `fbank` 特征
- [pybind11][pybind11] , 用于 `C++` 和 `Python` 之间的交互

> 小编注：如果要使用麦克风，`sherpa-ncnn` 还依赖 `portaudio` （`C++`） 和
> `sounddevice` （`Python`）。

更多安装方法，请参考文档：

https://k2-fsa.github.io/sherpa/ncnn/python/index.html

## 演示

下面这个视频，展示了使用 `sherpa-ncnn` 的 `Python API` 进行实时语音识别的效果。
用到的代码，作为本文的附录，附于文末。


视频链接如下：

https://www.bilibili.com/video/BV1eK411y788/

> 小编注：如果你对 `endpoint detection` 感兴趣，请参考 `sherpa-ncnn` 的文档：
>
> https://k2-fsa.github.io/sherpa/ncnn/endpoint.html

> 小编注：视频中用到的模型也是开源的。请见
>
> https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/conv-emformer-transducer-models.html#csukuangfj-sherpa-ncnn-conv-emformer-transducer-2022-12-06-chinese-english



## 总结

本文介绍了可能是目前为止，最容易安装的 `实时语音识别` `Python` 包。

接下来的工作，是给识别结果加上时间戳。如果你对语音识别感兴趣，请给我们提
`pull-request`。

> 小编注：感谢
>
> https://github.com/pingfengluo
>
> 在
>
> https://github.com/k2-fsa/sherpa-ncnn/pull/42
>
> 中贡献了 `endpointing` 和 `modified_beam_search`。


## 附录

为了方便大家阅读，我们把

https://github.com/k2-fsa/sherpa-ncnn/tree/master/python-api-examples

中的 `speech-recognition-from-microphone-with-endpoint-detection.py`
做为附录，供大家阅读。

**使用 `sherpa-ncnn` 的 `Python API` 进行实时语音识别的代码如下。**

代码的详细解释，请参考文档：

https://k2-fsa.github.io/sherpa/ncnn/python/index.html#real-time-recognition-with-a-microphone


```python3
#!/usr/bin/env python3

# Real-time speech recognition from a microphone with sherpa-ncnn Python API
# with endpoint detection.
#
# Please refer to
# https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
# to download pre-trained models

import sys

try:
    import sounddevice as sd
except ImportError as e:
    print("Please install sounddevice first. You can use")
    print()
    print("  pip install sounddevice")
    print()
    print("to install it")
    sys.exit(-1)

import sherpa_ncnn


def create_recognizer():
    # Please replace the model files if needed.
    # See https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
    # for download links.
    recognizer = sherpa_ncnn.Recognizer(
        tokens="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/tokens.txt",
        encoder_param="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.param",
        encoder_bin="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.bin",
        decoder_param="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.param",
        decoder_bin="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.bin",
        joiner_param="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.param",
        joiner_bin="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.bin",
        num_threads=4,
        decoding_method="modified_beam_search",
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,
    )
    return recognizer


def main():
    print("Started! Please speak")
    recognizer = create_recognizer()
    sample_rate = recognizer.sample_rate
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms
    last_result = ""
    segment_id = 0
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            recognizer.accept_waveform(sample_rate, samples)

            is_endpoint = recognizer.is_endpoint

            result = recognizer.text
            if result and (last_result != result):
                last_result = result
                print(f"{segment_id}: {result}")

            if result and is_endpoint:
                segment_id += 1


if __name__ == "__main__":
    devices = sd.query_devices()
    print(devices)
    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
```

[sherpa-ncnn]: https://github.com/k2-fsa/sherpa-ncnn
[ncnn]: https://github.com/tencent/ncnn
[kaldi-native-fbank]: https://github.com/tencent/ncnn
[pybind11]: https://github.com/pybind/pybind11
