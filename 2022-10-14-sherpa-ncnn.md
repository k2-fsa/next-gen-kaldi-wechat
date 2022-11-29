# sherpa + ncnn 进行语音识别

> 本文介绍据我们所知, 第一个使用 ``ncnn`` 进行语音识别的开源项目
> ``sherpa-ncnn``。

## 简介

[sherpa][sherpa] 目前支持使用 ``PyTorch`` 做推理框架，进行语音识别。
当模型使用 ``PyTorch`` 训练好之后，我们可以把模型导出成 ``torchscript``
支持的格式，脱离 ``Python``, 使用 ``C++`` 在 ``sherpa`` 中进行部署。

``PyTorch`` 对 ``CPU`` 和 ``GPU`` 都有良好的支持，适合在基于 ``x86`` 架
构的服务器上使用。可是，``PyTorch`` 是一个重量级的框架，对资源的使用
要求较高，对嵌入式的支持也不是那么的友好。某些情况下，我们也希望构建一个
没有依赖或者引入很轻量级依赖的语音识别应用。这时，我们就需要寻找 ``PyTorch``
以外的推理框架。

由于模型是使用 ``PyTorch`` 训练的，如果我们使用其他的框架，首先要解决的
问题，就是模型格式的转换。``PyTorch`` 对 [onnx][onnx] 提供了内置的
支持，因此支持 ``onnx`` 格式的框架是我们的首选。

但是，并不是所有的 ``PyTorch`` 算子，都支持使用 ``onnx`` 进行导出。因此在选用
推理框架的时候，我们还要考虑该框架是否容易扩展。经过综合考虑，我们决定先支持
 [ncnn][ncnn] 推理框架。一方面，它提供了 [PNNX][PNNX] 模型转换工具，
可以很方便的把 ``PyTorch`` 模型转为 ``ncnn`` 支持的格式；
``ncnn`` 和 ``PNNX`` 的代码可读性和可扩展性都很不错，当碰到不支持的算子
时，我们可以很方便的扩展 ``ncnn`` 和 ``PNNX``。另一方面，尽管 ``ncnn`` 开源
已经有长达 5 年的时间，它的开发者社区仍然很活跃，并且 up 主还在不断的更新和维护
``ncnn``; 当我们碰到问题的时候，可以很容易的获得帮助。


本文介绍如何使用 [sherpa-ncnn][sherpa-ncnn]
进行语音识别。我们目前支持和测试过的 平台有 ``Linux``, ``macOS``, ``Windows``，和
``Raspberry Pi`` 等。

下面的视频演示了如何使用 ``sherpa-ncnn`` 在 ``Windows`` 平台进行实时的语音识别。

> （1）你能找出那些（个）识别错误的字么？
>
> （2）你能想出通过什么方法来解决这个识别错误的问题么？
>
> （3） 你知道这个是谁的声音么？

https://user-images.githubusercontent.com/5284924/195835352-d0bfb6e6-fb71-4bf8-a6ac-c6102bdd0aa3.mp4


## 模型转换

我们以 [LSTM transducer][LSTM transducer] 为例，描述如何把一个 [icefall][icefall]
中训练好的模型从 ``PyTorch`` 转换为 ``ncnn`` 支持的格式。

由于我们使用了带 ``projection`` 的 ``LSTM`` 模型，目前 ``PyTorch`` 还不支持
通过 ``onnx`` 的方式导出这类模型。 这意味着直接通过先转成 ``onnx`` 格式，
再把 ``onnx`` 格式的模型转成 ``ncnn`` 支持的格式这条路，就行不通了。

剩下的选择就是通过 ``PNNX`` 进行模型转换。 不幸的是 ``ncnn`` 和 ``PNNX``
直到今天 （2022.10.14）才支持带 ``projection`` 的 ``LSTM``。幸运的是，
``ncnn`` 和 ``PNNX`` 扩展性能很好，添加自定义算子的步骤简便、快捷，
为此我们修改了 ``ncnn`` 和 ``PNNX``，增加了对带 ``projection`` 的 ``LSTM``
的支持。

> 在我们的要求下，``ncnn``  的开发者增加了上述功能。
> 后续我们会使用 ``ncnn`` 和 ``PNNX`` 内置的实现。

解决了不支持的算子问题后，接下来就是正式的转模型了。目前 ``PNNX`` 只支持
``torch.jit.trace()`` 导出的模型。因此，我们首先需要把模型通过 ``torch.jit.trace()``
进行导出。所需命令如下:

```bash
cd egs/librispeech/ASR

iter=468000
avg=16

./lstm_transducer_stateless2/export.py \
  --exp-dir ./lstm_transducer_stateless2/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --iter $iter \
  --avg  $avg \
  --pnnx 1
```

> 注：``egs/librispeech/ASR`` 是 [icefall][icefall] 中的路径。
> 上述命令假设你已经使用 ``PyTorch`` 完成了模型训练。

上述命令会生成以下 3 个文件：

  - ``./lstm_transducer_stateless2/exp/encoder_jit_trace-pnnx.pt``
  - ``./lstm_transducer_stateless2/exp/decoder_jit_trace-pnnx.pt``
  - ``./lstm_transducer_stateless2/exp/joiner_jit_trace-pnnx.pt``

剩下的就是使用 ``PNNX`` 对 ``torch.jit.trace()`` 导出的文件进行转换。

下述命令展示了如何安装我们修改过后的 ``ncnn`` 和 ``PNNX``。

```bash
git clone https://github.com/csukuangfj/ncnn
cd ncnn
git submodule update --recursive --init
python3 setup.py bdist_wheel
ls -lh dist/
pip install ./dist/*.whl

# now build pnnx
cd tools/pnnx
mkdir build
cd build
make -j4
export PATH=$PWD/src:$PATH

./src/pnnx # view help message
```

> 注: ``export PATH=$PWD/src:$PATH`` 把可执行程序 ``pnnx`` 所在的路径加入
> 到环境变量 ``PATH``。

安装好 ``ncnn`` 和 ``PNNX`` 之后，我们就可以把 ``torch.jit.trace()`` 导出
的模型通过下面的命令转换成 ``ncnn`` 支持的格式：

```bash
cd egs/librispeech/ASR

pnnx ./lstm_transducer_stateless2/exp/encoder_jit_trace-pnnx.pt
pnnx ./lstm_transducer_stateless2/exp/decoder_jit_trace-pnnx.pt
pnnx ./lstm_transducer_stateless2/exp/joiner_jit_trace-pnnx.pt
```

它会生成下面几个文件：

- ``./lstm_transducer_stateless2/exp/encoder_jit_trace-pnnx.ncnn.param``
- ``./lstm_transducer_stateless2/exp/encoder_jit_trace-pnnx.ncnn.bin``
- ``./lstm_transducer_stateless2/exp/decoder_jit_trace-pnnx.ncnn.param``
- ``./lstm_transducer_stateless2/exp/decoder_jit_trace-pnnx.ncnn.bin``
- ``./lstm_transducer_stateless2/exp/joiner_jit_trace-pnnx.ncnn.param``
- ``./lstm_transducer_stateless2/exp/joiner_jit_trace-pnnx.ncnn.bin``


至此，我们完成了模型从 ``PyTorch`` 到 ``ncnn``  的转换。

## 验证

模型转换完之后，一个重要的步骤就是验证模型的正确性。一种方法是给定同样的输入，
逐层比对 ``ncnn`` 的输出是否和 ``PyTorch`` 的输出一致。这种方法耗时耗力，
一般只有在排查问题时，没有办法中的办法。

我们采用另外一种端到端的验证方法：看使用转换后的模型能否正确识别
给定的音频。幸运的是 ``ncnn`` 也提供了 ``Python`` 的接口，我们可以直接
在 ``Python`` 中验证模型的正确性。

我们提供两种模式的验证：

- （1）离线识别验证
- （2）流式识别验证

> 注：我们建议，在进行模型转换的时候，采用逐层转换的方法。转换一层，验证
> 一层，并做好单元测试。

### 离线识别验证

命令如下：

```bash
cd egs/librispeech/ASR

./lstm_transducer_stateless2/ncnn-decode.py \
 --bpe-model-filename ./data/lang_bpe_500/bpe.model \
 --encoder-param-filename ./lstm_transducer_stateless2/exp/encoder_jit_trace-pnnx.ncnn.param \
 --encoder-bin-filename ./lstm_transducer_stateless2/exp/encoder_jit_trace-pnnx.ncnn.bin \
 --decoder-param-filename ./lstm_transducer_stateless2/exp/decoder_jit_trace-pnnx.ncnn.param \
 --decoder-bin-filename ./lstm_transducer_stateless2/exp/decoder_jit_trace-pnnx.ncnn.bin \
 --joiner-param-filename ./lstm_transducer_stateless2/exp/joiner_jit_trace-pnnx.ncnn.param \
 --joiner-bin-filename ./lstm_transducer_stateless2/exp/joiner_jit_trace-pnnx.ncnn.bin \
 /path/to/foo.wav
```

有关模型转换和验证的详细文档，可以参考如下链接：

https://k2-fsa.github.io/icefall/model-export/export-ncnn.html

### 流式识别验证

命令如下：

```bash
./lstm_transducer_stateless2/streaming-ncnn-decode.py \
 --bpe-model-filename ./data/lang_bpe_500/bpe.model \
 --encoder-param-filename ./lstm_transducer_stateless2/exp/encoder_jit_trace-pnnx.ncnn.param \
 --encoder-bin-filename ./lstm_transducer_stateless2/exp/encoder_jit_trace-pnnx.ncnn.bin \
 --decoder-param-filename ./lstm_transducer_stateless2/exp/decoder_jit_trace-pnnx.ncnn.param \
 --decoder-bin-filename ./lstm_transducer_stateless2/exp/decoder_jit_trace-pnnx.ncnn.bin \
 --joiner-param-filename ./lstm_transducer_stateless2/exp/joiner_jit_trace-pnnx.ncnn.param \
 --joiner-bin-filename ./lstm_transducer_stateless2/exp/joiner_jit_trace-pnnx.ncnn.bin \
 /path/to/foo.wav
```

## 使用

当验证转换后的模型没有问题时，我们就可以利用它进行语音识别了。这一步
主要的工作就是使用 ``ncnn`` 提供的 ``C++ API`` 进行神经网络的计算，
然后实现解码算法，进行解码。

所有代码都开源在如下 repo:

- https://github.com/k2-fsa/sherpa-ncnn

下面我们讲述如何编译 ``sherpa-ncnn`` 并用它进行语音识别 (包含非流式和流式）。

### Linux/macOS 平台编译方法

```bash
git clone https://github.com/k2-fsa/sherpa-ncnn
cd sherpa-ncnn
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j6
```

上述命令会生成两个可执行程序：

```bash
(py38) kuangfangjun:build$ ls -lh bin/
total 9.5M
-rwxr-xr-x 1 kuangfangjun root 4.8M Sep 21 20:17 sherpa-ncnn
-rwxr-xr-x 1 kuangfangjun root 4.8M Sep 21 20:17 sherpa-ncnn-microphone
```

- ``sherpa-ncnn`` 可以用于识别给定的音频文件 （非流式）
- ``sherpa-ncnn-microphone`` 用于实时语音识别

值得注意的是，这两个可执行程序，只依赖系统库，没有任何的外部依赖。验证方法如下：

```bash
(py38) kuangfangjun:build$ readelf -d bin/sherpa-ncnn-microphone | head -n 12

Dynamic section at offset 0x438858 contains 33 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]
 0x0000000000000001 (NEEDED)             Shared library: [libgomp.so.1]
 0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
 0x000000000000001d (RUNPATH)            Library runpath: [$ORIGIN]
 0x000000000000000c (INIT)               0x1d4b0
 0x000000000000000d (FINI)               0x3d0f94
```

### Windows 平台编译方法

```bash
git clone https://github.com/k2-fsa/sherpa-ncnn
cd sherpa-ncnn
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release -- -m:6
```

上述命令会生成两个可执行程序：

- ``./build/bin/Release/sherpa-ncnn.exe``
- ``./build/bin/Release/sherpa-ncnn-microphone.exe``

我们默认采用的是静态链接，这意味着上述两个 ``.exe`` 没有外部依赖。在一台
``Windows`` 电脑上编译完后，可以直接拷贝到其他的 ``Windows``  电脑运行。

### arm 平台编译

``sherpa-ncnn`` 还支持交叉编译。首先配置工具链：

```bash
mkdir /ceph-fj/fangjun/software
cd /ceph-fj/fangjun/software
tar xvf /path/to/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz

export PATH=/ceph-fj/fangjun/software/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin:$PATH
```

> 注：你可以选择适合你自己的工具链。
>
> ``gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz``
>
> 可以从
>
> https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads/8-3-2019-03
>
> 进行下载。
>
> 为了便于大家下载，我们在 ``huggingface`` 上做了一个镜像。链接如下
>
> https://huggingface.co/csukuangfj/arm-linux-gcc

安装完工具链后，可以采用如下命令编译 ``sherpa-ncnn``:

```bash
git clone https://github.com/k2-fsa/sherpa-ncnn
cd sherpa-ncnn

./build-arm-linux-gnueabihf.sh
```

上述命令生成如下两个文件

```bash
$ ls -lh build-arm-linux-gnueabihf/bin/
total 2.6M
-rwxr-xr-x 1 kuangfangjun root 1.3M Oct 14 23:00 sherpa-ncnn
-rwxr-xr-x 1 kuangfangjun root 1.4M Oct 14 23:00 sherpa-ncnn-microphone

$ file build-arm-linux-gnueabihf/bin/sherpa-ncnn
build-arm-linux-gnueabihf/bin/sherpa-ncnn: ELF 32-bit LSB executable, ARM, EABI5 version 1 (GNU/Linux), dynamically linked, interpreter /lib/ld-linux-armhf.so.3, for GNU/Linux 3.2.0, with debug_info, not stripped
```

### aarch64 平台编译

首先配置工具链：

```bash
wget https://releases.linaro.org/components/toolchain/binaries/latest-7/aarch64-linux-gnu/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu.tar.xz

tar xvf gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu.tar.xz -C /ceph-fj/fangjun/software

export PATH=/ceph-fj/fangjun/software/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin:$PATH
```

> 注：你可以选择适合你自己的工具链。

安装完工具链后，可以采用如下命令编译 ``sherpa-ncnn``:

```bash
git clone https://github.com/k2-fsa/sherpa-ncnn
cd sherpa-ncnn

./sherpa-ncnn/build-aarch64-linux-gnu.sh
```

上述命令生成如下两个文件

```bash
$ ls -lh build-aarch64-linux-gnu/bin/
total 4.4M
-rwxr-xr-x 1 kuangfangjun root 2.2M Oct 14 23:27 sherpa-ncnn
-rwxr-xr-x 1 kuangfangjun root 2.2M Oct 14 23:27 sherpa-ncnn-microphone

$ file build-aarch64-linux-gnu/bin/sherpa-ncnn
build-aarch64-linux-gnu/bin/sherpa-ncnn: ELF 64-bit LSB executable, ARM aarch64, version 1 (GNU/Linux), dynamically linked, interpreter /lib/ld-linux-aarch64.so.1, for GNU/Linux 3.7.0, BuildID[sha1]=30d340e9aff6feb75605548b4abb1ca89f9de093, with debug_info, not stripped
```

### 识别

编译好 ``sherpa-ncnn`` 之后，我们就可以用转换过的模型进行语音识别了。

> 注：目前 ``sherpa-ncnn`` 只实现了 ``greedy search`` 解码方法，不带任何形式的外部语言模型。

为了方便大家测试，我们针对英语和中文，分别提供了转换后的模型。模型下载方法如下：

- 下载英文模型

```bash
git lfs install
git clone https://huggingface.co/csukuangfj/sherpa-ncnn-2022-09-05
```

下载完后，我们可以得到下述文件：

```bash
./sherpa-ncnn-2022-09-05
|-- README.md
|-- bar
|   |-- decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin
|   |-- decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param
|   |-- encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin
|   |-- encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param
|   |-- joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin
|   `-- joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.param
|-- test_wavs
|   |-- 1089-134686-0001.wav
|   |-- 1221-135766-0001.wav
|   |-- 1221-135766-0002.wav
|   `-- trans.txt
`-- tokens.txt

2 directories, 12 files
```

文件大小信息如下：

```bash
$ ls -lh tokens.txt
-rw-r--r-- 1 kuangfangjun root 5.0K Sep  7 15:56 tokens.txt

$ ls -lh bar/
total 161M
-rw-r--r-- 1 kuangfangjun root 503K Sep  5 15:21 decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin
-rw-r--r-- 1 kuangfangjun root  437 Sep  5 15:21 decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param
-rw-r--r-- 1 kuangfangjun root 159M Sep  5 15:21 encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin
-rw-r--r-- 1 kuangfangjun root  21K Sep  5 15:21 encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param
-rw-r--r-- 1 kuangfangjun root 1.5M Sep  5 15:21 joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin
-rw-r--r-- 1 kuangfangjun root  488 Sep  5 15:21 joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.param
```

> 注：上述模型文件，目前还没有使用任何的量化操作。里面存储的都是 ``float32``
> 类型的参数

- 下载中文模型

```bash
git lfs install
git clone https://huggingface.co/csukuangfj/sherpa-ncnn-2022-09-30
```

下载完后，我们可以得到下述文件：

```bash
sherpa-ncnn-2022-09-30
|-- README.md
|-- decoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin
|-- decoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.param
|-- encoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin
|-- encoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.param
|-- joiner_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin
|-- joiner_jit_trace-epoch-11-avg-2-pnnx.ncnn.param
|-- test_wavs
|   |-- 0.wav
|   |-- 1.wav
|   |-- 2.wav
|   `-- RESULTS.md
`-- tokens.txt

1 directory, 12 files
```

文件大小信息如下：

```bash
$ ls -lh tokens.txt *ncnn*
-rw-r--r-- 1 kuangfangjun root 5.5M Sep 30 17:26 decoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin
-rw-r--r-- 1 kuangfangjun root  439 Sep 30 17:26 decoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.param
-rw-r--r-- 1 kuangfangjun root 159M Sep 30 17:25 encoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin
-rw-r--r-- 1 kuangfangjun root  21K Sep 30 17:25 encoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.param
-rw-r--r-- 1 kuangfangjun root 6.5M Sep 30 17:28 joiner_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin
-rw-r--r-- 1 kuangfangjun root  490 Sep 30 17:28 joiner_jit_trace-epoch-11-avg-2-pnnx.ncnn.param
-rw-r--r-- 1 kuangfangjun root  48K Sep 30 17:24 tokens.txt
```

> 注：上述模型文件，目前还没有使用任何的量化操作。里面存储的都是 ``float32``
> 类型的参数


#### 使用英文模型进行识别

**识别一个文件**

```bash
./build/bin/sherpa-ncnn \
  ./sherpa-ncnn-2022-09-05/tokens.txt \
  ./sherpa-ncnn-2022-09-05/bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
  ./sherpa-ncnn-2022-09-05/bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
  ./sherpa-ncnn-2022-09-05/bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
  ./sherpa-ncnn-2022-09-05/bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
  ./sherpa-ncnn-2022-09-05/bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
  ./sherpa-ncnn-2022-09-05/bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
  ./sherpa-ncnn-2022-09-05/test_wavs/1089-134686-0001.wav
```

> 注：如果你使用的是 ``Windows``, 请用 ``./build/bin/Release/sherpa-ncnn.exe``。
>
> 目前只支持单通道、16 kHz 采样率、``.wav`` 格式的音频文件。

**利用麦克风进行实时识别**

```bash
./build/bin/sherpa-ncnn-microphone \
  ./sherpa-ncnn-2022-09-05/tokens.txt \
  ./sherpa-ncnn-2022-09-05/bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
  ./sherpa-ncnn-2022-09-05/bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
  ./sherpa-ncnn-2022-09-05/bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
  ./sherpa-ncnn-2022-09-05/bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
  ./sherpa-ncnn-2022-09-05/bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
  ./sherpa-ncnn-2022-09-05/bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin
```

> 注：如果你使用的是 ``Windows``, 请用 ``./build/bin/Release/sherpa-ncnn-microphone.exe``。

下面是一个 ``bilibili`` 的视频链接，详细演示了如何在 ``macOS`` 系统编译
``sherpa-ncnn`` 以及如何使用
``sherpa-ncnn`` 利用麦克风进行实时的语音识别：

https://www.bilibili.com/video/BV1TP411p7dh/


#### 使用中文模型进行识别

**识别一个文件**

```bash
./build/bin/sherpa-ncnn \
 ./sherpa-ncnn-2022-09-30/tokens.txt \
 ./sherpa-ncnn-2022-09-30/encoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.param \
 ./sherpa-ncnn-2022-09-30/encoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin \
 ./sherpa-ncnn-2022-09-30/decoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.param \
 ./sherpa-ncnn-2022-09-30/decoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin \
 ./sherpa-ncnn-2022-09-30/joiner_jit_trace-epoch-11-avg-2-pnnx.ncnn.param \
 ./sherpa-ncnn-2022-09-30/joiner_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin \
 ./sherpa-ncnn-2022-09-30/test_wavs/0.wav
```

> 注：如果你使用的是 ``Windows``, 请用 ``./build/bin/Release/sherpa-ncnn.exe``。
>
> 目前只支持单通道、16 kHz 采样率、``.wav`` 格式的音频文件。

**利用麦克风进行实时识别**

```bash
./build/bin/sherpa-ncnn-microphone \
 ./sherpa-ncnn-2022-09-30/tokens.txt \
 ./sherpa-ncnn-2022-09-30/encoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.param \
 ./sherpa-ncnn-2022-09-30/encoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin \
 ./sherpa-ncnn-2022-09-30/decoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.param \
 ./sherpa-ncnn-2022-09-30/decoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin \
 ./sherpa-ncnn-2022-09-30/joiner_jit_trace-epoch-11-avg-2-pnnx.ncnn.param \
 ./sherpa-ncnn-2022-09-30/joiner_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin
```

> 注：如果你使用的是 ``Windows``, 请用 ``./build/bin/Release/sherpa-ncnn-microphone.exe``。

下面是一个 ``bilibili`` 的视频链接，详细演示了如何使用
``sherpa-ncnn`` 在 ``Windows`` 平台利用麦克风进行实时的语音识别：

https://www.bilibili.com/video/BV1214y177vu/

> https://github.com/k2-fsa/sherpa-ncnn/blob/master/.github/workflows/arm-linux-gnueabihf.yaml
> 和
> https://github.com/k2-fsa/sherpa-ncnn/blob/master/.github/workflows/arm-linux-gnueabihf.yaml
> 演示了如何使用 ``qemu-arm`` 和 ``qemu-aarch64`` 运行 ``sherpa-ncnn``.

## 总结

本文详细介绍了如何使用 ``sherpa-ncnn`` 进行语音识别，涉及到模型转换、
模型验证、``sherpa-ncnn``  的安装及使用方法。

目前我们只提供了不带任何语言模型的解码方法，这直接导致了本文开头视频中的
``出生为人`` 识别成了 ``出声为人``。我们下一步工作需要实现结合外部语言模型
进行解码（主要是基于 n-gram LM）。

值得指出的是，``ncnn`` 只支持 ``batch size == 1``。我们正在支持其他的推理
框架，如 [mace][mace]。

> 对 ``Android`` 和 ``iOS`` 等平台的支持，我们需要来自社区的你进行支持。
> 如果你感兴趣，欢迎给 ``sherpa-ncnn`` 提 ``PR``，我们有新一代 ``Kaldi`` 周边的
> 文创产品赠送。


[sherpa]: http://github.com/k2-fsa/sherpa
[sherpa-ncnn]: https://github.com/k2-fsa/sherpa-ncnn
[ncnn]: https://github.com/Tencent/ncnn
[onnx]: https://github.com/onnx/onnx
[PNNX]: https://github.com/Tencent/ncnn/tree/master/tools/pnnx
[LSTM transducer]: https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/lstm_transducer_stateless2
[icefall]: https://github.com/k2-fsa/icefall/
[mace]: https://github.com/XiaoMi/mace
