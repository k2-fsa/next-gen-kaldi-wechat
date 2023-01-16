# 新一代 Kaldi - 嵌入式端实时语音识别

## 简介

本文向大家汇报我们最近的进展：

> 使用新一代 `Kaldi`， 在嵌入式环境中进行`实时`的语音识别。

我们用下面两块嵌入式开发板进行测试：

- 树莓派4 (Raspberry pi 4 model B)
- 爱芯派 (MAIX-III AXera-Pi)

两块开发板的主要参数如下：

||Raspberry pi 4 model B| MAIX-III AXera-Pi |
|---|---|---|
|CPU|4 核 `64-bit` ARM Cortex-`A72`|4 核 `32-bit` ARM Cortex-`A7`|
|RAM| 8 GB | 2 GB|
|麦克风| 无，需外接。|板子自带|

> 小编注：细心的读者可以从第一个视频的 `htop` 输出发现，
识别进程只占用 几十 `MB` 的内存。

> 小编注：对于树莓派，淘宝可以买一个 `USB` 免驱动的麦克风进行测试。
> 9 块包邮的那种 （是的，9 块钱一个）。


感谢
- 张老师 (https://github.com/jimbozhang) 提供树莓派进行测试
- `Sipeed` (https://wiki.sipeed.com/hardware/en/maixIII/ax-pi/axpi.html) 赠送爱芯派进行测试

新一代 `Kaldi` 开源框架，是完全开源的。比如使用文档、训练代码、
训练好的模型、部署代码等。

感兴趣的读者，如果想复现本文的演示视频，可以参考：

|芯片类型|文档地址|
|---|---|
|64-bit ARM|https://k2-fsa.github.io/sherpa/ncnn/install/aarch64-embedded-linux.html|
|32-bit ARM|https://k2-fsa.github.io/sherpa/ncnn/install/arm-embedded-linux.html|

## 树莓派 4

本节描述如何在树莓派 4 上，使用新一代 Kaldi 进行实时的语音识别。

> 小编注：本节适用于所有 64-bit 的 ARM 开发板。
> 如果是 32-bit 的开发板，请参考下节的 爱芯派。

### 1. 准备工具链

我们采用交叉编译的方式，在 Ubuntu 上进行编译。请参考

https://k2-fsa.github.io/sherpa/ncnn/install/aarch64-embedded-linux.html

安装交叉编译工具链。

为了便于大家阅读，我们上述文档重要的命令摘录如下：

```bash
mkdir -p $HOME/software
cd $HOME/software
wget https://huggingface.co/csukuangfj/sherpa-ncnn-toolchains/resolve/main/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu.tar.xz
tar xvf gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu.tar.xz

export PATH=$HOME/software/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin:$PATH
```

然后测试工具链是否安装正确：
```bash
aarch64-linux-gnu-gcc --version
```

上述命令应该有如下输出：

```bash
aarch64-linux-gnu-gcc (Linaro GCC 7.5-2019.12) 7.5.0
Copyright (C) 2017 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

### 2. 编译 `sherpa-ncnn`

```bash
git clone https://github.com/k2-fsa/sherpa-ncnn
cd sherpa-ncnn
./build-aarch64-linux-gnu.sh
```

上述命令会生成如下 3 个可执行文件：

```bash
$ ls -lh build-aarch64-linux-gnu/install/bin/
total 10M
-rwxr-xr-x 1 kuangfangjun root 3.4M Jan 13 21:16 sherpa-ncnn
-rwxr-xr-x 1 kuangfangjun root 3.4M Jan 13 21:16 sherpa-ncnn-alsa
-rwxr-xr-x 1 kuangfangjun root 3.4M Jan 13 21:16 sherpa-ncnn-microphone
```

我们需要用到下面两个：

- `sherpa-ncnn`: 用于识别单个音频文件。可以用它来测试 RTF。如果 RTF 小于 1，
   那么就可以用下面的 `sherpa-ncnn-alsa` 在板子上进行实时的语音识别。

- `sherpa-ncnn-alsa`： 用于实时读取麦克风并进行识别。

上述两个可执行文件，采用的是动态连接，只依赖系统库。验证方式如下：

```bash
$ readelf -d build-aarch64-linux-gnu/install/bin/sherpa-ncnn

Dynamic section at offset 0x302a80 contains 30 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)             Shared library: [libgomp.so.1]
 0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]
 0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
 0x000000000000000f (RPATH)              Library rpath: [$ORIGIN]
```

```bash
$ readelf -d build-aarch64-linux-gnu/install/bin/sherpa-ncnn-alsa

Dynamic section at offset 0x34ea48 contains 31 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)             Shared library: [libasound.so.2]
 0x0000000000000001 (NEEDED)             Shared library: [libgomp.so.1]
 0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]
 0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
 0x000000000000000f (RPATH)              Library rpath: [$ORIGIN]
```

### 4. 下载预训练模型

我们测试下面两个模型：

| 模型名字| 参数量|模型下载链接|
|---|---|---|
| 中英文混合模型| 约 80 million|https://huggingface.co/csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-06 |
| 纯英文模型 | 约 8.8 million| https://huggingface.co/marcoyang/sherpa-ncnn-conv-emformer-transducer-small-2023-01-09|

下述文档链接

https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html

包含了详细的模型下载和使用方式。

为了便于各位读者阅读，我们摘录上述两个模型的下载方式如下：

**下载中英文混合模型**

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-06
cd sherpa-ncnn-conv-emformer-transducer-2022-12-06
git lfs pull --include "*.bin"
```

执行完上述命令后，会得到下述文件：

```bash
sherpa-ncnn-conv-emformer-transducer-2022-12-06 fangjun$ ls -lh {encoder,decoder,joiner}*
-rw-r--r--  1 fangjun  staff   5.9M Dec  6 12:06 decoder_jit_trace-pnnx.ncnn.bin
-rw-r--r--  1 fangjun  staff   439B Dec  6 12:05 decoder_jit_trace-pnnx.ncnn.param
-rw-r--r--  1 fangjun  staff   141M Dec  6 12:06 encoder_jit_trace-pnnx.ncnn.bin
-rw-r--r--  1 fangjun  staff    99M Dec 28 11:03 encoder_jit_trace-pnnx.ncnn.int8.bin
-rw-r--r--  1 fangjun  staff    78K Dec 28 11:02 encoder_jit_trace-pnnx.ncnn.int8.param
-rw-r--r--  1 fangjun  staff    79K Jan 10 21:13 encoder_jit_trace-pnnx.ncnn.param
-rw-r--r--  1 fangjun  staff   6.9M Dec  6 12:06 joiner_jit_trace-pnnx.ncnn.bin
-rw-r--r--  1 fangjun  staff   3.5M Dec 28 11:03 joiner_jit_trace-pnnx.ncnn.int8.bin
-rw-r--r--  1 fangjun  staff   498B Dec 28 11:02 joiner_jit_trace-pnnx.ncnn.int8.param
-rw-r--r--  1 fangjun  staff   490B Dec  6 12:05 joiner_jit_trace-pnnx.ncnn.param
```

> 小编注：请注意核对 `*.bin` 文件的大小。

**下载纯英文模型**

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/marcoyang/sherpa-ncnn-conv-emformer-transducer-small-2023-01-09

cd sherpa-ncnn-conv-emformer-transducer-small-2023-01-09
git lfs pull --include "*.bin"
```

执行完上述命令后，会得到下述文件：

```bash
sherpa-ncnn-conv-emformer-transducer-small-2023-01-09 fangjun$ ls -lh {encoder,decoder,joiner}*
-rw-r--r--  1 fangjun  staff   314K Jan 16 14:44 decoder_jit_trace-pnnx.ncnn.bin
-rw-r--r--  1 fangjun  staff   436B Jan 16 14:44 decoder_jit_trace-pnnx.ncnn.param
-rw-r--r--  1 fangjun  staff    16M Jan 16 14:44 encoder_jit_trace-pnnx.ncnn.bin
-rw-r--r--  1 fangjun  staff    11M Jan 16 14:44 encoder_jit_trace-pnnx.ncnn.int8.bin
-rw-r--r--  1 fangjun  staff   103K Jan 16 14:44 encoder_jit_trace-pnnx.ncnn.int8.param
-rw-r--r--  1 fangjun  staff   105K Jan 16 14:44 encoder_jit_trace-pnnx.ncnn.param
-rw-r--r--  1 fangjun  staff   607K Jan 16 14:44 joiner_jit_trace-pnnx.ncnn.bin
-rw-r--r--  1 fangjun  staff   310K Jan 16 14:44 joiner_jit_trace-pnnx.ncnn.int8.bin
-rw-r--r--  1 fangjun  staff   495B Jan 16 14:44 joiner_jit_trace-pnnx.ncnn.int8.param
-rw-r--r--  1 fangjun  staff   487B Jan 16 14:44 joiner_jit_trace-pnnx.ncnn.param
```

> 小编注：请注意核对 `*.bin` 文件的大小。

### 5. RTF 测试

请把上面的模型文件以及生成的 `sherpa-ncnn` 和 `sherpa-ncnn-alsa` 拷贝到
树莓派4 开发板。

> 小编注：由于我们默认采用的是静态链接，你只需要拷贝生成的可执行文件即可。


我们使用下面的命令，测试两个模型在树莓派4上的 `RTF`.

> 小编注：`RTF` 是 `real-time factor` 的缩写。它反应了处理1秒音频所需的时间。
>
> 例子：如果处理 16 秒的数据，耗时 2 秒，那么 RTF 就是 `2/16 = 1/8 = 0.125`


**中英文混合模型 RTF 测试命令**
```bash
# fp16 测试, 4 线程, greedy search 解码
./sherpa-ncnn \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/tokens.txt \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.param \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.bin \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.param \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.bin\
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.param \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.bin\
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/test_wavs/1.wav \
  4 greedy_search
```

```bash
# int8 测试, 4 线程, greedy search 解码
./sherpa-ncnn \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/tokens.txt \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.int8.param \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.int8.bin \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.param \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.bin\
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.int8.param \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.int8.bin\
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/test_wavs/1.wav \
  4 greedy_search
```

**纯英文模型 RTF 测试命令**

```bash
# fp16 测试, 4 线程, greedy search 解码
./sherpa-ncnn \
  ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/tokens.txt \
  ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/encoder_jit_trace-pnnx.ncnn.param \
  ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/encoder_jit_trace-pnnx.ncnn.bin \
  ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/decoder_jit_trace-pnnx.ncnn.param \
  ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/decoder_jit_trace-pnnx.ncnn.bin \
  ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/joiner_jit_trace-pnnx.ncnn.param \
  ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/joiner_jit_trace-pnnx.ncnn.bin \
  ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/test_wavs/1089-134686-0001.wav \
  4 greedy_search
```

```bash
# int8 测试, 4 线程, greedy search 解码
./sherpa-ncnn \
  ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/tokens.txt \
  ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/encoder_jit_trace-pnnx.ncnn.int8.param \
  ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/encoder_jit_trace-pnnx.ncnn.int8.bin \
  ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/decoder_jit_trace-pnnx.ncnn.param \
  ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/decoder_jit_trace-pnnx.ncnn.bin \
  ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/joiner_jit_trace-pnnx.ncnn.int8.param \
  ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/joiner_jit_trace-pnnx.ncnn.int8.bin \
  ./sherpa-ncnn-conv-emformer-transducer-small-2023-01-09/test_wavs/1089-134686-0001.wav \
  4 greedy_search
```

> 小编注：上述两条命令中, 4 代表了使用线程的数量。`greedy_search` 代表解码方法。
> 目前支持的解码方法有 `greedy_search` 和 `modified_beam_search`。
>
> 本文所有测试，都是跑在 `CPU` 上。

RTF 测试结果如下 （取3次结果的平均值）:

|解码方法|线程数量|模型类型|中英文混合模型|纯英文模型|
|---|---|---|---|---|
|`greedy_search`| 1 | fp16 | 2.249 | 0.311|
|`greedy_search`| 2 | fp16 | 1.206 | 0.200|
|`greedy_search`| 3 | fp16 | 0.953 | 0.165|
|`greedy_search`| 4 | fp16 | 0.775 | 0.145|
|`modified_beam_search`| 1 | fp16 | 2.552 | 0.381|
|`modified_beam_search`| 2 | fp16 | 1.503 | 0.263|
|`modified_beam_search`| 3 | fp16 | 1.254 | 0.230|
|`modified_beam_search`| 4 | fp16 | 1.081 | 0.211|
|`greedy_search`| 1 | int8 | 1.389 | 0.222|
|`greedy_search`| 2 | int8 | 0.761 | 0.153|
|`greedy_search`| 3 | int8 | 0.609 | 0.132|
|`greedy_search`| 4 | int8 | 0.511 | 0.120|
|`modified_beam_search`| 1 | int8 | 1.683 | 0.287|
|`modified_beam_search`| 2 | int8 | 0.993 | 0.215|
|`modified_beam_search`| 3 | int8 | 0.846 | 0.191|
|`modified_beam_search`| 4 | int8 | 0.754 | 0.180|





