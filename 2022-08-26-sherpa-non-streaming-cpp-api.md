# sherpa C++ API 之非流式篇

> 本文介绍新一代 [Kaldi][Kaldi] 服务端框架 [sherpa][sherpa] 中
> 近期添加的非流式语音识别 C++ API。

## 简介

虽然 `sherpa` 的底层实现如神经网络计算和解码等都是基于 C++, 但 `sherpa`
一直只提供 Python 接口给用户使用。 Python 接口的优点在于它能够
降低用户使用 `sherpa` 的门槛，方便用户修改和扩展 `sherpa`。

近来有部分用户呼吁 `sherpa` 应该提供 C++ 接口，以便把 `sherpa` 与现有的基于
C++ 的产线系统相结合，如 <https://github.com/k2-fsa/sherpa/issues/85>
和 <https://github.com/k2-fsa/sherpa/issues/101> 都希望 `sherpa` 提供
C++ 接口。

作为回应，我们最近着手在 `sherpa` 中添加 C++ 接口。截至目前，我们已经完成了
基于 `RNN-T` 的`非流式`语音识别 C++ 接口。 `流式`识别的接口仍在开发中，将于近期
与大家见面。

> 小编注：如果大家有什么需求，请在新一代 Kaldi 对应的 repo 中提 issue, 详细
> 描述你需要的功能。

非流式接口目前提供如下功能:

- 给定音频对应的文件名进行识别
- 输入 audio samples 进行识别
- 输入 fbank 特征进行识别

为了方便大家使用，我们提供一个 `.cc` 文件，编译成 `binary` 之后，通过命令行
调用，可以支持：

- 提供一个文件名进行解码
- 提供多个文件名进行解码
- 对 `wav.scp` 进行解码。对的，就是 `Kaldi`里面的 `wav.scp`
- 对 `feats.scp` 进行解码。

值得指出的是，上述功能均支持以 `batch` 的方式进行识别，并支持 CPU 和 GPU。

同时，我们也提供预训练模型供大家下载。详细的使用方法，可以参考 `sherpa`
的使用文档：https://k2-fsa.github.io/sherpa/cpp/offline_asr/index.html


接下来我们介绍非流式 C++ 接口的**安装**及**使用**。

## 安装

`sherpa` 对 Linux，macOS 和 Windows 这三个平台都有良好的支持。
在上述三个平台上，可以通过编译源代码的方式安装 `sherpa`。针对 Linux 和 Windows,
我们还支持通过 `conda install` 安装 `sherpa`。

### 使用 conda install 安装 `sherpa`

> 小编注：此种方法仅支持 CPU 版本的 `sherpa`, 并且仅限于 Linux 和 Windows。

为了方便用户安装 `sherpa`，我们事先把 `sherpa` 及其依赖都编译好，并托管在
<https://anaconda.org/> 上。 用户只需下面一条命令即可安装 `sherpa` 及其依赖:

```bash
conda install \
  -c k2-fsa \
  -c k2-fsa-sherpa \
  -c kaldifeat \
  -c kaldi_native_io \
  -c pytorch \
  cpuonly \
  k2 \
  sherpa \
  kaldifeat \
  kaldi_native_io \
  pytorch=1.12.0 \
  python=3.8
```

或者写成单独一行：

```bash
conda install -c k2-fsa -c k2-fsa-sherpa -c kaldifeat -c kaldi_native_io -c pytorch cpuonly k2 sherpa kaldifeat kaldi_native_io pytorch=1.12.0 python=3.8
```

> 请务必别漏掉上面的 `-c` 选项。你可以交换 `-c` 的顺序，但是他们`一个都不能少`。

用户可以选择不同的 PyTorch 版本和对应的 Python 版本。 可以使用如下命令查看
`sherpa` 针对哪些版本的 PyTorch 和 Python 提供了预编译版本。

```bash
conda search -c k2-fsa-sherpa sherpa
```

部分输出如下所示（以 Linux 为例）：

```bash
Loading channels: done
# Name                       Version           Build  Channel
sherpa                         0.7.1 cpu_py3.8_torch1.9.0  k2-fsa-sherpa
sherpa                         0.7.1 cpu_py3.8_torch1.9.1  k2-fsa-sherpa
sherpa                         0.7.1 cpu_py3.9_torch1.10.0  k2-fsa-sherpa
sherpa                         0.7.1 cpu_py3.9_torch1.10.1  k2-fsa-sherpa
sherpa                         0.7.1 cpu_py3.9_torch1.10.2  k2-fsa-sherpa
sherpa                         0.7.1 cpu_py3.9_torch1.11.0  k2-fsa-sherpa
sherpa                         0.7.1 cpu_py3.9_torch1.12.0  k2-fsa-sherpa
sherpa                         0.7.1 cpu_py3.9_torch1.12.1  k2-fsa-sherpa
sherpa                         0.7.1 cpu_py3.9_torch1.8.1  k2-fsa-sherpa
sherpa                         0.7.1 cpu_py3.9_torch1.9.0  k2-fsa-sherpa
sherpa                         0.7.1 cpu_py3.9_torch1.9.1  k2-fsa-sherpa
```

如果上述输出中，找不到你希望的 PyTorch 和 Python 的组合，我们建议从源码安装
`sherpa`。

#### 查看是否成功安装 `sherpa`

为了检查是否成功安装 `sherpa`，可以在命令行里面输入：

```bash
sherpa-version
```

和

```bash
sherpa --help
```

> 由于 Windows 平台不支持在 executable 里面设置 runtime path。在运行
> 上述两条检查命令之前，你还需要设置如下环境变量

```bash
set path=%conda_prefix%\lib\site-packages\sherpa\bin;%path%
set path=%conda_prefix%\lib\site-packages\torch\lib;%path%
```

### 从源码安装 `sherpa`

从源码安装 `sherpa` 之前，用户需要先安装 `k2`。请参考
https://k2-fsa.github.io/k2/installation/index.html

安装好 `k2` 之后，还需要使用下述命令安装 `sherpa` 的另外
两个依赖：

```bash
pip install -U kaldifeat kaldi_native_io
```

> 小编注： `kaldifeat` 用于计算 fbank 特征；`kaldi_native_io` 用于
> 读取 `wav.scp` 和 `feats.scp`。

所有的依赖都安装好之后，我们可以用如下命令安装 `sherpa`：

```bash
git clone https://github.com/k2-fsa/sherpa
cd sherpa
python3 setup.py bdist_wheel
```
以 Linux 为例，上述命令执行完后，你可以在 `./dist` 目录里面找到一个文件名类似
`k2_sherpa-0.7.1-cp38-cp38-linux_x86_64.whl` 的文件。然后可以使用

```bash
pip install ./dist/k2_sherpa-0.7.1-cp38-cp38-linux_x86_64.whl
```

安装 `sherpa`。 安装完之后，你就可以在命令行里面执行

```bash
sherpa-version
```

和

```bash
sherpa --help
```

> 小编注：你也可以使用如下方式，从源码安装 `sherpa`：

```bash
git clone https://github.com/k2-fsa/sherpa
cd sherpa
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

> 执行完后，你可以在 `./bin` 目录下，找到 `sherpa-version` 和
`sherpa` 这两个可执行程序。

## 使用

> 以 Linux 为例。文中的描述，对 macOS 和 Windows 也是适用的。

安装好 `sherpa` 之后，接下来我们介绍使用 `sherpa` 的两种方式：
- （1）使用生成的 executable `sherpa`
- （2）如何在 C++ 代码中，
使用相应的头文件 `sherpa/cpp_api/offline_recognizer.h` 和
链接生成的动态库 `libsherpa_offline_recognizer.so`

### 使用生成的 executable `sherpa`

```bash
sherpa --help
```

可以查看 `sherpa` 的帮助。 目前它支持以下四种功能：

- （1）指定一个音频路径进行解码
- （2）指定多个音频路径，以 batch 的方式进行解码
- （3）给定 `wav.scp` 进行解码
- （4）给定 `feats.scp` 进行解码

下面我们以 [WenetSpeech][WenetSpeech] 预训练模型为例，向各位读者展示上面四种功能。

#### 下载预训练模型

开始实验之前，我们先下载使用 [icefall][icefall] 在 [WenetSpeech][WenetSpeech]
数据集上训练好的模型：

```bash
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless2
```

#### （1）指定一个音频路径进行解码

使用上面下载的模型，对一个音频文件进行识别的命令为

```bash
nn_model=./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt
tokens=./icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char/tokens.txt

wav1=./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000000.wav

sherpa \
  --nn-model=$nn_model \
  --tokens=$tokens \
  --use-gpu=false \
  --decoding-method=greedy_search \
  $wav1
```

输出结果如下所示：

```bash
[I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/parse_options.cc:495:int sherpa::ParseOptions::Read(int, const char* const*) 2022-08-20 23:06:09 sherpa --nn-model=./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt --tokens=./icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char/tokens.txt --use-gpu=false --decoding-method=greedy_search ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000000.wav

[I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/sherpa.cc:126:int main(int, char**) 2022-08-20 23:06:10
--nn-model=./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt
--tokens=./icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char/tokens.txt
--decoding-method=greedy_search
--use-gpu=false

[I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/sherpa.cc:270:int main(int, char**) 2022-08-20 23:06:11
filename: ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000000.wav
result: 对我做了介绍那么我想说的是呢大家如果对我的研究感兴趣呢
```

> 小编注：你需要提供的内容有：（a）模型 `--nn-model`。（b）符号表 `--tokens`。
> （c）解码方法 `--decoding-method`。（d）是否使用 GPU `--use-gpu`。
>
> `sherpa --help` 可以查看详细的说明。


#### （2）指定多个音频路径，以 batch 的方式进行解码

若要同时识别多条音频，可以使用如下命令：

```bash
nn_model=./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt
tokens=./icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char/tokens.txt

wav1=./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000000.wav
wav2=./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000001.wav
wav3=./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000002.wav

sherpa \
  --nn-model=$nn_model \
  --tokens=$tokens \
  --use-gpu=false \
  --decoding-method=greedy_search \
  $wav1 \
  $wav2 \
  $wav3
```

是的，你只要在第一条音频后面，继续添加更多的文件就可以了。

上述命令的输出如下所示：

```bash
[I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/parse_options.cc:495:int sherpa::ParseOptions::Read(int, const char* const*) 2022-08-20 23:07:05 sherpa --nn-model=./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt --tokens=./icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char/tokens.txt --use-gpu=false --decoding-method=greedy_search ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000000.wav ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000001.wav ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000002.wav

[I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/sherpa.cc:126:int main(int, char**) 2022-08-20 23:07:06
--nn-model=./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt
--tokens=./icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char/tokens.txt
--decoding-method=greedy_search
--use-gpu=false

[I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/sherpa.cc:284:int main(int, char**) 2022-08-20 23:07:07
filename: ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000000.wav
result: 对我做了介绍那么我想说的是呢大家如果对我的研究感兴趣呢

filename: ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000001.wav
result: 重点想谈三个问题首先呢就是这一轮全球金融动荡的表现

filename: ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000002.wav
result: 深入地分析这一次全球金融动荡背后的根源
```

#### （3）给定 `wav.scp` 进行解码

首先我们用下面的命令，先创建一个 `wav.scp` 文件：

```bash
cat > wav.scp <<EOF
wav0 ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000000.wav
wav1 ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000001.wav
wav2 ./icefall_asr_wenetspeech_pruned_transducer_stateless2/test_wavs/DEV_T0000000002.wav
EOF
```

然后可以用下面命令，对创建的 `wav.scp` 进行识别：

```bash
nn_model=./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt
tokens=./icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char/tokens.txt

sherpa \
  --nn-model=$nn_model \
  --tokens=$tokens \
  --use-gpu=false \
  --decoding-method=greedy_search \
  --use-wav-scp=true \
  scp:wav.scp \
  ark,scp,t:results.ark,results.scp
```

输出 log 如下：

```bash
[I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/parse_options.cc:495:int sherpa::ParseOptions::Read(int, const char* const*) 2022-08-20 23:10:01 sherpa --nn-model=./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt --tokens=./icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char/tokens.txt --use-gpu=false --decoding-method=greedy_search --use-wav-scp=true scp:wav.scp ark,scp,t:results.ark,results.scp

[I] /usr/share/miniconda/envs/sherpa/conda-bld/sherpa_1661003501349/work/sherpa/csrc/sherpa.cc:126:int main(int, char**) 2022-08-20 23:10:02
--nn-model=./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt
--tokens=./icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char/tokens.txt
--decoding-method=greedy_search
--use-gpu=false
```

用下面的命令可以查看识别结果：

```
$ cat results.ark

wav0 对我做了介绍那么我想说的是呢大家如果对我的研究感兴趣呢
wav1 重点想谈三个问题首先呢就是这一轮全球金融动荡的表现
wav2 深入地分析这一次全球金融动荡背后的根源
```

> 小编注：如果你已经有来自 `Kaldi` 生成的 `wav.scp`，那么你可以很方便的用
> 这种方式对他们进行解码。


> 可以使用 `--batch-size` 这个选项，来调整解码时使用的 batch size。

#### （4）给定 `feats.scp` 进行解码

如果你有事先计算好的特征并保存在 `feats.scp` 和 `feats.ark` 里面，那么你可以
用下面的命令对它们进行识别：

```bash
nn_model=./icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/cpu_jit_epoch_10_avg_2_torch_1.7.1.pt
tokens=./icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char/tokens.txt

sherpa \
  --nn-model=$nn_model \
  --tokens=$tokens \
  --use-gpu=false \
  --decoding-method=greedy_search \
  --use-feats-scp=true \
  scp:feats.scp \
  ark,scp,t:results.ark,results.scp
```

> `注意`: 如果你使用的是 `icefall` 提供的预训练模型，那么你不能使用 `Kaldi` 生成
> 的 `feats.scp`。因为计算特征时，`Kaldi` 使用的是没有归一化的 audio samples。
> 即 audio samples 的取值范围是 `[-32768, 32767]`。而 `icefall` 中采用的范围是
> `[-1, 1]`。

我们上面为大家介绍了如何使用生成的 executable `sherpa` 进行语音识别。接下来我们为大家介绍
如何在 C++ 代码里面使用 `sherpa` 提供的 C++ API 进行语音识别。

### 使用 `sherpa` 提供的 C++ API

安装好 `sherpa` 后，我们可以在安装目录里面，找到如下两个文件：

- （1）头文件： `sherpa/cpp_api/offline_recognizer.h`
- （2）动态库：`lib/libsherpa_offline_recognizer.so`

下面的代码展示了如何使用 `sherpa` 的 C++ API 识别给定的音频。

```c++
#include <iostream>

#include "sherpa/cpp_api/offline_recognizer.h"

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: ./test_decode_files /path/to/nn_model "
                 "/path/to/tokens.txt foo.wav [bar.wav [foobar.wav] ... ]\n";
    exit(EXIT_FAILURE);
  }
  std::string nn_model = argv[1];
  std::string tokens = argv[2];
  float sample_rate = 16000;
  bool use_gpu = false;

  sherpa::DecodingOptions opts;
  opts.method = sherpa::kGreedySearch;
  sherpa::OfflineRecognizer recognizer(nn_model, tokens, opts, use_gpu,
                                       sample_rate);

  if (argc == 4) {
    std::cout << "Decode single file\n";
    auto result = recognizer.DecodeFile(argv[3]);
    std::cout << argv[3] << "\n" << result.text << "\n";
    return 0;
  }

  std::cout << "Decode multiple files\n";

  std::vector<std::string> filenames;
  for (int i = 3; i != argc; ++i) {
    filenames.push_back(argv[i]);
  }

  auto results = recognizer.DecodeFileBatch(filenames);
  for (size_t i = 0; i != filenames.size(); ++i) {
    std::cout << filenames[i] << "\n" << results[i].text << "\n\n";
  }
  return 0;
}
```

假设上述代码保存在文件 `test_decode_files.cc` 中，下面的 `Makefile` 展示了
如何编译和链接：

```Makefile
sherpa_install_dir := $(shell python3 -c 'import os; import sherpa; print(os.path.dirname(sherpa.__file__))')
sherpa_cxx_flags := $(shell python3 -c 'import os; import sherpa; print(sherpa.cxx_flags)')

$(info sherpa_install_dir: $(sherpa_install_dir))
$(info sherpa_cxx_flags: $(sherpa_cxx_flags))

CXXFLAGS := -I$(sherpa_install_dir)/include
CXXFLAGS += -Wl,-rpath,$(sherpa_install_dir)/lib
CXXFLAGS += $(sherpa_cxx_flags)
CXXFLAGS += -std=c++14

LDFLAGS := -L $(sherpa_install_dir)/lib -lsherpa_offline_recognizer

$(info CXXFLAGS: $(CXXFLAGS))
$(info LDFLAGS: $(LDFLAGS))

test_decode_files: test_decode_files.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

.PHONY: clean
clean:
	$(RM) test_decode_files
```

由于使用了 runtime path，在 Linux 和 macOS 平台上，生成的 executable `test_decode_files`
可以直接运行，无须设置 `LD_LIBRARY_PATH` 环境变量。

> 小编注：上面的 API 只展示了如何识别音频文件。我们其实还支持输入 audio samples
> 和特征进行解码。更多的例子，可以参考：
>
> - https://github.com/k2-fsa/sherpa/blob/master/sherpa/cpp_api/test_decode_features.cc
> - https://github.com/k2-fsa/sherpa/blob/master/sherpa/cpp_api/test_decode_samples.cc

## 总结

本文介绍了如何安装 `sherpa` 和如何使用 `sherpa` 提供的 C++ 接口进行语音识别。
支持的输入类型有音频文件、`wav.scp` 和 `feats.scp` 等。

目前我们只提供了非流式的 C++ 接口，流式的 C++ 接口仍在开发中。

## What's next

目前 C++ 的实现还依赖 PyTorch, 我们后续会提供不依赖 PyTorch 的 C++ 实现。
敬请关注 `sherpa` 的开发进展。





[Kaldi]: https://github.com/kaldi-asr/kaldi
[sherpa]: https://github.com/k2-fsa/sherpa
[WenetSpeech]: https://github.com/wenet-e2e/WenetSpeech
[icefall]: https://github.com/k2-fsa/icefall
