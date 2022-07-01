# 如何基于新一代 Kaldi 框架快速搭建服务端 ASR 系统
>本文将介绍如何基于新一代 Kaldi 框架快速搭建一个服务端的 ASR 系统，包括数据准备、模型训练测试、服务端部署运行。
>
> 更多内容建议参考：
> 
> - [k2](https://github.com/k2-fsa/k2 "k2")
> - [icefall](https://github.com/k2-fsa/icefall "icefall")
> - [lhotse](https://github.com/lhotse-speech/lhotse "lhotse")
> - [sherpa](https://github.com/k2-fsa/sherpa "sherpa")


## 前言
距离新一代 Kaldi 开源框架的正式发布已经有一段时间了。截至目前，框架基本的四梁八柱都已经立起来了。那么，如何用它快速搭建一个 ASR 系统呢？

阅读过前面几期公众文的读者可能都知道新一代 Kaldi 框架主要包含了四个不同的子项目：`k2`、`icefall`、`lhotse`、`sherpa`。其中，`k2` 是核心算法库；`icefall` 是数据集训练测试示例脚本；`lhotse` 是语音数据处理工具集；`sherpa` 是服务端框架，四个子项目共同构成了新一代 Kaldi 框架。

另一方面，截至目前，新一代 Kaldi 框架在很多公开数据集上都获得了很有竞争力的识别结果，在 WenetSpeech 和 GigaSpeech 上甚至都获得了 SOTA 的性能。

看到这，相信很多小伙伴都已经摩拳擦掌、跃跃欲试了，那么本文的目标就是试图贯通新一代 Kaldi 的四个子项目，为快速搭建一个服务端的 ASR 系统提供一个简易的教程。希望看完本文的小伙伴都能顺利搭建出自己的 ASR 系统。

## 三步搭建 ASR 服务端系统
本文主要介绍如何从原始数据下载处理、模型训练测试、到得到一个服务端 ASR 系统的过程，根据功能，分为三步：

 - 数据准备和处理
 - 模型训练和测试
 - 服务端部署演示

本文介绍的 ASR 系统是基于 RNN-T 框架且不涉及外加的语言模型。所以，本文将不涉及 WFST 等语言模型的内容，如后期有需要，会在后面的文章中另行讲述。

为了更加形象、具体地描述这个过程，本文以构建一个基于 WenetSpeech 数据集训练的 [pruned transducer stateless2](https://github.com/k2-fsa/icefall/tree/master/egs/wenetspeech/ASR "pruned transducer stateless2 recipe") recipe 为例，希望尽可能为读者详细地描述这一过程，也希望读者在本文的基础上能够无障碍地迁移到其他数据集的处理、训练和部署使用上去。

本文描述的过程和展示的代码更多的是为了描述功能，而非详细的实现过程。详细的实现代码请读者自行参考 [egs/wenetspeech/ASR](https://github.com/k2-fsa/icefall/tree/master/egs/wenetspeech/ASR "pruned transducer stateless2 recipe")。

**Note**: 使用者应该事先安装好 `k2`、`icefall`、`lhotse`、`sherpa`。

### 第一步：数据准备和处理
对于数据准备和处理部分，所有的运行指令都集成在文件 [prepare.sh](https://github.com/k2-fsa/icefall/blob/master/egs/wenetspeech/ASR/prepare.sh "prepare.sh") 中，主要的作用可以总结为两个：`准备音频文件并进行特征提取`、`构建语言建模文件`。

#### 准备音频文件并进行特征提取

（注：在这里我们也用了 musan 数据集对训练数据进行增广，具体的可以参考 [prepare.sh](https://github.com/k2-fsa/icefall/blob/master/egs/wenetspeech/ASR/prepare.sh "prepare.sh") 中对 musan 处理和使用的相关指令，这里不针对介绍。）

##### 下载并解压数据

为了统一文件名，这里将数据包文件名变为 WenetSpeech, 其中 audio 包含了所有训练和测试的音频数据
```bash
>> tree download/WenetSpeech -L 1
download/WenetSpeech
├── audio
├── TERMS_OF_ACCESS
└── WenetSpeech.json

>> tree download/WenetSpeech/audio -L 1
download/WenetSpeech/audio
├── dev
├── test_meeting
├── test_net
└── train
```
`WenetSpeech.json` 中包含了音频文件路径和相关的监督信息，我们可以查看 `WenetSpeech.json` 文件，部分信息如下所示：

```json
    "audios": [
        {
            "aid": "Y0000000000_--5llN02F84",
            "duration": 2494.57,
            "md5": "48af998ec7dab6964386c3522386fa4b",
            "path": "audio/train/youtube/B00000/Y0000000000_--5llN02F84.opus",
            "source": "youtube",
            "tags": [
                "drama"
            ],
            "url": "https://www.youtube.com/watch?v=--5llN02F84",
            "segments": [
                {
                    "sid": "Y0000000000_--5llN02F84_S00000",
                    "confidence": 1.0,
                    "begin_time": 20.08,
                    "end_time": 24.4,
                    "subsets": [
                        "L"
                    ],
                    "text": "怎么样这些日子住得还习惯吧"
                },
                {
                    "sid": "Y0000000000_--5llN02F84_S00002",
                    "confidence": 1.0,
                    "begin_time": 25.0,
                    "end_time": 26.28,
                    "subsets": [
                        "L"
                    ],
                    "text": "挺好的"
```

（注：WenetSpeech 中文数据集中包含了 S，M，L 三个不同规模的训练数据集）

##### 利用 lhotse 生成 manifests

关于 lhotse 是如何将原始数据处理成 `jsonl.gz` 格式文件的，这里可以参考文件[wenet_speech.py](https://github.com/lhotse-speech/lhotse/blob/master/lhotse/recipes/wenet_speech.py "wenet_speech.py")， 其主要功能是生成 `recordings` 和 `supervisions` 的 `jsonl.gz` 格式文件

```bash
>> lhotse prepare wenet-speech download/WenetSpeech data/manifests -j 15
>> tree data/manifests -L 1
├── wenetspeech_recordings_DEV.jsonl.gz
├── wenetspeech_recordings_L.jsonl.gz
├── wenetspeech_recordings_M.jsonl.gz
├── wenetspeech_recordings_S.jsonl.gz
├── wenetspeech_recordings_TEST_MEETING.jsonl.gz
├── wenetspeech_recordings_TEST_NET.jsonl.gz
├── wenetspeech_supervisions_DEV.jsonl.gz
├── wenetspeech_supervisions_L.jsonl.gz
├── wenetspeech_supervisions_M.jsonl.gz
├── wenetspeech_supervisions_S.jsonl.gz
├── wenetspeech_supervisions_TEST_MEETING.jsonl.gz
└── wenetspeech_supervisions_TEST_NET.jsonl.gz
```

这里，可用 `vim` 对 `recordings` 和 `supervisions` 的 `jsonl.gz` 文件进行查看, 其中：

wenetspeech_recordings_S.jsonl.gz:
- ![wenetspeech_recordings_S.jsonl.gz](https://github.com/k2-fsa/next-gen-kaldi-wechat/raw/master/pic/pic_lms/wenetspeech_recordings_S.png)

wenetspeech_supervisions_S.jsonl.gz:
- ![wenetspeech_supervisions_S.jsonl.gz](https://github.com/k2-fsa/next-gen-kaldi-wechat/raw/master/pic/pic_lms/wenetspeech_supervisions_S.png)

由上面两幅图可知，`recordings` 用于描述音频文件信息，包含了音频样本的 id、具体路径、通道、采样率、子样本数和时长等。`supervisions` 用于记录监督信息，包含了音频样本对应的 id、起始时间、时长、通道、文本和语言类型等。

接下来，我们将对音频数据提取特征。

##### 计算、提取和贮存音频特征

首先，对数据进行预处理，包括对文本进行标准化和对音频进行时域上的增广，可参考文件 [preprocess_wenetspeech.py](https://github.com/k2-fsa/icefall/blob/master/egs/wenetspeech/ASR/local/preprocess_wenetspeech.py "preprocess_wenetspeech.py")。

```bash
python3 ./local/preprocess_wenetspeech.py
```

其次，将数据集切片并对每个切片数据集进行特征提取。可参考文件  [compute_fbank_wenetspeech_splits.py](https://github.com/k2-fsa/icefall/blob/master/egs/wenetspeech/ASR/local/compute_fbank_wenetspeech_splits.py "compute_fbank_wenetspeech_splits.py")。

（注：这里的切片是为了可以开启多个进程同时对大规模数据集进行特征提取，提高效率。如果数据集比较小，对数据进行切片处理不是必须的。）

```bash
# 这里的 L 也可修改为 M 或 S, 表示训练数据子集

lhotse split 1000 ./data/fbank/cuts_L_raw.jsonl.gz data/fbank/L_split_1000

python3 ./local/compute_fbank_wenetspeech_splits.py \
    --training-subset L \
    --num-workers 20 \
    --batch-duration 600 \
    --start 0 \
    --num-splits 1000
```

最后，待提取完每个切片数据集的特征后，将所有切片数据集的特征数据合并成一个总的特征数据集：

```bash
# 这里的 L 也可修改为 M 或 S, 表示训练数据子集

pieces=$(find data/fbank/L_split_1000 -name "cuts_L.*.jsonl.gz")
lhotse combine $pieces data/fbank/cuts_L.jsonl.gz
```

至此，我们基本完成了音频文件的准备和特征提取。接下来，我们将构建语言建模文件。

#### 构建语言建模文件
在 `RNN-T` 模型框架中，我们实际需要的用于训练和测试的建模文件有 `tokens.txt`、`words.txt` 和 `Linv.pt` 。 我们按照如下步骤构建语言建模文件：

##### 规范化文本并生成 text

在这一步骤中，规范文本的函数文件可参考 [text2token.py](https://github.com/k2-fsa/icefall/blob/master/egs/wenetspeech/ASR/local/text2token.py "text2token.py")。

```bash
# Note: in Linux, you can install jq with the following command:
# 1. wget -O jq https://github.com/stedolan/jq/releases/download/jq-1.6/jq-linux64
# 2. chmod +x ./jq
# 3. cp jq /usr/bin

gunzip -c data/manifests/wenetspeech_supervisions_L.jsonl.gz \
      | jq 'text' | sed 's/"//g' \
      | ./local/text2token.py -t "char" > data/lang_char/text
```

`text` 的形式如下：

```
 怎么样这些日子住得还习惯吧
 挺好的
 对了美静这段日子经常不和我们一起用餐
 是不是对我回来有什么想法啊
 哪有的事啊
 她这两天挺累的身体也不太舒服
 我让她多睡一会那就好如果要是觉得不方便
 我就搬出去住
 ............
```

##### 分词并生成 words.txt

这里我们用 `jieba` 对中文句子进行分词，可参考文件 [text2segments.py](https://github.com/k2-fsa/icefall/blob/master/egs/wenetspeech/ASR/local/text2segments.py "text2segments.py") 。

```bash
python3 ./local/text2segments.py \
    --input-file data/lang_char/text \
    --output-file data/lang_char/text_words_segmentation

cat data/lang_char/text_words_segmentation | sed 's/ /\n/g' \
    | sort -u | sed '/^$/d' | uniq > data/lang_char/words_no_ids.txt

python3 ./local/prepare_words.py \
    --input-file data/lang_char/words_no_ids.txt \
    --output-file data/lang_char/words.txt
```
`text_words_segmentation` 的形式如下：
```
  怎么样 这些 日子 住 得 还 习惯 吧
  挺 好 的
  对 了 美静 这段 日子 经常 不 和 我们 一起 用餐
  是不是 对 我 回来 有 什么 想法 啊
  哪有 的 事 啊
  她 这 两天 挺累 的 身体 也 不 太 舒服
  我 让 她 多 睡 一会 那就好 如果 要是 觉得 不 方便
  我 就 搬出去 住
  ............
```
`words_no_ids.txt` 的形式如下：
```
............
阿
阿Q
阿阿虎
阿阿离
阿阿玛
阿阿毛
阿阿强
阿阿淑
阿安
............
```
`words.txt` 的形式如下：
```
............
阿 225
阿Q 226
阿阿虎 227
阿阿离 228
阿阿玛 229
阿阿毛 230
阿阿强 231
阿阿淑 232
阿安 233
............
```
##### 生成 tokens.txt 和 lexicon.txt

这里生成 `tokens.txt` 和 `lexicon.txt` 的函数文件可参考 [prepare_char.py](https://github.com/k2-fsa/icefall/blob/master/egs/wenetspeech/ASR/local/prepare_char.py "prepare_char.py") 。

```bash
python3 ./local/prepare_char.py \
    --lang-dir data/lang_char
```
`tokens.txt` 的形式如下：
```
<blk> 0
<sos/eos> 1
<unk> 2
怎 3
么 4
样 5
这 6
些 7
日 8
子 9
............
```
`lexicon.txt` 的形式如下：
```
............
X光 X 光
X光线 X 光 线
X射线 X 射 线
Y Y
YC Y C
YS Y S
YY Y Y
Z Z
ZO Z O
ZSU Z S U
○ ○
一 一
一一 一 一
一一二 一 一 二
一一例 一 一 例
............
```

至此，第一步全部完成。对于不同数据集来说，其基本思路也是类似的。在数据准备和处理阶段，我们主要做两件事情：`准备音频文件并进行特征提取`、`构建语言建模文件`。

这里我们使用的范例是中文汉语，建模单元是字。在英文数据中，我们一般用 BPE 作为建模单元，具体的可参考 [egs/librispeech/ASR/prepare.sh](https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR "egs/librispeech/ASR/prepare.sh") 。

### 第二步：模型训练和测试
在完成第一步的基础上，我们可以进入到第二步，即模型的训练和测试了。这里，我们根据操作流程和功能，将第二步划分为更加具体的几步：文件准备、数据加载、模型训练、解码测试。

#### 文件准备

首先，创建 pruned_transducer_stateless2 的文件夹。

```bash
mkdir pruned_transducer_stateless2
cd pruned_transducer_stateless2
```

其次，我们需要准备数据读取、模型、训练、测试、模型导出等脚本文件。在这里，我们在 [egs/librispeech/ASR/pruned_transducer_stateless2](https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless2 "egs/librispeech/ASR/pruned_transducer_stateless2") 的基础上创建我们需要的文件。

对于公共的脚本文件（即不需要修改的文件），我们可以用软链接直接复制过来，如：

```bash
ln -s ../../../librispeech/ASR/pruned_transducer_stateless2/conformer.py .
```

其他相同文件的操作类似。另外，读者也可以使用自己的模型，替换本框架内提供的模型文件即可。

对于不同的脚本文件（即因为数据集或者语言不同而需要修改的文件），我们先从 `egs/librispeech/ASR/pruned_transducer_stateless2` 中复制过来，然后再进行小范围的修改，如：
```bash
cp -r ../../../librispeech/ASR/pruned_transducer_stateless2/train.py .
```
在本示例中，我们需要对 `train.py` 中的数据读取、graph_compiler（图编译器）及
vocab_size 的获取等部分进行修改，如（截取部分代码，便于读者直观认识）：

数据读取：
```python
    ............
    from asr_datamodule import WenetSpeechAsrDataModule
    ............
    wenetspeech = WenetSpeechAsrDataModule(args)

    train_cuts = wenetspeech.train_cuts()
    valid_cuts = wenetspeech.valid_cuts()
    ............
```
graph_compiler:
```python
    ............
    y = graph_compiler.texts_to_ids(texts)
    if type(y) == list:
        y = k2.RaggedTensor(y).to(device)
    else:
        y = y.to(device)
    ............
    lexicon = Lexicon(params.lang_dir)
    graph_compiler = CharCtcTrainingGraphCompiler(
        lexicon=lexicon,
        device=device,
    )
    ............
```
vocab_size 的获取:
```python
    ............
    params.blank_id = lexicon.token_table["<blk>"]
    params.vocab_size = max(lexicon.tokens) + 1
    ............
```
更加详细的修改后的 train.py 可参考 [egs/wenetspeech/ASR/pruned_transducer_stateless2/train.py](https://github.com/k2-fsa/icefall/blob/master/egs/wenetspeech/ASR/pruned_transducer_stateless2/train.py "egs/wenetspeech/ASR/pruned_transducer_stateless2/train.py") 。
其他 decode.py、pretrained.py、export.py 等需要修改的文件也可以参照上述进行类似的修改和调整。

（注：在准备文件时，应该遵循`相同的文件不重复造轮子、不同的文件尽量小改、缺少的文件自己造`的原则。icefall 中大多数函数和功能文件在很多数据集上都进行了测试和验证，都是可以直接迁移使用的。）

#### 数据加载

实际上，对于数据加载这一步，也可以视为文件准备的一部分，即修改文件 [asr_datamodule.py](https://github.com/k2-fsa/icefall/blob/master/egs/wenetspeech/ASR/pruned_transducer_stateless2/asr_datamodule.py "asr_datamodule.py")，但是考虑到不同数据集的 asr_datamodule.py 都不一样，所以这里单独拿出来讲述。

首先，这里以 [egs/librispeech/ASR/pruned_transducer_stateless2/asr_datamodule.py](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless2/asr_datamodule.py "egs/librispeech/ASR/pruned_transducer_stateless2/asr_datamodule.py") 为基础，在这个上面进行修改：

```bash
cp -r ../../../librispeech/ASR/pruned_transducer_stateless2/asr_datamodule.py .
```

其次，修改函数类的名称，如这里将 `LibriSpeechAsrDataModule` 修改为 `WenetSpeechAsrDataModule` ，并读取第一步中生成的 `jsonl.gz` 格式的训练测试文件。本示例中，第一步生成了 `data/fbank/cuts_L.jsonl.gz`，我们用 `load_manifest_lazy` 读取它：

```python
    ............
        group.add_argument(
            "--training-subset",
            type=str,
            default="L",
            help="The training subset for using",
        )
    ............
    @lru_cache()
    def train_cuts(self) -> CutSet:
        logging.info("About to get train cuts")
        cuts_train = load_manifest_lazy(
            self.args.manifest_dir
            / f"cuts_{self.args.training_subset}.jsonl.gz"
        )
        return cuts_train
    ............
```

其他的训练测试集的 `jsonl.gz` 文件读取和上述类似。另外，对于 `train_dataloaders`、`valid_dataloaders` 和 `test_dataloaders` 等几个函数基本是不需要修改的，如有需要，调整其中的具体参数即可。

最后，调整修改后的 `asr_datamodule.py` 和 `train.py` 联合调试，把 `WenetSpeechAsrDataModule` 导入到 `train.py`，运行它，如果在数据读取和加载过程中不报错，那么数据加载部分就完成了。

另外，在数据加载的过程中，我们也有必要对数据样本的时长进行统计，并过滤一些过短、过长且占比极小的样本，这样可以使我们的训练过程更加稳定。

在本示例中，我们对 WenetSpeech 的样本进行了时长统计（L 数据集太大，这里没有对它进行统计），具体的可参考 [display_manifest_statistics.py](https://github.com/k2-fsa/icefall/blob/master/egs/wenetspeech/ASR/local/display_manifest_statistics.py, "display_manifest_statistics.py")，统计的部分结果如下：
```
............
Starting display the statistics for ./data/fbank/cuts_M.jsonl.gz
Cuts count: 4543341
Total duration (hours): 3021.1
Speech duration (hours): 3021.1 (100.0%)
***
Duration statistics (seconds):
mean    2.4
std     1.6
min     0.2
25%     1.4
50%     2.0
75%     2.9
99%     8.0
99.5%   8.8
99.9%   12.1
max     405.1
............
Starting display the statistics for ./data/fbank/cuts_TEST_NET.jsonl.gz
Cuts count: 24774
Total duration (hours): 23.1
Speech duration (hours): 23.1 (100.0%)
***
Duration statistics (seconds):
mean    3.4
std     2.6
min     0.1
25%     1.4
50%     2.4
75%     4.8
99%     13.1
99.5%   14.5
99.9%   18.5
max     33.3
```
根据上面的统计结果，我们在 `train.py` 中设置了样本的最大时长为 15.0 seconds:
```python
    ............
    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1 second and 15.0 seconds
        #
        # Caution: There is a reason to select 15.0 here. Please see
        # ../local/display_manifest_statistics.py
        #
        # You should use ../local/display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        return 1.0 <= c.duration <= 15.0

    train_cuts = train_cuts.filter(remove_short_and_long_utt)
    ............
```

#### 模型训练

在完成相关必要文件准备和数据加载成功的基础上，我们可以开始进行模型的训练了。

在训练之前，我们需要根据训练数据的规模和我们的算力条件（比如 GPU 显卡的型号、GPU 显卡的数量、每个卡的显存大小等）去调整相关的参数。

这里，我们将主要介绍几个比较关键的参数，其中，`world-size` 表示并行计算的 GPU 数量，`max-duration` 表示每个 batch 中所有音频样本的最大时长之和，`num-epochs` 表示训练的 epochs 数，`valid-interval` 表示在验证集上计算 loss 的 iterations 间隔，`model-warm-step` 表示模型热启动的 iterations 数，`use-fp16` 表示是否用16位的浮点数进行训练等，其他参数可以参考 [train.py](https://github.com/k2-fsa/icefall/blob/master/egs/wenetspeech/ASR/pruned_transducer_stateless2/train.py "train.py") 具体的参数解释和说明。

在这个示例中，我们用 WenetSpeech 中 `L subset` 训练集来进行训练，并综合考虑该数据集的规模和我们的算力条件，训练参数设置和运行指令如下（没出现的参数表示使用默认的参数值）：
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

python3 pruned_transducer_stateless2/train.py \
  --lang-dir data/lang_char \
  --exp-dir pruned_transducer_stateless2/exp \
  --world-size 8 \
  --num-epochs 15 \
  --start-epoch 0 \
  --max-duration 180 \
  --valid-interval 3000 \
  --model-warm-step 3000 \
  --save-every-n 8000 \
  --training-subset L
```

到这里，如果能看到训练过程中的 `loss` 记录的输出，则说明训练已经成功开始了。

另外，如果在训练过程中，出现了 `Out of Memory` 的报错信息导致训练中止，可以尝试使用更小一些的 `max-duration` 值。如果还有其他的报错导致训练中止，一方面希望读者可以灵活地根据实际情况修改或调整某些参数，另一方面，读者可以在相关讨论群或者在icefall 上通过 `issues` 和 `pull request` 等形式进行反馈。

如果程序在中途中止训练，我们也不必从头开始训练，可以通过加载保存的某个 `epoch-X.pt` 或 `checkpoint-X.pt` 模型文件（包含了模型参数、采样器和学习率等参数）继续训练，如加载 epoch-3.pt 的模型文件继续训练：

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

python3 pruned_transducer_stateless2/train.py \
  --lang-dir data/lang_char \
  --exp-dir pruned_transducer_stateless2/exp \
  --world-size 8 \
  --num-epochs 15 \
  --start-batch 3 \
  --max-duration 180 \
  --valid-interval 3000 \
  --model-warm-step 3000 \
  --save-every-n 8000 \
  --training-subset L
```
这样即使程序中断了，我们也不用从零开始训练模型。

另外，我们也不用从第一个 `batch` 进行迭代训练，因为采样器中保存了迭代的 batch 数，我们可以设置参数 `--start-batch xxx`, 使得我们可以从某一个 epoch 的某个 batch 处开始训练，这大大节省了训练时间和计算资源，尤其是在训练大规模数据集时。

在 icefall 中，还有更多类似这样人性化的训练设置，等待大家去发现和使用。

当训练完毕以后，我们可以得到相关的训练 `log` 文件和 `tensorboard` 损失记录，可以在终端使用如下指令：

```bash 
cd pruned_transducer_stateless2/exp

tensorboard dev upload --logdir tensorboard
```

如在使用上述指令之后，我们可以在终端看到如下信息：

```
............
To stop uploading, press Ctrl-C.

New experiment created. View your TensorBoard at: https://tensorboard.dev/experiment/wM4ZUNtASRavJx79EOYYcg/

[2022-06-30T15:49:38] Started scanning logdir.
Uploading 4542 scalars...
............
```

将上述显示的 `tensorboard` 记录查看网址复制到本地浏览器的网址栏中即可查看。如在本示例中，我们将 https://tensorboard.dev/experiment/wM4ZUNtASRavJx79EOYYcg/ 复制到本地浏览器的网址栏中，损失函数的 tensorboard 记录如下：
 - ![wenetspeech_L_tensorboard.png](https://github.com/k2-fsa/next-gen-kaldi-wechat/raw/master/pic/pic_lms/wenetspeech_L_tensorboard.png)

（PS: 读者可从上图发现，笔者在训练 WenetSpeech L subset 时，也因为某些原因中断了训练，但是，icefall 中人性化的接续训练操作让笔者避免了从零开始训练，并且前后两个训练阶段的 `loss` 和 `learning rate` 曲线还连接地如此完美。）

#### 解码测试

当模型训练完毕，我们就可以进行解码测试了。

在运行解码测试的指令之前，我们依然需要对 `decode.py` 进行如文件准备过程中对 `train.py` 相似位置的修改和调整，这里将不具体讲述，修改后的文件可参考 [decode.py](https://github.com/k2-fsa/icefall/blob/master/egs/wenetspeech/ASR/pruned_transducer_stateless2/decode.py "decode.py")。

这里为了在测试过程中更快速地加载数据，我们将测试数据导出为 `webdataset` 要求的形式（注：这一步不是必须的，如果测试过程中速度比较快，这一步可以省略），操作如下：

```python
    ............
    # Note: Please use "pip install webdataset==0.1.103"
    # for installing the webdataset.
    import glob
    import os

    from lhotse import CutSet
    from lhotse.dataset.webdataset import export_to_webdataset

    wenetspeech = WenetSpeechAsrDataModule(args)

    dev = "dev"
    ............

    if not os.path.exists(f"{dev}/shared-0.tar"):
        os.makedirs(dev)
        dev_cuts = wenetspeech.valid_cuts()
        export_to_webdataset(
            dev_cuts,
            output_path=f"{dev}/shared-%d.tar",
            shard_size=300,
        )
    ............
    dev_shards = [
        str(path)
        for path in sorted(glob.glob(os.path.join(dev, "shared-*.tar")))
    ]
    cuts_dev_webdataset = CutSet.from_webdataset(
        dev_shards,
        split_by_worker=True,
        split_by_node=True,
        shuffle_shards=True,
    )
    ............
    dev_dl = wenetspeech.valid_dataloaders(cuts_dev_webdataset)
    ............
```

同时，在 `asr_datamodule.py` 中修改 `test_dataloader` 函数，修改如下（注：这一步不是必须的，如果测试过程中速度比较快，这一步可以省略）：

```python
        ............
        from lhotse.dataset.iterable_dataset import IterableDatasetWrapper

        test_iter_dataset = IterableDatasetWrapper(
            dataset=test,
            sampler=sampler,
        )
        test_dl = DataLoader(
            test_iter_dataset,
            batch_size=None,
            num_workers=self.args.num_workers,
        )
        return test_dl
```

待修改完毕，联合调试 decode.py 和 asr_datamodule.py, 解码过程能正常加载数据即可。

在进行解码测试时，icefall 为我们提供了四种解码方式：`greedy_search`、`beam_search`、`modified_beam_search` 和 `fast_beam_search`，更为具体实现方式，可参考文件 [beam_search.py](https://github.com/k2-fsa/icefall/blob/master/egs/wenetspeech/ASR/pruned_transducer_stateless2/train.py "beam_search.py")。

这里，因为建模单元的数量非常多（5500+），导致解码速度非常慢，所以，笔者不建议使用 beam_search 的解码方式。

在本示例中，如果使用 greedy_search 进行解码，我们的解码指令如下 （
关于如何使用其他的解码方式，读者可以自行参考 decode.py）：

```bash
export CUDA_VISIBLE_DEVICES='0'
python pruned_transducer_stateless2/decode.py \
        --epoch 10 \
        --avg 2 \
        --exp-dir ./pruned_transducer_stateless2/exp \
        --lang-dir data/lang_char \
        --max-duration 100 \
        --decoding-method greedy_search
```

运行上述指令进行解码，在终端将会展示如下内容（部分）：

```
............
2022-06-30 16:58:17,232 INFO [decode.py:487] About to create model
2022-06-30 16:58:17,759 INFO [decode.py:508] averaging ['pruned_transducer_stateless2/exp/epoch-9.pt', 'pruned_transducer_stateless2/exp/epoch-10.pt']
............
2022-06-30 16:58:42,260 INFO [decode.py:393] batch 0/?, cuts processed until now is 104
2022-06-30 16:59:41,290 INFO [decode.py:393] batch 100/?, cuts processed until now is 13200
2022-06-30 17:00:35,961 INFO [decode.py:393] batch 200/?, cuts processed until now is 27146
2022-06-30 17:00:38,370 INFO [decode.py:410] The transcripts are stored in pruned_transducer_stateless2/exp/greedy_search/recogs-DEV-greedy_search-epoch-10-avg-2-context-2-max-sym-per-frame-1.txt
2022-06-30 17:00:39,129 INFO [utils.py:410] [DEV-greedy_search] %WER 7.80% [51556 / 660996, 6272 ins, 18888 del, 26396 sub ]
2022-06-30 17:00:41,084 INFO [decode.py:423] Wrote detailed error stats to pruned_transducer_stateless2/exp/greedy_search/errs-DEV-greedy_search-epoch-10-avg-2-context-2-max-sym-per-frame-1.txt
2022-06-30 17:00:41,092 INFO [decode.py:440]
For DEV, WER of different settings are:
greedy_search   7.8     best for DEV
............
```

这里，读者可能还有一个疑问，如何选取合适的 `epoch` 和 `avg` 参数，以保证平均模型的性能最佳呢？这里我们通过遍历所有的 epoch 和 avg 组合来搜索最好的平均模型，可以使用如下指令得到所有可能的平均模型的性能，然后进行找到最好的解码结果所对应的平均模型的 epoch 和 avg 即可，如：

```bash
export CUDA_VISIBLE_DEVICES="0"
num_epochs=15
for ((i=$num_epochs; i>=0; i--));
do
    for ((j=1; j<=$i; j++));
    do
        python3 pruned_transducer_stateless2/decode.py \
            --exp-dir ./pruned_transducer_stateless2/exp \
            --lang-dir data/lang_char \
            --epoch $i \
            --avg $j \
            --max-duration 100 \
            --decoding-method greedy_search
    done
done
```

以上方法仅供读者参考，读者可根据自己的实际情况进行修改和调整。目前，icefall 也提供了一种新的平均模型参数的方法，性能更好，这里将不作细述，有兴趣可以参考文件 [decode.py](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless5/train.py "decode.py") 中的参数 `--use-averaged-model`。

至此，解码测试就完成了。使用者也可以通过查看 `egs/pruned_transducer_stateless2/exp/greedy_search` 中 `recogs-*.txt`、`errs-*.txt` 和 `wer-*.txt` 等文件，看看每个样本的具体解码结果和最终解码性能。

本示例中，笔者的训练模型和测试结果可以参考 [icefall_asr_wenetspeech_pruned_transducer_stateless2](https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless2 "icefall_asr_wenetspeech_pruned_transducer_stateless2")，读者可以在 [icefall_asr_wenetspeech_pruned_transducer_stateless2_colab_demo](https://colab.research.google.com/drive/1EV4e1CHa1GZgEF-bZgizqI9RyFFehIiN?usp=sharing "icefall_asr_wenetspeech_pruned_transducer_stateless2_colab_demo") 上直接运行和测试提供的模型，这些仅供读者参考。

### 第三步：服务端部署演示

在顺利完成第一步和第二步之后，我们就可以得到训练模型和测试结果了。

接下来，笔者将讲述如何利用 sherpa 框架把训练得到的模型部署到服务端，笔者强烈建议读者参考和阅读 [sherpa使用文档](https://k2-fsa.github.io/sherpa/ "sherpa使用文档")，该框架还在不断地更新和优化中，感兴趣的读者可以保持关注并参与到开发中来。

本示例中，我们用的 sherpa 版本为 [sherpa-for-wenetspeech-pruned-rnnt2](https://github.com/k2-fsa/sherpa/tree/9da5b0779ad6758bf3150e1267399fafcdef4c67 "sherpa-for-wenetspeech-pruned-rnnt2")。

为了将整个过程描述地更加清晰，笔者同样将第三步细分为以下几步：`将训练好的模型编译为 TorchScript 代码`、`服务器终端运行`、`本地 web 端测试使用`。

#### 将训练好的模型编译为 TorchScript 代码

这里，我们使用 `torch.jit.script` 对模型进行编译，使得 `nn.Module` 形式的模型在生产环境下变得可用，具体的代码实现可参考文件 [export.py](https://github.com/k2-fsa/icefall/blob/master/egs/wenetspeech/ASR/pruned_transducer_stateless2/export.py "export.py")，操作指令如下：

```bash
python3 pruned_transducer_stateless2/export.py \
    --exp-dir ./pruned_transducer_stateless2/exp \
    --lang-dir data/lang_char \
    --epoch 10 \
    --avg 2 \
    --jit True
```
运行上述指令，我们可以在 `egs/wenetspeech/ASR/pruned_transducer_stateless2/exp` 中得到一个 `cpu_jit.pt` 的文件，这是我们在 sherpa 框架里将要使用的模型文件。

#### 服务器终端运行

本示例中，我们的模型是中文非流式的，所以我们选择非流式模式来运行指令，同时，我们需要选择在上述步骤中生成的 `cpu_jit.pt` 和 `tokens.txt` ：
```bash
python3 sherpa/bin/conformer_rnnt/offline_server.py \
    --port 6006 \
    --num-device 1 \
    --max-batch-size 10 \
    --max-wait-ms 5 \
    --max-active-connections 500 \
    --feature-extractor-pool-size 5 \
    --nn-pool-size 1 \
    --nn-model-filename ~/icefall/egs/wenetspeech/ASR/pruned_transducer_stateless2/exp/cpu_jit.pt \
    --token-filename ~/icefall/egs/wenetspeech/ASR/data/lang_char/tokens.txt

```
注：在上述指令的参数中，port 为6006，这里的端口也不是固定的，读者可以根据自己的实际情况进行修改，如6007等。但是，修改本端口的同时，必须要在 `sherpa/bin/web/js` 中对 `offline_record.js` 和 `streaming_record.js`中的端口进行同步修改，以保证 web 的数据和 server 的数据可以互通。

与此同时，我们还需要在服务器终端另开一个窗口开启 web 网页端服务，指令如下：
```bash
cd sherpa/bin/web
python3 -m http.server 6008
```

#### 本地 web 端测试使用

在服务器端运行相关功能的调用指令后，为了有更好的 ASR 交互体验，我们还需要将服务器端的 web 网页端服务进行本地化，所以使用 ssh 来连接本地端口和服务器上的端口：
```bash
ssh -R 6006:localhost:6006 -R 6008:localhost:6008 local_username@local_ip
```

接下来，我们可以在本地浏览器的网址栏输入：`localhost:6008`，我们将可以看到如下页面：
 - ![next-gen Kaldi web demo](https://github.com/k2-fsa/next-gen-kaldi-wechat/raw/master/pic/pic_lms/next-gen-kaldi-web-demo.png)

我们选择 `Offline-Record`，并打开麦克风，即可录音识别了。笔者的一个识别结果如下图所示：
 - ![a-picture-for-offline-asr](https://github.com/k2-fsa/next-gen-kaldi-wechat/raw/master/pic/pic_lms/offline-asr.png)

到这里，从数据准备和处理、模型训练和测试、服务端部署演示等三步就基本完成了。

新一代 Kaldi 语音识别开源框架还在快速地迭代和发展之中，本文所展示的只是其中极少的一部分内容，笔者在本文中也只是粗浅地概述了它的部分使用流程，更多详细具体的细节，希望读者能够自己去探索和发现。

## 总结
在本文中，笔者试图以 WenetSpeech 的 pruned transducer stateless2 recipe 构建、训练、部署的全流程为线索，贯通 k2、icefall、lhotse、sherpa四个独立子项目, 将新一代 Kaldi 框架的数据准备和处理、模型训练和测试、服务端部署演示等流程一体化地全景展示出来，形成一个简易的教程，希望能够更好地帮助读者认识和使用新一代 Kaldi 语音识别开源框架，真正做到上手即用。
