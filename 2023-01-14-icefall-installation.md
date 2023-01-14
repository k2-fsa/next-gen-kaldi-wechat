# Icefall Installation

> 本期文章带你在 linux 服务器上安装 icefall、k2、lhotse。 

## anaconda + python 3.10
```
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
bash Anaconda3-2022.10-Linux-x86_64.sh

source anaconda3/bin/activate
conda init

conda create --name icefall python=3.10
```

## pytorch 1.13.0 + cuda 11.6

```
conda activate icefall
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

## k2

-  需要 `cmake 3.11.0`, `GCC >= 7.0`

```
git clone https://github.com/k2-fsa/k2.git
cd k2
export K2_MAKE_ARGS="-j6"
python3 setup.py install
```

- 如果提示报错 ImportError: libpython3.10.so.1.0: cannot open shared object file: No such file or directory`，使用下面命令

  ```
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
  ```

## lhotse

```
pip install lhotse -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## icefall

```
git clone https://github.com/k2-fsa/icefall
cd icefall
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
export PYTHONPATH=<icefall_directory_path>$PYTHONPATH
```

- <icefall_directory_path> 是你的 icefall 工程文件夹目录