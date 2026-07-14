# ArtGate

论文 **“ArtGate: Injecting Fake Artifact Features into CLIP for AI-Generated Image Detection”** 官方代码，已被 IEEE Transactions on Multimedia 接收。

## 环境配置

本项目只支持 NVIDIA GPU 推理，推荐使用 Python 3.10。下面的命令从创建环境开始，可以直接依次执行。

### 1. 创建 Conda 环境

```bash
conda create -n artgate python=3.10 -y
conda activate artgate
python -m pip install --upgrade pip setuptools wheel
```

### 2. 安装 CUDA 版 PyTorch

安装 CUDA 12.4 版本：

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 3. 安装项目依赖

在 ArtGate 仓库根目录执行：

```bash
python -m pip install -r requirements.txt
```



### 4. 验证环境

```bash
python -c "import torch, torchvision, transformers, peft, kornia, sklearn; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('transformers:', transformers.__version__); print('peft:', peft.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda)"
nvidia-smi
```

其中 `CUDA available` 应输出 `True`。

## 模型权重

从 [Google Drive](https://drive.google.com/drive/folders/1jK0BC6_rRx9f9yWn8MJoGfQYLpRy7e9T?usp=sharing) 下载 `model_artgate_progan.pth`，放到 `weights/` 目录即可。



## 数据集

下载 [AIGCDetectBenchMark](https://github.com/Ekko-zn/AIGCDetectBenchmark)。`--dataset_root` 指向包含 `progan`、`stylegan` 等子目录的路径；每个测试子集内应包含 `0_real/` 和 `1_fake/`。

## 评测

```bash
python ArtGate_eval.py \
  --model_path ./weights/model_artgate_progan.pth \
  --dataset_root /path/to/AIGCDetectBenchMark/test
```

可用 `--max_test_image 100` 缩小测试范围进行快速测试；多卡机器可通过 `--device cuda:1` 指定显卡。CSV 结果保存在 `results/ArtGate/`。


## 引用

```bibtex
@article{fan2026artgate,
  title={ArtGate: Injecting Fake Artifact Features into CLIP for AI-Generated Image Detection},
  author={Fan, Zheming and Zhu, Guopu and Sun, Long and Ding, Feng and Zhang, Hongli and Wu, Ligang},
  journal={IEEE Transactions on Multimedia},
  year={2026},
  publisher={IEEE}
}
```
