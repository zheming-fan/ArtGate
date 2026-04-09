# ArtGate

本仓库为论文 **"ArtGate: Injecting Fake Artifact Features into CLIP for AI-Generated Image Detection"** 的官方代码仓库。


## 环境配置

```bash
# 创建 conda 环境
conda create --name artgate python=3.10
conda activate artgate

# 安装 PyTorch（CUDA 版本）
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# 安装依赖
pip install scikit-learn tqdm ftfy regex kornia
pip install huggingface_hub requests tokenizers==0.21
pip install loralib timm opencv-python pandas tensorboard
pip install efficientnet_pytorch imageio scikit-image blobfile
pip install PyWavelets peft==0.13.2 tensorboardX

conda install -c conda-forge mpi4py

# 卸载环境中的 transformers（使用项目内置版本）
pip uninstall transformers
```

---

## 模型下载

从以下链接下载预训练权重：

**Google Drive：** [ArtGate Weights](https://drive.google.com/drive/folders/1jK0BC6_rRx9f9yWn8MJoGfQYLpRy7e9T?usp=sharing)

需要下载的文件：

| 文件名 | 说明 |
|---|---|
| `freq_progan.pth` | 训练阶段1: 频域伪影分支 (ResNet-50) 权重 |
| `model_clip_progan.pth` | 训练阶段2: 微调后的 CLIP 骨干网络权重 |
| `model_artgate_progan.pth` | 训练阶段3: ArtGate 主模型权重 |

此外，还需要下载 CLIP 预训练模型（`openai/clip-vit-large-patch14`）：

```bash
# 通过 huggingface_hub 下载
python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/clip-vit-large-patch14', local_dir='./pretrained/clip-vit-large-patch14')"
```

---

## 模型文件结构

将所有权重文件按如下结构放置：

```
ArtGate/
├── weights/
│   ├── model_artgate_progan.pth
│   ├── model_clip_progan.pth
│   └── freq_progan.pth
└── pretrained/
    └── clip-vit-large-patch14/   # CLIP 基础模型
```

下载完成后，修改 `ArtGate_model.py` 中的模型路径以匹配本地路径：

```python
# ArtGate_model.py
self.model = CLIPModel.from_pretrained('./pretrained/clip-vit-large-patch14')
clip_state_dict = torch.load('./weights/model_clip_progan.pth', map_location='cpu')
...
state_dict = torch.load('./weights/freq_progan.pth', map_location='cpu', weights_only=False)
```

同样修改 `ArtGate_eval.py` 中的 CLIP 路径：

```python
# ArtGate_eval.py
model = ArtGate_CLIP(name='./pretrained/clip-vit-large-patch14', num_classes=1)
```

---

## 数据集下载

本项目使用 **AIGCDetectionBenchMark** 数据集进行测试。

数据集下载地址：[AIGCDetectionBenchMark](https://github.com/Ekko-zn/AIGCDetectBenchmark)

数据集目录结构如下：

```
AIGCDetectionBenchMark/
└── test/
    ├── progan/
    ├── stylegan/
    ├── biggan/
    ├── cyclegan/
    ├── stargan/
    ├── gaugan/
    ├── stylegan2/
    ├── whichfaceisreal/
    ├── ADM/
    ├── Glide/
    ├── Midjourney/
    ├── stable_diffusion_v_1_4/
    ├── stable_diffusion_v_1_5/
    ├── VQDM/
    ├── wukong/
    ├── DALLE2/
    └── sd_xl/
```

每个子目录下需包含 `0_real/` 和 `1_fake/` 两个文件夹。

下载完成后，修改 `eval_config.py` 中的数据集路径：

```python
# eval_config.py
dataroot = '/path/to/AIGCDetectionBenchMark/test'
```

---

## 模型测试

确保环境、模型权重和数据集均已就绪后，运行以下命令进行测试：

```bash
python ArtGate_eval.py \
    --model_path ./weights/model_artgate_progan.pth
```

```bash
python ArtGate_eval.py \
    --model_path ./weights/model_artgate_progan.pth \
    --noise_type jpeg
```

> **快速测试提示：** 可以使用 `--max_test_image` 限制每个测试集的图片数量，以快速验证流程是否正常：
> ```bash
> python ArtGate_eval.py \
>     --model_path ./weights/model_artgate_progan.pth \
>     --max_test_image 100
> ```
