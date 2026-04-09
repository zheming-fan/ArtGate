# ArtGate

This is the official code repository for the paper **"ArtGate: Injecting Fake Artifact Features into CLIP for AI-Generated Image Detection"**.

> Chinese README: [README_zh.md](README_zh.md)


## Environment Setup

**Python version required:** Python 3.10

```bash
# Create conda environment
conda create --name artgate python=3.10
conda activate artgate

# Install PyTorch (CUDA)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# Install dependencies
pip install scikit-learn tqdm ftfy regex kornia
pip install huggingface_hub requests tokenizers==0.21
pip install loralib timm opencv-python pandas tensorboard
pip install efficientnet_pytorch imageio scikit-image blobfile
pip install PyWavelets peft==0.13.2 tensorboardX

conda install -c conda-forge mpi4py

# Uninstall transformers (project uses its own bundled version)
pip uninstall transformers
```

---

## Model Download

Download the pretrained weights from:

**Google Drive:** [ArtGate Weights](https://drive.google.com/drive/folders/1jK0BC6_rRx9f9yWn8MJoGfQYLpRy7e9T?usp=sharing)

Files to download:

| File | Description |
|---|---|
| `freq_progan.pth` | Stage 1: Frequency-domain artifact branch (ResNet-50) weights |
| `model_clip_progan.pth` | Stage 2: Fine-tuned CLIP backbone weights |
| `model_artgate_progan.pth` | Stage 3: ArtGate main model weights |

You also need to download the CLIP base model (`openai/clip-vit-large-patch14`):

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/clip-vit-large-patch14', local_dir='./pretrained/clip-vit-large-patch14')"
```

---

## Model File Structure

Place all weight files as follows:

```
ArtGate/
├── weights/
│   ├── model_artgate_progan.pth
│   ├── model_clip_progan.pth
│   └── freq_progan.pth
└── pretrained/
    └── clip-vit-large-patch14/   # CLIP base model
```

After downloading, update the model paths in `ArtGate_model.py` to match your local paths:

```python
# ArtGate_model.py
self.model = CLIPModel.from_pretrained('./pretrained/clip-vit-large-patch14')
clip_state_dict = torch.load('./weights/model_clip_progan.pth', map_location='cpu')
...
state_dict = torch.load('./weights/freq_progan.pth', map_location='cpu', weights_only=False)
```

Also update the CLIP path in `ArtGate_eval.py`:

```python
# ArtGate_eval.py
model = ArtGate_CLIP(name='./pretrained/clip-vit-large-patch14', num_classes=1)
```

---

## Dataset Download

This project uses the **AIGCDetectionBenchMark** dataset for evaluation.

Dataset repository: [AIGCDetectionBenchMark](https://github.com/Ekko-zn/AIGCDetectBenchmark)

Expected directory structure:

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

Each subdirectory should contain `0_real/` and `1_fake/` folders.

After downloading, update the dataset path in `eval_config.py`:

```python
# eval_config.py
dataroot = '/path/to/AIGCDetectionBenchMark/test'
```

---

## Evaluation

Once the environment, model weights, and dataset are ready, run the following command:

```bash
python ArtGate_eval.py \
    --model_path ./weights/model_artgate_progan.pth
```

To test with JPEG noise augmentation:

```bash
python ArtGate_eval.py \
    --model_path ./weights/model_artgate_progan.pth \
    --noise_type jpeg
```

> **Quick test tip:** Use `--max_test_image` to limit the number of images per test set for a fast sanity check:
> ```bash
> python ArtGate_eval.py \
>     --model_path ./weights/model_artgate_progan.pth \
>     --max_test_image 100
> ```