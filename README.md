# ArtGate

Official code for **“ArtGate: Injecting Fake Artifact Features into CLIP for AI-Generated Image Detection”**, accepted by IEEE Transactions on Multimedia.

## Environment setup

This project supports NVIDIA GPU inference only. Python 3.10 is recommended. Run the following commands in order.

### 1. Create the Conda environment

```bash
conda create -n artgate python=3.10 -y
conda activate artgate
python -m pip install --upgrade pip setuptools wheel
```

### 2. Install the CUDA build of PyTorch

Install the CUDA 12.4 build:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install project dependencies

Run the following command from the ArtGate repository root:

```bash
python -m pip install -r requirements.txt
```

### 4. Verify the environment

```bash
python -c "import torch, torchvision, transformers, peft, kornia, sklearn; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('transformers:', transformers.__version__); print('peft:', peft.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda)"
nvidia-smi
```

`CUDA available` should print `True`.

## Model weight

Download `model_artgate_progan.pth` from [Google Drive](https://drive.google.com/drive/folders/1jK0BC6_rRx9f9yWn8MJoGfQYLpRy7e9T?usp=sharing) and place it in the `weights/` directory.

## Dataset

Download [AIGCDetectBenchMark](https://github.com/Ekko-zn/AIGCDetectBenchmark). The path passed to `--dataset_root` should contain subdirectories such as `progan` and `stylegan`; each test subset should contain `0_real/` and `1_fake/`.

## Evaluation

```bash
python ArtGate_eval.py \
  --model_path ./weights/model_artgate_progan.pth \
  --dataset_root /path/to/AIGCDetectBenchMark/test
```

Use `--max_test_image 100` for a quick test on fewer images. On a multi-GPU machine, select a GPU with `--device cuda:1`. CSV output is written to `results/ArtGate/`.

## Citation

```bibtex
@article{fan2026artgate,
  title={ArtGate: Injecting Fake Artifact Features into CLIP for AI-Generated Image Detection},
  author={Fan, Zheming and Zhu, Guopu and Sun, Long and Ding, Feng and Zhang, Hongli and Wu, Ligang},
  journal={IEEE Transactions on Multimedia},
  year={2026},
  publisher={IEEE}
}
```
