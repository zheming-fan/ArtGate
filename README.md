# ArtGate

Official code for **[“ArtGate: Injecting Fake Artifact Features into CLIP for AI-Generated Image Detection”](https://ieeexplore.ieee.org/document/11523662)**, accepted by IEEE Transactions on Multimedia (2026).

> 中文文档：[README_zh.md](README_zh.md)

![Overview of ArtGate](figure.png)

## Environment setup

This project supports NVIDIA GPU inference only. Python 3.10 is recommended. Run the following commands in order.

### 1. Create the Conda environment

```bash
conda create -n artgate python=3.10 -y
conda activate artgate
python -m pip install --upgrade pip setuptools wheel
```

### 2. Install the CUDA build of PyTorch

Install the CUDA 11.8 build:

```bash
python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install project dependencies

Run the following command from the ArtGate repository root:

```bash
python -m pip install -r requirements.txt
```

## Model weight

Download `model_artgate_progan.pth` from [Google Drive](https://drive.google.com/drive/folders/1jK0BC6_rRx9f9yWn8MJoGfQYLpRy7e9T?usp=sharing) and place it in the `weights/` directory.

## Dataset

Download [AIGCDetectBenchMark](https://github.com/Ekko-zn/AIGCDetectBenchmark). The path passed to `--dataset_root` should contain subdirectories such as `progan` and `stylegan`; each test subset should contain `0_real/` and `1_fake/`.

## Evaluation

### Single-image inference

```bash
CUDA_VISIBLE_DEVICES=0 python ArtGate_eval.py \
  --model_path ./weights/model_artgate_progan.pth \
  --image_path figure.png
```

The command prints the predicted class, AI-generated probability, and raw model logit as JSON:

```json
{
  "image": "/home/ubuntu/2026/ArtGate/figure.png",
  "prediction": "real",
  "fake_probability": 0.17971085011959076,
  "logits": [-1.5183076858520508]
}
```

### Dataset evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python ArtGate_eval.py \
  --model_path ./weights/model_artgate_progan.pth \
  --dataset_root /path/to/AIGCDetectBenchMark/test
```

To evaluate with random JPEG compression:

```bash
CUDA_VISIBLE_DEVICES=0 python ArtGate_eval.py \
  --model_path ./weights/model_artgate_progan.pth \
  --dataset_root /path/to/AIGCDetectBenchMark/test \
  --noise_type jpeg
```

Use the `--max_test_image` option for a quick test:

```bash
CUDA_VISIBLE_DEVICES=0 python ArtGate_eval.py \
  --model_path ./weights/model_artgate_progan.pth \
  --dataset_root /path/to/AIGCDetectBenchMark/test \
  --max_test_image 100
```

On a multi-GPU machine, select a GPU with `--device cuda:1`. CSV output is written to `results/ArtGate/`.

## Acknowledgments

This project benefited from the implementation and ideas provided by the following open-source repositories. We gratefully acknowledge the authors for making their code and resources publicly available:

- [SAFE](https://github.com/Ouxiang-Li/SAFE)
- [AIGCDetectBenchmark](https://github.com/Ekko-zn/AIGCDetectBenchmark)

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@article{fan2026artgate,
  title={ArtGate: Injecting Fake Artifact Features into CLIP for AI-Generated Image Detection},
  author={Fan, Zheming and Zhu, Guopu and Sun, Long and Ding, Feng and Zhang, Hongli and Wu, Ligang},
  journal={IEEE Transactions on Multimedia},
  year={2026},
  publisher={IEEE}
}
```
