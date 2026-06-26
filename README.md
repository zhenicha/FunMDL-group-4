# FunMDL-group-4 — AugMix Reproduction

Reproduction of **AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty** (Hendrycks et al., ICLR 2020) — [[paper]](https://arxiv.org/abs/1912.02781) [[original code]](https://github.com/google-research/augmix)

Full re-implementation from scratch. Experiments focus on CIFAR-10/100 (ImageNet out of scope due to compute power needed).

We specifically targeted the evaluation of standard training versus AugMix across four key architectures: AllConvNet, DenseNet-BC (k=12, d=100), WideResNet 40-2, and ResNeXt-29 32x4. We also recreated the original paper's visual comparisons (e.g., AugMix vs. CutOut, MixUp, and CutMix).

## Install

Requires [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/zhenicha/FunMDL-group-4.git
cd FunMDL-group-4
uv sync
```

## Structure

```
src/augmix/
├── augmentations.py   # AugMix ops + augment_and_mix()
├── loss.py            # Jensen-Shannon divergence consistency loss
├── models.py          # WideResNet, ResNeXt, DenseNet, AllConvNet
├── datasets.py        # CIFAR-10/100 + -C / -P variants
├── metrics.py         # mCE, mFP, RMS calibration error, Brier score
├── test.py            # Sanity check for pipeline
└── train.py           # Training loop

experiments/
├── figure1_augmix_visualization.py   # AugMix vs CutOut/MixUp/CutMix
├── table1_cifar_corruption.py        # CIFAR-10/100-C across architectures
└── table4_ablation.py                # ablation: diversity / JSD / mixing
```

## Usage

```bash
uv run python src/augmix/train.py --dataset cifar10 --arch wrn40_2
uv run python experiments/table1_cifar_corruption.py --dataset both
uv run python experiments/table4_ablation.py
```
