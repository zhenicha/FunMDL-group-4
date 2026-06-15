"""
Reproduce Table 4 (last row): AugMix on CIFAR-10-C and CIFAR-100-C with WideResNet 40-2.

Evaluates the AugMix checkpoint on all 15 standard corruptions × 5 severity levels
and reports mCE alongside the paper's reported numbers for reference.

Run:
    uv run python experiments/table4_ablation.py \
        --checkpoint-cifar10 checkpoints/augmix/checkpoint_wrn40_2_best.pt \
        --checkpoint-cifar100 checkpoints/augmix_cifar100/checkpoint_wrn40_2_best.pt \
        --data-root data/
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.augmix.datasets import CIFAR_C, get_normalization
from src.augmix.models import get_model
from src.augmix.metrics import mean_corruption_error

CORRUPTIONS_15 = [
    "brightness", "contrast", "defocus_blur", "elastic_transform",
    "fog", "frost", "gaussian_noise", "glass_blur",
    "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
    "shot_noise", "snow", "zoom_blur",
]

# Paper's reported numbers for Table 4 for reference and checking if results were reproduced
PAPER_NUMBERS = {
    "Standard":                          {"cifar10c": 26.9, "cifar100c": 53.3},
    "AutoAugment*":                      {"cifar10c": 23.9, "cifar100c": 49.6},
    "Random AutoAugment*":               {"cifar10c": 17.0, "cifar100c": 43.6},
    "Random AutoAugment* + JSD":         {"cifar10c": 14.7, "cifar100c": 40.8},
    "AugmentAndMix (No JSD)":            {"cifar10c": 13.1, "cifar100c": 39.8},
    "AugMix (paper)":                    {"cifar10c": 11.2, "cifar100c": 35.9},
}


def load_model(checkpoint_path: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = get_model(checkpoint["arch"], num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def evaluate_cifar_c(model, data_root: Path, batch_size: int, device: torch.device, dataset: str = 'cifar10') -> float:
    """Evaluate mCE on all 15 standard corruptions × 5 severities."""
    errors = {c: [] for c in CORRUPTIONS_15}

    for corruption in CORRUPTIONS_15:
        for severity in range(1, 6):
            ds = CIFAR_C(data_root, corruption, severity, dataset=dataset)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
            correct, total = 0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
            errors[corruption].append(1 - correct / total)

    return mean_corruption_error(errors) * 100  # percentage


def print_table(cifar10c: float | None, cifar100c: float | None) -> None:
    print(f"\n{'Method':<35} {'CIFAR-10-C':>12} {'CIFAR-100-C':>13}")
    print("-" * 62)
    for method, vals in PAPER_NUMBERS.items():
        print(f"{method:<35} {vals['cifar10c']:>11.1f}% {vals['cifar100c']:>12.1f}%")
    print("-" * 62)
    c10 = f"{cifar10c:.1f}%" if cifar10c is not None else "N/A"
    c100 = f"{cifar100c:.1f}%" if cifar100c is not None else "N/A"
    print(f"{'AugMix (reproduced)':<35} {c10:>12} {c100:>13}")


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    cifar10c_mce, cifar100c_mce = None, None

    if args.checkpoint_cifar10 is not None:
        print(f"Evaluating on CIFAR-10-C...")
        model = load_model(args.checkpoint_cifar10, num_classes=10, device=device)
        cifar10c_mce = evaluate_cifar_c(model, args.data_root / "CIFAR-10-C", args.batch_size, device, dataset='cifar10')
        print(f"CIFAR-10-C mCE: {cifar10c_mce:.1f}%")

    if args.checkpoint_cifar100 is not None:
        print(f"Evaluating on CIFAR-100-C...")
        model = load_model(args.checkpoint_cifar100, num_classes=100, device=device)
        cifar100c_mce = evaluate_cifar_c(model, args.data_root / "CIFAR-100-C", args.batch_size, device, dataset='cifar100')
        print(f"CIFAR-100-C mCE: {cifar100c_mce:.1f}%")

    print_table(cifar10c_mce, cifar100c_mce)

    results = {"cifar10c_mce": cifar10c_mce, "cifar100c_mce": cifar100c_mce}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-cifar10", type=Path, default=None,
                   help="WRN-40-2 checkpoint trained on CIFAR-10")
    p.add_argument("--checkpoint-cifar100", type=Path, default=None,
                   help="WRN-40-2 checkpoint trained on CIFAR-100")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("checkpoints/table4_results.json"))
    p.add_argument("--batch-size", type=int, default=256)
    main(p.parse_args())
