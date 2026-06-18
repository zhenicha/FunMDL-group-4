"""
Reproduce Table 1: CIFAR-10-C and CIFAR-100-C corruption error across architectures.

Architectures: AllConvNet, DenseNet-BC (k=12, d=100), WideResNet 40-2, ResNeXt-29 32x4
Saves results to JSON for use by figure5_bar_chart.py.

Only runs Standard vs. AugMix. Saves to JSON.


Usage:
    uv run python experiments/table1_cifar_corruption.py \
        --checkpoint-dir checkpoints/ --data-root data/ --dataset both
"""

import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.augmix.models import get_model
from src.augmix.datasets import get_cifar_loaders, CIFAR_C
from src.augmix.metrics import mean_corruption_error

def evaluate_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Returns classification error (0.0 to 1.0) on the given loader."""
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            total_correct += preds.eq(labels).sum().item()
            total_samples += labels.size(0)
    return 1.0 - (total_correct / total_samples)

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 100 if args.dataset == "cifar100" else 10
    
    architectures = ["allconvnet", "densenet100", "wrn40_2", "resnext29_32x4"]
    methods = ["standard", "augmix"]
    
    results = {}

    print(f"Loading clean {args.dataset} test data...")
    _, clean_test_loader = get_cifar_loaders(
        dataset=args.dataset, 
        data_root=args.data_root, 
        batch_size=args.batch_size, 
        use_augmix=False
    )

    for arch in architectures:
        results[arch] = {}
        for method in methods:
            ckpt_name = f"checkpoint_{arch}_{method}_best.pt"
            ckpt_path = args.checkpoint_dir / ckpt_name
            
            if not ckpt_path.exists():
                print(f"Skipping {arch} ({method}) - Checkpoint not found: {ckpt_name}")
                continue

            print(f"\nEvaluating {arch} ({method})...")
            
            model = get_model(arch, num_classes).to(device)
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])

            clean_error = evaluate_loader(model, clean_test_loader, device)
            print(f"  ↳ Clean Error: {clean_error * 100:.2f}%")

            # Mean Corruption Error (mCE)
            corruption_errors = {}
            cifar_c_dir = args.data_root / f"{args.dataset.upper()}-C"
            
            for corruption in CIFAR_C.CORRUPTIONS:
                errs_per_severity = []
                for severity in range(1, 6):
                    cifar_c = CIFAR_C(cifar_c_dir, corruption, severity=severity)
                    c_loader = DataLoader(cifar_c, batch_size=args.batch_size, shuffle=False, num_workers=4)
                    
                    err = evaluate_loader(model, c_loader, device)
                    errs_per_severity.append(err)
                
                corruption_errors[corruption] = errs_per_severity

            # Compute normalized mCE 
            mce = mean_corruption_error(corruption_errors)
            print(f"  ↳ Mean Corruption Error (mCE): {mce * 100:.2f}%")

            results[arch][method] = {
                "clean_error": clean_error * 100,
                "mce": mce * 100
            }

    output_file = Path("experiments") / f"table1_{args.dataset}_results.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nResults saved successfully to {output_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--batch-size", type=int, default=256)
    main(p.parse_args())
