"""
Training loop for AugMix on CIFAR-10/100.

Paper 4.1 setup:
  - SGD + Nesterov momentum, lr=0.1, cosine decay
  - WideResNet / AllConvNet: 100 epochs
  - DenseNet / ResNeXt: 200 epochs
  - weight_decay=0.0001 for Mixup, 0.0005 otherwise

Each forward pass produces three outputs (orig, aug1, aug2) — needed for JSD loss.

"""

import argparse
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .datasets import get_cifar_loaders
from .models import get_model

def jsd_loss(logits_orig, logits_aug1, logits_aug2, labels, lam=12.0):
    """AugMix loss: cross-entropy on clean image + λ * JSD consistency term."""
    p_orig = F.softmax(logits_orig, dim=1)
    p_aug1 = F.softmax(logits_aug1, dim=1)
    p_aug2 = F.softmax(logits_aug2, dim=1)
    
    M = torch.clamp((p_orig + p_aug1 + p_aug2) / 3, min=1e-7, max=1).log()
    JS = (F.kl_div(M, p_orig, reduction='batchmean') +
          F.kl_div(M, p_aug1, reduction='batchmean') +
          F.kl_div(M, p_aug2, reduction='batchmean')) / 3
          
    return F.cross_entropy(logits_orig, labels) + lam * JS 
    
    """
    AugMix loss: cross-entropy on clean image + λ * JSD consistency term.
    Loss = CE(p_orig, y) + λ * JS(p_orig; p_augmix1; p_augmix2)

    JS is computed as:
        M = (p_orig + p_augmix1 + p_augmix2) / 3
        JS = (KL(p_orig‖M) + KL(p_augmix1‖M) + KL(p_augmix2‖M)) / 3
        CE = cross_entropy(logits_orig, labels) applies softmax internally in PyTorch so we don't pass p_orig.
    Args:
        logits_orig:  (B, C) — model output for original images
        logits_aug1:  (B, C) — model output for first AugMix sample
        logits_aug2:  (B, C) — model output for second AugMix sample
        labels:       (B,)   — ground-truth class indices
        lam:          JSD loss weight (paper default = 12)

    Returns:
        scalar loss
    """
    
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    lam: float = 12.0
) -> dict[str, float]:
    
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in loader:
        optimizer.zero_grad()
        labels = labels.to(device)

        # Check if the dataset yielded the AugMix 3-tuple
        if isinstance(images, (tuple, list)) and len(images) == 3:
            # Concatenate for a single efficient forward pass
            images_all = torch.cat(images, dim=0).to(device)
            logits_all = model(images_all)
            
            # Split them back apart: size is batch_size
            batch_size = labels.size(0)
            logits_orig, logits_aug1, logits_aug2 = torch.split(logits_all, batch_size)
            
            loss = jsd_loss(logits_orig, logits_aug1, logits_aug2, labels, lam=lam)
            preds = logits_orig.argmax(dim=1)
        else:
            # Fallback for standard training (no AugMix)
            images = images.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=1)

        loss.backward()
        optimizer.step()
        scheduler.step() # Cosine schedule steps per batch

        total_loss += loss.item() * labels.size(0)
        total_correct += preds.eq(labels).sum().item()
        total_samples += labels.size(0)

    return {"loss": total_loss / total_samples, "acc": total_correct / total_samples}


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += preds.eq(labels).sum().item()
            total_samples += labels.size(0)

    return {"loss": total_loss / total_samples, "acc": total_correct / total_samples}


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Setting up {args.dataset} loaders (AugMix={args.augmix})...")
    train_loader, test_loader = get_cifar_loaders(
        dataset=args.dataset, 
        data_root=args.data_root, 
        batch_size=args.batch_size, 
        use_augmix=args.augmix
    )

    num_classes = 100 if args.dataset == "cifar100" else 10
    model = get_model(args.arch, num_classes).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True
    )
    
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    print(f"Beginning training for {args.epochs} epochs on {device}...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, device, lam=args.jsd_lambda)
        test_metrics = evaluate(model, test_loader, device)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:03d}/{args.epochs} [{epoch_time:.0f}s] | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Test Loss: {test_metrics['loss']:.4f} | "
              f"Test Acc: {test_metrics['acc']*100:.2f}%")

        is_best = test_metrics['acc'] > best_acc
        best_acc = max(test_metrics['acc'], best_acc)
        
        checkpoint = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }
        
        torch.save(checkpoint, args.output_dir / f"checkpoint_{args.arch}_latest.pt")
        if is_best:
            torch.save(checkpoint, args.output_dir / f"checkpoint_{args.arch}_best.pt")


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AugMix on CIFAR")
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--arch", default="wrn40_2", choices=["wrn40_2", "resnext29_32x4", "densenet100", "allconvnet"])
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--augmix", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--jsd-lambda", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    train(get_args())