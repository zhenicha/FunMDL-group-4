import torch
from src.augmix.datasets import get_cifar_loaders
from src.augmix.models import get_model
from src.augmix.train import jsd_loss

def run_pipeline_test():
    print("1. Loading CIFAR-10 data with AugMix...")
    train_loader, _ = get_cifar_loaders('cifar10', data_root='./data', batch_size=4, use_augmix=True)
    
    print("2. Getting one batch...")
    images, labels = next(iter(train_loader))
    
    assert isinstance(images, list) or isinstance(images, tuple), "Dataset did not return a tuple of images!"
    assert len(images) == 3, f"Expected 3 images (orig, aug1, aug2), got {len(images)}"
    print(f"   Success: Got 3 image tensors of shape {images[0].shape}")

    print("3. Instantiating WideResNet (wrn40_2)...")
    model = get_model('wrn40_2', num_classes=10)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    images_all = torch.cat(images, dim=0).to(device)
    labels = labels.to(device)

    print("4. Testing Forward Pass...")
    logits_all = model(images_all)
    batch_size = labels.size(0)
    
    logits_orig, logits_aug1, logits_aug2 = torch.split(logits_all, batch_size)
    print(f"   Success: Forward pass yielded logits of shape {logits_orig.shape}")

    print("5. Testing JSD Loss Computation...")
    loss = jsd_loss(logits_orig, logits_aug1, logits_aug2, labels)
    print(f"   Success: Calculated Loss = {loss.item():.4f}")

    print("6. Testing Backward Pass...")
    loss.backward()
    print("Success")
    
    print("\nAll good")

if __name__ == "__main__":
    run_pipeline_test()
