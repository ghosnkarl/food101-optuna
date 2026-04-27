from omegaconf import OmegaConf
from src.data.dataset import get_dataset_loaders

if __name__ == "__main__":
    print("Testing dataset loading and augmentation...")
    cfg = OmegaConf.load("configs/dataset/food101.yaml")
    main_config = OmegaConf.load("configs/config.yaml")

    train_loader, val_loader, test_loader = get_dataset_loaders(
        cfg=cfg, batch_size=32, augmentation_level="aggressive", device=main_config.device
    )

    images, labels = next(iter(train_loader))
    print(f"train  — images: {images.shape}, labels: {labels.shape}")

    images, labels = next(iter(val_loader))
    print(f"val    — images: {images.shape}, labels: {labels.shape}")

    images, labels = next(iter(test_loader))
    print(f"test   — images: {images.shape}, labels: {labels.shape}")

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
