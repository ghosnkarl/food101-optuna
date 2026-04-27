"""
Integration test: forward pass of all models on a real Food-101 batch.

Run with:
    python -m src.tests.test_models_food101

What this tests:
  - Food-101 data loads correctly (224x224, 101 classes)
  - FlexibleCNN produces [B, 101] output on a real batch
  - All 6 TransferModel backbones produce [B, 101] output on the same real batch
  - param_groups() returns exactly 2 groups (backbone + head) for each TransferModel

Uses batch_size=4 and pretrained=False (no download) for speed.
Uses the val loader so labels are clean integer class indices (no MixUp/CutMix).
"""

import torch
from omegaconf import OmegaConf

from src.data.dataset import get_dataset_loaders
from src.models.flexible_cnn import build_flexible_cnn
from src.models.transfer import TransferModel

BACKBONES = [
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "mobilenet_v3_small",
    "convnext_tiny",
    "vit_b_16",
]

BATCH_SIZE = 4


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load one real batch from Food-101 val split ───────────────────────────
    print("\nLoading Food-101 val loader (this may download the dataset)...")
    dataset_cfg = OmegaConf.load("configs/dataset/food101.yaml")

    _, val_loader, _ = get_dataset_loaders(
        cfg=dataset_cfg,
        batch_size=BATCH_SIZE,
        augmentation_level="minimal",
        device=str(device),
    )

    images, labels = next(iter(val_loader))
    images = images.to(device)

    print(
        f"Batch loaded — images: {tuple(images.shape)}, labels: {tuple(labels.shape)}"
    )
    assert images.shape == (
        BATCH_SIZE,
        3,
        224,
        224,
    ), f"Expected image shape ({BATCH_SIZE}, 3, 224, 224), got {tuple(images.shape)}"
    assert labels.shape == (
        BATCH_SIZE,
    ), f"Expected label shape ({BATCH_SIZE},), got {tuple(labels.shape)}"
    assert labels.max().item() < 101, "Label index out of range for 101 classes"
    print("  Data shape OK")

    # ── Test FlexibleCNN ──────────────────────────────────────────────────────
    print("\n[FlexibleCNN]")
    model_cfg = OmegaConf.load("configs/model/flexible_cnn.yaml")
    model = build_flexible_cnn(model_cfg).to(device)
    model.eval()

    with torch.inference_mode():
        out = model(images)

    assert out.shape == (
        BATCH_SIZE,
        101,
    ), f"Expected ({BATCH_SIZE}, 101), got {tuple(out.shape)}"
    print(f"  Output: {tuple(out.shape)}  OK")

    # ── Test all TransferModel backbones ──────────────────────────────────────
    print("\n[TransferModel — 6 backbones, pretrained=False]")
    for backbone in BACKBONES:
        model = TransferModel(
            backbone=backbone,
            freeze_backbone=False,
            fine_tune_from_layer="none",
            fc_hidden_size=256,
            dropout_rate=0.3,
            num_classes=101,
            pretrained=False,
        ).to(device)
        model.eval()

        with torch.inference_mode():
            out = model(images)

        assert out.shape == (
            BATCH_SIZE,
            101,
        ), f"{backbone}: Expected ({BATCH_SIZE}, 101), got {tuple(out.shape)}"

        groups = model.param_groups(lr=1e-3, lr_multiplier_backbone=0.1)
        assert (
            len(groups) == 2
        ), f"{backbone}: Expected 2 param groups, got {len(groups)}"

        print(
            f"  {backbone:<22} output: {tuple(out.shape)}  param_groups: {len(groups)}  OK"
        )

    print("\nAll tests passed.")
