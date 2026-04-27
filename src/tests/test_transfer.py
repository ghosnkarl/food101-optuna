import torch
from omegaconf import OmegaConf
from src.models.transfer import TransferModel, build_transfer_model, _BACKBONE_REGISTRY

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    x = torch.randn(2, 3, 224, 224, device=device)

    # --- Test 1: build from config (resnet18) ---
    cfg = OmegaConf.load("configs/model/resnet18.yaml")
    model = build_transfer_model(cfg).to(device)
    out = model(x)
    assert out.shape == (2, 101), f"Expected (2, 101), got {out.shape}"
    print(f"[Test 1] build_transfer_model(resnet18) -> output: {out.shape}  OK")

    # --- Test 2: all backbones forward pass ---
    for backbone_name in _BACKBONE_REGISTRY:
        model = TransferModel(
            backbone=backbone_name,
            freeze_backbone=False,
            fine_tune_from_layer="none",
            fc_hidden_size=256,
            dropout_rate=0.3,
            pretrained=False,  # skip downloading weights for speed
        ).to(device)
        out = model(x)
        assert out.shape == (
            2,
            101,
        ), f"{backbone_name}: expected (2, 101), got {out.shape}"
        print(f"[Test 2] {backbone_name} -> output: {out.shape}  OK")

    # --- Test 3: freeze_backbone=True — only head params should require grad ---
    model = TransferModel(
        backbone="resnet18",
        freeze_backbone=True,
        fine_tune_from_layer="none",
        fc_hidden_size=256,
        dropout_rate=0.3,
        pretrained=False,
    )
    frozen = [
        n
        for n, p in model.backbone.named_parameters()
        if not p.requires_grad and not n.startswith("fc")
    ]
    trainable = [n for n, p in model.backbone.named_parameters() if p.requires_grad]
    assert len(frozen) > 0, "Expected frozen backbone params"
    assert all(
        n.startswith("fc") for n in trainable
    ), f"Non-head params are trainable: {trainable}"
    print(
        f"\n[Test 3] freeze_backbone=True: {len(frozen)} frozen, {len(trainable)} trainable (head only)  OK"
    )

    # --- Test 4: fine_tune_from_layer — layer3+ should be trainable ---
    model = TransferModel(
        backbone="resnet18",
        freeze_backbone=False,
        fine_tune_from_layer="layer3",
        fc_hidden_size=256,
        dropout_rate=0.3,
        pretrained=False,
    )
    frozen = [n for n, p in model.backbone.named_parameters() if not p.requires_grad]
    trainable = [n for n, p in model.backbone.named_parameters() if p.requires_grad]
    assert any("layer1" in n for n in frozen), "layer1 should be frozen"
    assert any("layer3" in n for n in trainable), "layer3 should be trainable"
    assert any("fc" in n for n in trainable), "head should be trainable"
    print(
        f"[Test 4] fine_tune_from_layer=layer3: {len(frozen)} frozen, {len(trainable)} trainable  OK"
    )

    # --- Test 5: param_groups returns correct LRs ---
    model = TransferModel(
        backbone="resnet18",
        freeze_backbone=False,
        fine_tune_from_layer="layer3",
        fc_hidden_size=256,
        dropout_rate=0.3,
        pretrained=False,
    )
    groups = model.param_groups(lr=1e-3, lr_multiplier_backbone=0.1)
    assert len(groups) == 2
    assert (
        abs(groups[0]["lr"] - 1e-4) < 1e-9
    ), f"Backbone LR should be 1e-4, got {groups[0]['lr']}"
    assert (
        abs(groups[1]["lr"] - 1e-3) < 1e-9
    ), f"Head LR should be 1e-3, got {groups[1]['lr']}"
    print(
        f"[Test 5] param_groups: backbone_lr={groups[0]['lr']}, head_lr={groups[1]['lr']}  OK"
    )

    # --- Test 6: unknown backbone raises ValueError ---
    try:
        TransferModel("unknown_backbone", False, "none", 256, 0.3)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"\n[Test 6] ValueError caught correctly: {e}")

    print("\nAll tests passed.")
