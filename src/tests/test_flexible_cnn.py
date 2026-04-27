import torch
from omegaconf import OmegaConf
from src.models.flexible_cnn import FlexibleCNN, build_flexible_cnn

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Test 1: build from config defaults ---
    cfg = OmegaConf.load("configs/model/flexible_cnn.yaml")
    model = build_flexible_cnn(cfg).to(device)
    print(
        f"\n[Test 1] Built from config: {cfg.n_layers} layers, filters={list(cfg.n_filters)}"
    )

    x = torch.randn(4, 3, 224, 224, device=device)
    out = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")  # expect [4, 101]
    assert out.shape == (4, 101), f"Expected (4, 101), got {out.shape}"

    # --- Test 2: varying n_layers ---
    for n_layers in [3, 4, 5, 6]:
        n_filters = [64] * n_layers
        kernel_sizes = [3] * n_layers
        model = FlexibleCNN(
            n_layers, n_filters, kernel_sizes, dropout_rate=0.3, fc_size=512
        ).to(device)
        out = model(x)
        assert out.shape == (
            4,
            101,
        ), f"n_layers={n_layers}: expected (4, 101), got {out.shape}"
        print(f"[Test 2] n_layers={n_layers} -> output: {out.shape}  OK")

    # --- Test 3: mixed kernel sizes ---
    model = FlexibleCNN(
        n_layers=4,
        n_filters=[32, 64, 128, 256],
        kernel_sizes=[3, 5, 3, 5],
        dropout_rate=0.2,
        fc_size=256,
    ).to(device)
    out = model(x)
    assert out.shape == (4, 101)
    print(f"\n[Test 3] Mixed kernel sizes -> output: {out.shape}  OK")

    # --- Test 4: assertion errors fire correctly ---
    try:
        FlexibleCNN(
            n_layers=3,
            n_filters=[64, 128],
            kernel_sizes=[3, 3, 3],
            dropout_rate=0.3,
            fc_size=256,
        )
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        print(f"\n[Test 4] AssertionError caught correctly: {e}")

    print("\nAll tests passed.")
