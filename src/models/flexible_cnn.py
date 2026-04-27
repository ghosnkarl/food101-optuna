import torch
import torch.nn as nn
from omegaconf import DictConfig


def _make_conv_block(
    in_channels: int, out_channels: int, kernel_size: int
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class FlexibleCNN(nn.Module):
    """
    Configurable CNN for Food-101 (224x224 input, 101 classes).

    Each conv block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d(2,2)
    Spatial dims halve per block: 224 -> 112 -> 56 -> 28 -> 14 -> 7 -> 3 (for 3-6 layers)
    Final pooling: AdaptiveAvgPool2d((1,1)) -> flattened size is always n_filters[-1]
    """

    def __init__(
        self,
        n_layers: int,
        n_filters: list[int],
        kernel_sizes: list[int],
        dropout_rate: float,
        fc_size: int,
        num_classes: int = 101,
    ) -> None:
        super().__init__()

        assert (
            len(n_filters) == n_layers
        ), f"n_filters must have {n_layers} elements, got {len(n_filters)}"
        assert (
            len(kernel_sizes) == n_layers
        ), f"kernel_sizes must have {n_layers} elements, got {len(kernel_sizes)}"

        blocks: list[nn.Module] = []
        in_channels = 3
        for i in range(n_layers):
            blocks.append(_make_conv_block(in_channels, n_filters[i], kernel_sizes[i]))
            in_channels = n_filters[i]

        self.conv_blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(n_filters[-1], fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fc_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def build_flexible_cnn(cfg: DictConfig, num_classes: int = 101) -> FlexibleCNN:
    """Build a FlexibleCNN from a Hydra model config (fixed values, not search space)."""
    n_layers: int = cfg.n_layers
    n_filters = list(cfg.n_filters)
    kernel_sizes = list(cfg.kernel_sizes)

    # If lists are shorter than n_layers, repeat the last value to fill
    while len(n_filters) < n_layers:
        n_filters.append(n_filters[-1])
    while len(kernel_sizes) < n_layers:
        kernel_sizes.append(kernel_sizes[-1])

    return FlexibleCNN(
        n_layers=n_layers,
        n_filters=n_filters[:n_layers],
        kernel_sizes=kernel_sizes[:n_layers],
        dropout_rate=cfg.dropout_rate,
        fc_size=cfg.fc_size,
        num_classes=num_classes,
    )
